#![deny(unsafe_op_in_unsafe_fn)]

use std::borrow::Borrow;
use std::collections::hash_map::Entry::{Occupied, Vacant};
use std::collections::{hash_map, HashMap};
#[cfg(feature = "full_validation")]
use std::fmt::Debug;
use std::hash::Hash;
use std::mem::{replace, swap};
use std::ops::{Deref, DerefMut};
use std::ptr::null_mut;

use widders::types::CountedIter;
pub use widders::types::Ratio;

#[cfg(feature = "full_validation")]
use crate::node::{validate_tree, MutableBinaryNode, ViewingBinaryTreeIter};
use crate::node::{
    Consumable, Height, InsertDuplicates, KdBoxNodeParent, KdBuildableNode, KdInsertable, KdNode,
    Ownable, Parent, ScapegoatKdNode, Statistic, Tree, TreeHeightBound, Value, WithHeight,
};
use crate::search::{AdhocKdSearcher, KdSearchable, KdSearcher};
use crate::value::{KdValue, KdValuePlus, ValueStatisticPlus};

mod node;
pub mod search;
pub mod value;

struct NewKdInsert<'a, V, Node> {
    value: V,
    created: &'a mut *mut Node,
}

impl<'a, V, Node> KdInsertable<Node> for NewKdInsert<'a, V, Node>
where
    V: KdValue,
    Node: KdNode<Value = V>,
{
    fn value(&self) -> &V {
        &self.value
    }

    fn swap_value_with(&mut self, _receiving_node: *mut Node, val: &mut V) {
        swap(&mut self.value, val)
    }

    fn node(self, dim: usize) -> Node::Ownership {
        let mut node = Node::new(self.value, dim);
        *self.created = node.deref_mut();
        node
    }
}

pub struct KdValueMapImpl<K, V, Node, Balance: TreeHeightBound = Ratio<5, 4>>
where
    V: KdValue,
    Node: KdNode<Value = KdValuePlus<V, K>>,
{
    tree: Node::Tree,
    items: HashMap<K, *mut Node>,
    balance: Balance,
}

pub type KdValueMap<K, V, Stat = (), Balance = Ratio<5, 4>> = KdValueMapImpl<
    K,
    V,
    KdBoxNodeParent<KdValuePlus<V, K>, WithHeight<ValueStatisticPlus<Stat>>>,
    Balance,
>;

impl<K, V, Node, Balance> KdValueMapImpl<K, V, Node, Balance>
where
    K: Clone + Eq + Hash,
    V: KdValue,
    Node: KdNode<Value = KdValuePlus<V, K>>,
    Balance: TreeHeightBound,
{
    pub fn new() -> Self
    where
        Balance: Default,
    {
        Default::default()
    }

    pub fn with_balance(balance: Balance) -> Self {
        Self {
            balance,
            ..Default::default()
        }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn clear(&mut self) {
        self.tree.take();
        self.items.clear();
    }

    pub fn contains_key(&mut self, key: &K) -> bool {
        self.items.contains_key(key)
    }

    pub fn keys(&self) -> impl Iterator<Item = &K> {
        self.items.keys()
    }

    pub fn into_keys(self) -> impl Iterator<Item = K> {
        self.items.into_keys()
    }

    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.into_iter().map(|(_, v)| v)
    }
}

impl<K, V, Node, Balance> KdValueMapImpl<K, V, Node, Balance>
where
    K: Clone + Eq + Hash,
    V: KdValue,
    Node: KdNode<Value = KdValuePlus<V, K>>,
    Node::Ownership: Consumable<Value = KdValuePlus<V, K>>,
    Balance: TreeHeightBound,
{
    pub fn drain(self) -> impl Iterator<Item = (K, V)> {
        CountedIter::new(
            self.tree.into_iter().map(|node| {
                let KdValuePlus { val: v, plus: k } = node.consume();
                (k, v)
            }),
            self.items.len(),
        )
    }

    pub fn into_values(self) -> impl Iterator<Item = V> {
        CountedIter::new(
            self.tree.into_iter().map(|node| node.consume().val),
            self.items.len(),
        )
    }
}

impl<K, V, Stat, Node, Balance> KdValueMapImpl<K, V, Node, Balance>
where
    K: Clone + Eq + Hash,
    V: KdValue,
    Stat: Height,
    Node: KdNode<Value = KdValuePlus<V, K>> + Statistic<Stat = Stat>,
    Balance: TreeHeightBound,
{
    fn validate_height(&self) {
        let population: usize;
        let actual_height: isize;
        let max_height: isize;
        debug_assert!(
            {
                population = self.len();
                actual_height = self
                    .tree
                    .as_ref()
                    .map_or(-1, |root| root.stat().height() as isize);
                max_height = self.balance.height_budget(population) as isize;
                actual_height <= max_height + 1
            },
            "tree with {population} nodes has height {actual_height} which is more than 1 over the \
            allowed {max_height}"
        );
    }
}

impl<K, V, Stat, Node, Balance> KdValueMapImpl<K, V, Node, Balance>
where
    K: Clone + Eq + Hash,
    V: KdValue,
    Stat: Height,
    Node: KdNode<Value = KdValuePlus<V, K>> + Statistic<Stat = Stat>,
    Balance: TreeHeightBound,
{
    fn validate(&self) {
        self.validate_height();
    }
}

impl<K, V, Stat, Node, Balance> KdValueMapImpl<K, V, Node, Balance>
where
    K: Clone + Eq + Hash,
    V: KdValue,
    Stat: Height,
    Node: ScapegoatKdNode<Value = KdValuePlus<V, K>> + Parent + Statistic<Stat = Stat>,
    <Node as Ownable>::Ownership: KdInsertable<Node> + Consumable<Value = KdValuePlus<V, K>>,
    Balance: TreeHeightBound,
{
    /// When a node's value is removed from the tree, the node itself is often not removed; instead
    /// another node is removed and its value swapped into the target node. When this happens we
    /// have to ensure the pointers in `items` are updated for the swapped value still in the tree.
    ///
    /// SAFETY: `actually_removed` is a still-owned `Node` that was just removed from the tree by
    /// removing its value with `Node::remove_node(removal_target)`; thus, `removal_target` points
    /// either to a still live node inside the tree, or to the same node as `actually_removed`.
    unsafe fn track_removal_swap(
        &mut self,
        removal_target: *mut Node,
        actually_removed: *const Node,
    ) {
        if actually_removed != removal_target {
            // The node we triggered the removal of now has a new value. Track that key in items.
            // SAFETY: `removal_target` is still in the tree. No references into the tree exist.
            let node_new_key = &unsafe { &*removal_target }.value().plus;
            // SAFETY: Because we just found this key inside the tree, its corresponding entry must
            // exist in items.
            let new_key_ptr = unsafe { self.items.get_mut(node_new_key).unwrap_unchecked() };
            // The key there used to live in the removed node.
            debug_assert!(actually_removed == *new_key_ptr);
            // Associate that key with the node it's in now.
            *new_key_ptr = removal_target;
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        let current_size = self.len();
        match self.items.entry(key) {
            Vacant(entry) => {
                let mut created_node: *mut Node = null_mut();
                Node::tree_insert(
                    &mut self.tree,
                    self.balance,
                    current_size + 1,
                    NewKdInsert {
                        value: KdValuePlus {
                            val: value,
                            plus: entry.key().clone(),
                        },
                        created: &mut created_node,
                    },
                    InsertDuplicates,
                );
                entry.insert(created_node);
                self.validate();
                None
            }
            Occupied(mut entry) => {
                let removal_target = *entry.get();
                // SAFETY: No references into the tree exist.
                let mut reinsert = unsafe { Node::remove_node(removal_target) }
                    // SAFETY: We know at least one item is in the tree because we found a
                    // corresponding value in `items`. Furthermore, `remove_node` only returns None
                    // when the node to be removed is the sole item in its tree, which means it
                    // lives in `head`.
                    .unwrap_or_else(|| unsafe { self.tree.take().unwrap_unchecked() });
                // Fix up the pointer for this value's node in case the value was swapped
                *entry.get_mut() = reinsert.deref_mut();
                // Fix up the removal-target node too
                // SAFETY: We just did the removal.
                unsafe {
                    self.track_removal_swap(removal_target, reinsert.deref());
                }
                // If the whole tree is still unbalanced after a removal, rebuild one of its deepest
                // leaves.
                self.maybe_rebuild_after_removal();
                // Swap the k-d value
                let old_value = replace(&mut reinsert.value_mut().val, value);
                Node::tree_insert(
                    &mut self.tree,
                    self.balance,
                    current_size,
                    reinsert,
                    InsertDuplicates,
                );
                self.validate();
                Some(old_value)
            }
        }
    }

    pub fn try_insert(&mut self, key: K, value: V) -> Result<(), &V> {
        let current_size = self.len();
        match self.items.entry(key) {
            Vacant(entry) => {
                let mut created_node: *mut Node = null_mut();
                Node::tree_insert(
                    &mut self.tree,
                    self.balance,
                    current_size + 1,
                    NewKdInsert {
                        value: KdValuePlus {
                            val: value,
                            plus: entry.key().clone(),
                        },
                        created: &mut created_node,
                    },
                    InsertDuplicates,
                );
                entry.insert(created_node);
                self.validate();
                Ok(())
            }
            // SAFETY: No other references into the tree exist.
            Occupied(entry) => Err(&<Node as Value>::value(unsafe { &**entry.get() }).val),
        }
    }

    fn maybe_rebuild_after_removal(&mut self) {
        Node::tree_maybe_rebuild_deepest_leaf(
            self.balance.height_budget(self.len()),
            &mut self.tree,
            self.balance,
        );
        self.validate();
    }

    // TODO(widders): instead of borrowable to K, make the lookup key any borrowable type of K like
    //  HashMap does it. same for contains etc.
    pub fn remove<BK: Borrow<K>>(&mut self, key: BK) -> Option<V> {
        self.items.remove(key.borrow()).map(|removal_target| {
            // SAFETY: No references into the tree exist.
            let removed_node = unsafe { Node::remove_node(removal_target) }
                // SAFETY: We know at least one item is in the tree because we found a corresponding
                // value in `items`. Furthermore, `remove_node` only returns None when the node to
                // be removed is the sole item in its tree, which means it lives in `head`.
                .unwrap_or_else(unsafe { || self.tree.take().unwrap_unchecked() });
            // SAFETY: We just did the removal.
            unsafe {
                self.track_removal_swap(removal_target, removed_node.deref());
            }
            let removed_val = removed_node.consume().val;
            // If the whole tree is unbalanced after a shrink, rebuild one of its deepest leaves.
            self.maybe_rebuild_after_removal();
            removed_val
        })
    }

    // TODO(widders): insertable entries would be nice
}

impl<K, V, Stat, Node, Balance> KdValueMapImpl<K, V, Node, Balance>
where
    V: KdValue,
    Stat: Default,
    Node: KdSearchable<KdValuePlus<V, K>, WithHeight<Stat>>
        + KdNode<Value = KdValuePlus<V, K>>
        + Statistic<Stat = WithHeight<Stat>>,
    Balance: TreeHeightBound,
{
    pub fn search_values_with<S>(&self, searcher: &mut S)
    where
        S: KdSearcher<KdValuePlus<V, K>, Stat>,
    {
        let mut adapter = AdhocKdSearcher {
            searcher,
            visitor: |s: &mut S, v: &KdValuePlus<V, K>| s.visit(v),
            guide: |s: &S, plane: &V::Dimension, dim: usize, stat: &WithHeight<Stat>| {
                s.guide_search(plane, dim, &stat.stat)
            },
            recheck: |searcher: &S| searcher.recheck(),
        };
        if let Some(root) = self.tree.as_ref() {
            root.search_with(&mut adapter)
        }
    }

    pub fn search_values<S>(&self, mut searcher: S) -> S
    where
        S: KdSearcher<KdValuePlus<V, K>, Stat>,
    {
        self.search_values_with(&mut searcher);
        searcher
    }
}

impl<K, V, Node, Balance> Default for KdValueMapImpl<K, V, Node, Balance>
where
    V: KdValue,
    Node: KdNode<Value = KdValuePlus<V, K>>,
    Balance: TreeHeightBound + Default,
{
    fn default() -> Self {
        Balance::validate();
        Self {
            tree: Default::default(),
            items: Default::default(),
            balance: Default::default(),
        }
    }
}

impl<K, V, Node, Balance> FromIterator<(K, V)> for KdValueMapImpl<K, V, Node, Balance>
where
    K: Clone + Eq + Hash,
    V: KdValue,
    Node: KdBuildableNode<Value = KdValuePlus<V, K>>,
    <Node as Ownable>::Ownership: KdInsertable<Node>,
    Balance: TreeHeightBound,
{
    fn from_iter<Is>(items: Is) -> Self
    where
        Is: IntoIterator<Item = (K, V)>,
    {
        let mut res: Self = Default::default();
        let items_building: Vec<_> = items
            .into_iter()
            .map(|(k, val)| KdValuePlus { plus: k, val })
            .collect();
        let mut nodes_building: Vec<Node::Ownership> = Vec::new();
        // Go over the key/values in reverse so we will only add the last for each duplicated key.
        for val in items_building.into_iter().rev() {
            match res.items.entry(val.plus.clone()) {
                Occupied(_) => {}
                Vacant(entry) => {
                    let mut node = Node::new(val, 0);
                    entry.insert(node.deref_mut());
                    nodes_building.push(node);
                }
            }
        }
        Node::build_from(nodes_building, 0).map(|new_root| res.tree.replace(new_root));
        res
    }
}

#[cfg(feature = "full_validation")]
impl<K, V, Node, Balance> KdValueMapImpl<K, V, Node, Balance>
where
    K: Eq + Hash,
    V: KdValue,
    Node: ScapegoatKdNode<Value = KdValuePlus<V, K>> + MutableBinaryNode,
    Node::Stat: Height + Eq + Debug,
    Balance: TreeHeightBound + Default,
{
    pub fn fully_validate(&mut self) {
        // Check all invariants inside the tree
        validate_tree(&mut self.tree);
        // Check that all nodes in the tree have an entry in items
        let mut count = 0usize;
        for node in ViewingBinaryTreeIter::new(&self.tree) {
            assert!(self
                .items
                .get(&node.value().plus)
                .is_some_and(|ptr| *ptr == node as *const Node as *mut Node));
            count += 1;
        }
        // Check that there are no entries in items that don't have a matching node in the tree.
        // If any duplicate keys existed in the tree, we would have found an entry with a
        // mismatched pointer in the prior loop, so we can be confident that the count of nodes is
        // equal to the count of unique keys at this point.
        assert_eq!(count, self.items.len());
    }
}

pub struct KdValueIterator<'a, K, V, Node>
where
    K: 'a,
    V: 'a + KdValue,
    Node: Value<Value = KdValuePlus<V, K>>,
{
    iter: hash_map::Iter<'a, K, *mut Node>,
}

impl<'a, K, V, Node> Iterator for KdValueIterator<'a, K, V, Node>
where
    V: KdValue,
    Node: Value<Value = KdValuePlus<V, K>>,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter
            .next()
            // SAFETY: Each entry in the items table points to a live node in the tree, which cannot
            // change because we hold a reference.
            .map(|(key, node_ptr)| (key, &unsafe { &*(*node_ptr as *const Node) }.value().val))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a, K, V, Node, Balance> IntoIterator for &'a KdValueMapImpl<K, V, Node, Balance>
where
    K: 'a + Clone + Eq + Hash,
    V: 'a + KdValue,
    Node: KdNode<Value = KdValuePlus<V, K>>,
    Balance: TreeHeightBound,
{
    type Item = (&'a K, &'a V);
    type IntoIter = KdValueIterator<'a, K, V, Node>;

    fn into_iter(self) -> KdValueIterator<'a, K, V, Node> {
        KdValueIterator {
            iter: self.items.iter(),
        }
    }
}

// TODO(widders): MORE FASTER: stem-and-leaf tree
// TODO(widders): plan to eliminate Ord requirement so we can use raw floating points, which is
//  probably good for vectorization
// TODO(widders): nonduplicate vs duplicate trees, parentless nodes
// TODO(widders): -> KdKeyMap
// TODO(widders): -> KdSet
// TODO(widders): cubes and bounding-box metrics
// TODO(widders): move tree_*() impls into trees? maybe not
