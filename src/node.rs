use std::borrow::Borrow;
use std::cmp::max;
use std::cmp::Ordering::{Equal, Greater, Less};
use std::fmt::Debug;
use std::mem::swap;
use std::ops::{Deref, DerefMut};
use std::ptr::null_mut;
use aliasable::boxed::AliasableBox;

use crate::node::ScapegoatChangeOutcome::{Balanced, Rebuilding};
use crate::node::ScapegoatPop::{Popped, RemoveThis};
use crate::node::StatUpdate::{NoChange, Updated};
use crate::search::KdSearchGuide::{SearchChildren, Skip};
use crate::search::OnlyOrBoth::{Both, Only};
use crate::search::{KdSearchable, KdSearcher, StopSearch};
use crate::value::{CycleDim, KdValue, TotalOrd};
use crate::RatioT;

pub trait TreeHeightBound: Default + Copy {
    /// Entry point to validate that the type does not place an impossible constraint on the tree's
    /// height.
    fn validate();

    fn height_budget(&self, size: usize) -> u32;

    fn tree_is_balanced(&self, height: u32, size: usize) -> bool {
        height <= self.height_budget(size)
    }
}

impl<R: RatioT> TreeHeightBound for R {
    fn validate() {
        assert!(Self::VAL > 1.0, "balance factor must be greater than 1.0");
    }

    fn height_budget(&self, size: usize) -> u32 {
        1 + ((size as f32).log2() * Self::VAL32) as u32
    }
}

pub trait ValueStatistic<V>: Clone + Eq + Default {
    fn combine<'a, Vs, Ss>(values: Vs, stats: Ss) -> Self
    where
        V: 'a,
        Self: 'a,
        Vs: IntoIterator<Item = &'a V>,
        <Vs as IntoIterator>::IntoIter: Clone,
        Ss: IntoIterator<Item = &'a Self>,
        <Ss as IntoIterator>::IntoIter: Clone;
}

impl<V> ValueStatistic<V> for () {
    fn combine<'a, Vs, Ss>(_: Vs, _: Ss) -> Self
    where
        V: 'a,
    {
    }
}

pub trait Height {
    fn height(&self) -> u32;
}

pub trait Ownable {
    type Ownership: DerefMut<Target = Self>;
}

pub trait IntoOwned: DerefMut
where
    Self::Target: Ownable,
{
    fn into_owned(self) -> <Self::Target as Ownable>::Ownership;
}

pub trait Tree: IntoIterator<Item = <Self::Node as Ownable>::Ownership> {
    type Node: Ownable;
    type Takeable<'a>: 'a + IntoOwned<Target = Self::Node>
    where
        Self: 'a;

    fn validate(&self);

    fn as_ref(&self) -> Option<&Self::Node>;

    fn get(&mut self) -> Option<Self::Takeable<'_>>;

    unsafe fn get_unchecked(&mut self) -> Self::Takeable<'_>;

    fn replace(
        &mut self,
        with: <Self::Node as Ownable>::Ownership,
    ) -> Option<<Self::Node as Ownable>::Ownership>;

    fn take(&mut self) -> Option<<Self::Node as Ownable>::Ownership>;

    fn into_owned(self) -> Option<<Self::Node as Ownable>::Ownership>;
}

pub trait Consumable {
    type Value;

    fn consume(self) -> Self::Value;
}

pub trait BinaryNode {
    fn left(&self) -> Option<&Self>;
    fn right(&self) -> Option<&Self>;
    fn is_leaf(&self) -> bool {
        self.left().is_none() && self.right().is_none()
    }
}

pub trait MutableBinaryNode: BinaryNode + Ownable {
    type Child<'a>: Tree<Node = Self>
    where
        Self: 'a;

    fn left_child(&mut self) -> Self::Child<'_>;
    fn right_child(&mut self) -> Self::Child<'_>;
    fn children(&mut self) -> [Self::Child<'_>; 2];
}

pub trait Parent {
    // TODO(widders): better model here that works for unboxed types as well? similar to Ownership
    fn parent(&self) -> *mut Self;
    /// Orphan this node, causing it to have no parent.
    fn orphan(&mut self);
}

pub trait Value {
    type Value;
    fn value(&self) -> &Self::Value;
    fn value_mut(&mut self) -> &mut Self::Value;
}

pub trait Statistic {
    type Stat: Default;
    fn stat(&self) -> &Self::Stat;
    fn stat_mut(&mut self) -> &mut Self::Stat;
}

pub enum StatUpdate {
    NoChange,
    Updated,
}

pub trait UpdateStat: Statistic {
    fn make_stat(&self) -> Self::Stat;
    fn update_stat(&mut self) -> StatUpdate;
}

pub trait KdNode: Ownable + Value + UpdateStat
where
    Self::Value: KdValue,
{
    type Tree: Tree<Node = Self> + Default;
    type Discriminant<'a>: Borrow<<Self::Value as KdValue>::Dimension>
    where
        Self: 'a;

    fn new(val: Self::Value, dim: usize) -> Self::Ownership;
    fn discriminant(&self, dim: usize) -> Self::Discriminant<'_>;
}

fn impl_search_binary<V, Stat, Node, S>(
    node: &Node,
    dim: usize,
    searcher: &mut S,
) -> Result<(), StopSearch>
where
    V: KdValue,
    Node: KdNode<Value = V> + Statistic<Stat = Stat> + BinaryNode,
    S: KdSearcher<V, Stat>,
{
    searcher.visit(node.value())?;
    if node.is_leaf() {
        return Ok(());
    }
    let (first, second, bothness, greater_first) =
        match searcher.guide_search(node.discriminant(dim), dim, node.stat()) {
            Skip => return Ok(()),
            SearchChildren(Less, bothness) => (node.left(), node.right(), bothness, false),
            SearchChildren(Equal, _) => (node.left(), node.right(), Both, false),
            SearchChildren(Greater, bothness) => (node.right(), node.left(), bothness, true),
        };
    match (first, second, bothness) {
        (Some(first_child), Some(second_child), bothness) => {
            let next_dim = V::next_dim(dim);
            impl_search_binary(first_child, next_dim, searcher)?;
            if searcher.recheck() {
                // Exit if the recheck indicates anything OTHER THAN a search covering the
                // second side from the original guide. We only continue if the new guide
                // either indicates Both or has chosen to search the opposite side first.
                match searcher.guide_search(node.discriminant(dim), dim, node.stat()) {
                    SearchChildren(_, Both) => { /* Don't exit: Searching both */ }
                    SearchChildren(new_side_first, Only) => {
                        if (new_side_first == Greater) == (greater_first) {
                            return Ok(());
                        }
                        /* Don't exit: Searching the other side */
                    }
                    Skip => return Ok(()),
                }
            } else if let Only = bothness {
                return Ok(()); // We aren't rechecking and won't look at the second child.
            }
            impl_search_binary(second_child, next_dim, searcher)
        }
        (Some(only_child), None, _) | (None, Some(only_child), Both) => {
            impl_search_binary(only_child, V::next_dim(dim), searcher)
        }

        (None, Some(_), Only) => Ok(()),
        (None, None, _) => unreachable!(), // We already checked this above.
    }
}

impl<V: KdValue, Stat, Node> KdSearchable<V, Stat> for Node
where
    Node: KdNode<Value = V> + Statistic<Stat = Stat> + BinaryNode,
{
    fn search_with<S>(&self, searcher: &mut S)
    where
        S: KdSearcher<V, Stat>,
    {
        let _ = impl_search_binary(self, 0, searcher);
    }
}

/// Separate trait to allow default implementations for build_from based only on topology
pub trait KdBuildableNode: KdNode
where
    Self::Value: KdValue,
{
    fn build_from<I>(items: Vec<I>, dim: usize) -> Option<Self::Ownership>
    where
        I: KdInsertable<Self>,
    {
        if items.is_empty() {
            None
        } else {
            Some(unsafe { Self::must_build_from(items, dim) })
        }
    }

    /// Build a tree from a set of items that must be non-empty.
    unsafe fn must_build_from<I>(items: Vec<I>, dim: usize) -> Self::Ownership
    where
        I: KdInsertable<Self>;
}

/// Binary node impl for kd tree building
impl<Node> KdBuildableNode for Node
where
    Node: KdNode + MutableBinaryNode,
    Node::Value: KdValue,
{
    unsafe fn must_build_from<I>(mut items: Vec<I>, dim: usize) -> Self::Ownership
    where
        I: KdInsertable<Self>,
    {
        debug_assert!(!items.is_empty());
        // Inner function does not have to repeatedly check items size
        // and allows us to pre-fill `begin`.
        fn build_from_recursive<Node, I>(
            items: &mut Vec<I>,
            begin: usize,
            dim: usize,
        ) -> Node::Ownership
        where
            Node: KdNode + MutableBinaryNode,
            Node::Value: KdValue,
            I: KdInsertable<Node>,
        {
            let next_dim = Node::Value::next_dim(dim);
            if begin + 1 == items.len() {
                // SAFETY: There must be an item in the vec, or that condition would not be true.
                let mut tree = unsafe { items.pop().unwrap_unchecked() }.node(next_dim);
                *tree.stat_mut() = Default::default();
                return tree;
            }
            let pivot = {
                let this_subtree = &mut items.as_mut_slice()[begin..];
                // When recursively building, the chosen midpoint biases rightward when the slice we
                // are building (from `begin` to the end of the vec) has an even number of items, such
                // that we bias left-heavy (including that all freshly built nodes with only one child
                // have only a left child).
                let pivot_in_subtree = this_subtree.len() / 2;
                order_stat::kth_by(this_subtree, pivot_in_subtree, |a, b| {
                    let (a_val, b_val) = (a.value(), b.value());
                    (a_val.get_dimension(dim).borrow(), a_val)
                        .total_cmp(&(b_val.get_dimension(dim).borrow(), b_val))
                });
                begin + pivot_in_subtree
            };
            // The values vec is rebuilt into a tree strictly from right to left, always removing
            // elements by popping them.
            let right = if pivot + 1 < items.len() {
                Some(build_from_recursive(items, pivot + 1, next_dim))
            } else {
                None
            };
            // SAFETY: At the top, `pivot` is some valid index. We recursed to deplete all the
            // items to the right of there, so it is now the last item in the vec.
            let mut tree = unsafe { items.pop().unwrap_unchecked() }.node(dim);
            debug_assert_eq!(items.len(), pivot);
            right.map(|right_child| tree.right_child().replace(right_child));
            if pivot > begin {
                tree.left_child()
                    .replace(build_from_recursive(items, begin, next_dim));
            }
            *tree.stat_mut() = tree.make_stat();
            tree
        }
        let result = build_from_recursive(&mut items, 0, dim);
        debug_assert!(items.is_empty());
        result
    }
}

// k-d node that can be part of a live, mutable tree.
pub trait MutableKdNode: KdBuildableNode
where
    Self::Value: KdValue,
{
    fn set_dim(&mut self, dim: usize);
}

pub trait KdInsertable<Node: Value + Ownable + ?Sized> {
    fn value(&self) -> &Node::Value;
    fn swap_value_with(&mut self, receiving_node: *mut Node, val: &mut Node::Value);
    fn node(self, dim: usize) -> Node::Ownership;
}

impl<V, Node> KdInsertable<Node> for Box<Node>
where
    V: KdValue,
    Node: MutableKdNode<Ownership = Box<Node>, Value = V>,
{
    fn value(&self) -> &V {
        <Node as Value>::value(self)
    }

    fn swap_value_with(&mut self, _receiving_node: *mut Node, val: &mut V) {
        swap(self.value_mut(), val)
    }

    fn node(mut self, dim: usize) -> Self {
        self.set_dim(dim);
        self
    }
}

impl<V, Node> KdInsertable<Node> for AliasableBox<Node>
where
    V: KdValue,
    Node: MutableKdNode<Ownership = AliasableBox<Node>, Value = V>,
{
    fn value(&self) -> &V {
        <Node as Value>::value(self)
    }

    fn swap_value_with(&mut self, _receiving_node: *mut Node, val: &mut V) {
        swap(self.value_mut(), val)
    }

    fn node(mut self, dim: usize) -> Self {
        self.set_dim(dim);
        self
    }
}

pub enum ScapegoatChangeOutcome<Ownership> {
    Balanced(StatUpdate),
    Rebuilding(Vec<Ownership>),
}

pub enum ScapegoatPop<Ownership> {
    RemoveThis,
    Popped(Ownership, StatUpdate),
}

pub enum InsertEqualResolutionFlag {
    InsertDuplicates,
    CancelInsert,
    SwapValue,
}

pub trait InsertEqualResolution {
    const METHOD: InsertEqualResolutionFlag;
}

pub struct InsertDuplicates;
impl InsertEqualResolution for InsertDuplicates {
    const METHOD: InsertEqualResolutionFlag = InsertEqualResolutionFlag::InsertDuplicates;
}

pub struct CancelInsert;
impl InsertEqualResolution for CancelInsert {
    const METHOD: InsertEqualResolutionFlag = InsertEqualResolutionFlag::CancelInsert;
}

pub struct SwapValue;
impl InsertEqualResolution for SwapValue {
    const METHOD: InsertEqualResolutionFlag = InsertEqualResolutionFlag::SwapValue;
}

pub trait ScapegoatKdNode: MutableKdNode
where
    Self::Value: KdValue,
{
    // TODO(widders): method on Tree? hmmm
    fn tree_insert<B, I, R>(
        tree: &mut Self::Tree,
        balance: B,
        new_tree_population: usize,
        ins: I,
        resolve_equal: R,
    ) where
        B: TreeHeightBound,
        I: KdInsertable<Self>,
        Self::Ownership: KdInsertable<Self>,
        R: InsertEqualResolution,
    {
        let Some(mut root) = tree.get() else {
            tree.replace(ins.node(0));
            return;
        };
        let gas = balance.height_budget(new_tree_population);
        let Rebuilding(mut collection) = root.insert(balance, 0, gas, ins, resolve_equal) else {
            return;
        };
        collection.push(root.into_owned());
        // SAFETY: We trivially have a non-empty collection because we just pushed the root into it.
        tree.replace(unsafe { Self::must_build_from(collection, 0) });
    }

    /// `balance` provides the height constraint for the tree when we are rebuilding subtrees. `dim`
    /// is the timension of self: its distance from root modulo the number of dimensions. `gas` is
    /// the maximum tolerated distance from the root for the insert, below which an insert should
    /// trigger an immediate rebuild.
    fn insert<B, I, R>(
        &mut self,
        balance: B,
        dim: usize,
        gas: u32,
        ins: I,
        resolve_equal: R,
    ) -> ScapegoatChangeOutcome<Self::Ownership>
    where
        B: TreeHeightBound,
        I: KdInsertable<Self>,
        R: InsertEqualResolution;

    // TODO(widders): method on Tree? hmmm
    fn tree_maybe_rebuild_deepest_leaf<Balance>(
        max_height: u32,
        tree: &mut Self::Tree,
        balance: Balance,
    ) where
        Self: Statistic,
        <Self as Statistic>::Stat: Height,
        Balance: TreeHeightBound,
        Self::Ownership: KdInsertable<Self>,
    {
        let Some(mut root) = tree.get() else { return; };
        if root.stat().height() <= max_height {
            return;
        }
        let Rebuilding(mut collection) = root.rebuild_deepest_leaf(balance, 0) else { return; };
        collection.push(root.into_owned());
        // SAFETY: We trivially have a non-empty collection because we just pushed the root into it.
        tree.replace(unsafe { Self::must_build_from(collection, 0) });
    }

    fn rebuild_deepest_leaf<Balance>(
        &mut self,
        balance: Balance,
        dim: usize,
    ) -> ScapegoatChangeOutcome<Self::Ownership>
    where
        Balance: TreeHeightBound;

    fn pop_deepest_leaf(&mut self) -> ScapegoatPop<Self::Ownership>;

    // TODO(widders): method on Tree? hmmm
    fn tree_remove_value<Vcmp>(tree: &mut Self::Tree, val: &Vcmp) -> Option<Self::Ownership>
    where
        Vcmp: KdValue<Dimension = <Self::Value as KdValue>::Dimension> + PartialOrd<Self::Value>,
    {
        let Some(mut root) = tree.get() else { return None };
        root.remove_value(val, 0).map(|pop| match pop {
            RemoveThis => root.into_owned(),
            Popped(popped, _) => popped,
        })
    }

    /// Finds and removes the node with the value equal to the given value.
    ///
    /// A `None` return value indicates the value was not found.
    ///
    /// `Some(RemoveThis)` signals that the callee itself can be removed from its owning location.
    /// When received from the root node, this means removing the last node in the tree.
    ///
    /// Otherwise, `Some(Popped(popped, ..))` will return the ownership of some node now removed
    /// from the tree and ready for reuse. This returned node is guaranteed to have been a leaf,
    /// with no children, a defaulted statistic, and the value that matched the one provided.
    fn remove_value<Vcmp>(
        &mut self,
        val: &Vcmp,
        dim: usize,
    ) -> Option<ScapegoatPop<Self::Ownership>>
    where
        Vcmp: KdValue<Dimension = <Self::Value as KdValue>::Dimension> + PartialOrd<Self::Value>;

    /// Removes the value in the node pointed to from the tree it resides in. Does not rebalance the
    /// tree; if the tree becomes unbalanced by a removal, call `rebuild_deepest_leaf_of` on the
    /// root's ownership.
    ///
    /// A `None` return value indicates that the node pointed to is the only value in its tree, with
    /// no children or parents. When received from the root node, this means removing the last node
    /// in the tree.
    ///
    /// `Some(popped)` will be the ownership of some node now removed from the tree and ready for
    /// reuse. This returned node is guaranteed to have been a leaf, with no children, a defaulted
    /// statistic, and the value that resided in the originally pointed-to node.
    ///
    /// SAFETY: The pointed-to node must exist in a tree. Never call this while any reference exists
    /// to any node in the tree.
    unsafe fn remove_node(node: *mut Self) -> Option<Self::Ownership>
    where
        Self: Parent;
}

fn subtree_height<Node>(edge: Option<&Node>) -> u32
where
    Node: Statistic,
    <Node as Statistic>::Stat: Height,
{
    edge.map_or(0, |child| child.stat().height() + 1)
}

impl<Node> ScapegoatKdNode for Node
where
    Node: MutableKdNode + MutableBinaryNode + Statistic,
    Node::Value: KdValue,
    <Self as Ownable>::Ownership: KdInsertable<Self>,
    <Node as Statistic>::Stat: Height + Eq,
{
    fn insert<Balance, I, R>(
        &mut self,
        balance: Balance,
        dim: usize,
        gas: u32,
        ins: I,
        resolve_equal: R,
    ) -> ScapegoatChangeOutcome<Self::Ownership>
    where
        Balance: TreeHeightBound,
        I: KdInsertable<Self>,
        R: InsertEqualResolution,
    {
        let compared = (ins.value().get_dimension(dim).borrow(), ins.value())
            .total_cmp(&(self.discriminant(dim).borrow(), self.value()));
        let go_left = match (compared, R::METHOD) {
            (Less, _) => true,
            (Greater, _) => false,
            // When inserting duplicates, insert into the shorter child subtree
            (Equal, InsertEqualResolutionFlag::InsertDuplicates) => {
                // True if the left side is not taller. In the event of a tie we insert into the
                // left side to bias left-heavy.
                subtree_height(self.right()) == self.stat().height()
            }
            (Equal, InsertEqualResolutionFlag::CancelInsert) => return Balanced(NoChange),
            (Equal, InsertEqualResolutionFlag::SwapValue) => {
                let mut swapping = ins;
                swapping.swap_value_with(self, self.value_mut());
                return Balanced(Updated);
            }
        };
        let [mut dest, sibling] = if go_left {
            self.children()
        } else {
            let [left, right] = self.children();
            [right, left]
        };
        let next_dim = Self::Value::next_dim(dim);
        let Some(mut node) = dest.get() else {
            dest.replace(ins.node(next_dim));
            drop((dest, sibling));
            return Balanced(self.update_stat());
        };
        if gas <= 1 {
            // We are out of gas; we can't insert as a child of this node, and a grandchild of this
            // node would be too deep to be balanced, so we must rebuild at least from `self`.
            // When creating the new node, dim doesn't matter here since the dim of the inserted
            // node will be reset during the rebuild anyway.
            let mut collection = vec![ins.node(0)];
            drop(node);
            collection.extend(dest.into_iter());
            collection.extend(sibling.into_iter());
            return Rebuilding(collection);
        }
        match node.insert(balance, next_dim, gas - 1, ins, resolve_equal) {
            Rebuilding(mut collection) => {
                let node_height = node.stat().height();
                collection.push(node.into_owned());
                if balance.tree_is_balanced(node_height, collection.len()) {
                    // Subtree is still balanced, collect sibling and continue
                    collection.extend(sibling.into_iter());
                    Rebuilding(collection)
                } else {
                    // Tree is re-buildable: rebuild it into the dest child.
                    // SAFETY: We trivially have a non-empty collection because we just pushed the
                    // dest child (`node`) into it.
                    dest.replace(unsafe { Node::must_build_from(collection, next_dim) });
                    drop((dest, sibling));
                    Balanced(self.update_stat())
                }
            }
            Balanced(NoChange) => Balanced(NoChange),
            Balanced(Updated) => {
                drop(node);
                drop((dest, sibling));
                Balanced(self.update_stat())
            }
        }
    }

    fn rebuild_deepest_leaf<Balance>(
        &mut self,
        balance: Balance,
        dim: usize,
    ) -> ScapegoatChangeOutcome<Self::Ownership>
    where
        Balance: TreeHeightBound,
    {
        let self_height = self.stat().height();
        if self_height <= 1 {
            // If this node has height 1, it has 1-2 children that are leaves; it is inherently
            // balanced, and its leaves are the deepest nodes. Start collecting here.
            // It shouldn't be possible to reach a leaf while traversing unless we erroneously
            // trigger a deepest-leaf-rebuild on a tree with only one member, but if we do the
            // correct thing to do would be to return Rebuilding(empty vec) anyway.
            return Rebuilding(
                self.children()
                    .into_iter()
                    .filter_map(Node::Child::into_owned)
                    .collect(),
            );
        }
        let next_dim = Self::Value::next_dim(dim);
        let [mut rebuilding_child, sibling] = if subtree_height(self.left()) == self_height {
            self.children()
        } else {
            debug_assert_eq!(subtree_height(self.right()), self_height);
            let [left, right] = self.children();
            [right, left]
        };
        // SAFETY: self's height is at least 2, so either the left side exists and is at least of
        // height 1 (which we just checked, with `subtree_height`) or the same is true of right.
        let mut node = unsafe { rebuilding_child.get_unchecked() };
        match node.rebuild_deepest_leaf(balance, next_dim) {
            Balanced(NoChange) => Balanced(NoChange),
            Balanced(Updated) => {
                drop(node);
                drop((rebuilding_child, sibling));
                Balanced(self.update_stat())
            }
            Rebuilding(mut collection) => {
                collection.push(node.into_owned());
                if balance.tree_is_balanced(self_height - 1, collection.len()) {
                    // Subtree is still balanced, collect sibling and continue
                    collection.extend(sibling.into_iter());
                    Rebuilding(collection)
                } else {
                    // Tree is re-buildable: rebuild it into the dest child.
                    // SAFETY: We trivially have a non-empty collection because we just pushed the
                    // rebuilding child into it.
                    rebuilding_child
                        .replace(unsafe { Node::must_build_from(collection, next_dim) });
                    drop((rebuilding_child, sibling));
                    Balanced(self.update_stat())
                }
            }
        }
    }

    fn pop_deepest_leaf(&mut self) -> ScapegoatPop<Self::Ownership> {
        if self.stat().height() == 0 {
            return RemoveThis;
        }
        fn pop_deepest_impl<Node>(node: &mut Node) -> (Node::Ownership, StatUpdate)
        where
            Node: MutableBinaryNode + UpdateStat,
            <Node as Statistic>::Stat: Height,
        {
            let node_height = node.stat().height();
            if node_height == 1 {
                return (
                    // We prefer to take right children to bias left-heavy.
                    // SAFETY: Node has height 1, which means it has either 1 or 2 leaves. If right
                    // is None, left must be Some.
                    node.right_child().into_owned().unwrap_or_else(|| {
                        unsafe { node.left_child().get_unchecked() }.into_owned()
                    }),
                    node.update_stat(),
                );
            }
            // In the event of a potential tie, we take from the right side to bias left-heavy.
            let mut deeper_child = if subtree_height(node.right()) == node_height {
                node.right_child()
            } else {
                node.left_child()
            };
            // SAFETY: node's height is at least 2, so either the right side exists and is at least
            // of height 1 (which we just checked, with `subtree_height`) or the same is true of
            // left.
            let (popped, update) =
                pop_deepest_impl(unsafe { deeper_child.get_unchecked() }.deref_mut());
            drop(deeper_child);
            (
                popped,
                match update {
                    NoChange => NoChange,
                    Updated => node.update_stat(),
                },
            )
        }
        let (popped, update) = pop_deepest_impl(self);
        Popped(popped, update)
    }

    fn remove_value<Vcmp>(
        &mut self,
        val: &Vcmp,
        dim: usize,
    ) -> Option<ScapegoatPop<Self::Ownership>>
    where
        Vcmp: KdValue<Dimension = <Self::Value as KdValue>::Dimension> + PartialOrd<Self::Value>,
    {
        assert_eq!(Vcmp::DIMS, <Self::Value as KdValue>::DIMS);
        let compared = val
            .get_dimension(dim)
            .borrow()
            .total_cmp(self.discriminant(dim).borrow());
        let go_left = match compared {
            Less => true,
            Greater => false,
            Equal => match val.partial_cmp(self.value()) {
                Some(Less) => true,
                Some(Greater) => false,
                Some(Equal) => {
                    // We're removing this node's value. Pop the deepest leaf and swap its value
                    // into this node, then return that popped leaf node.
                    return Some(match self.pop_deepest_leaf() {
                        RemoveThis => RemoveThis,
                        Popped(mut popped, updated) => {
                            swap(popped.value_mut(), self.value_mut());
                            if let Updated = updated {
                                *self.stat_mut() = self.make_stat();
                            }
                            // Because this node's *value* has changed, the parent must *always*
                            // update its own stat.
                            Popped(popped, Updated)
                        }
                    });
                }
                None => return None, // When val is unordered with the node value we give up.
            },
        };
        let mut descend_into = if go_left {
            self.left_child()
        } else {
            self.right_child()
        };
        let Some(mut child) = descend_into.get() else { return None };
        match child.remove_value(val, Self::Value::next_dim(dim)) {
            None => None,
            Some(RemoveThis) => {
                let popped = child.into_owned();
                drop(descend_into);
                Some(Popped(popped, self.update_stat()))
            }
            Some(Popped(popped, updated)) => {
                drop(child);
                drop(descend_into);
                Some(Popped(
                    popped,
                    match updated {
                        NoChange => NoChange,
                        Updated => self.update_stat(),
                    },
                ))
            }
        }
    }

    unsafe fn remove_node(node: *mut Self) -> Option<Self::Ownership>
    where
        Self: Parent,
    {
        debug_assert!(!node.is_null());
        // SAFETY: We can enter the tree mutably here as this function's contract includes that no
        // other references into any node in the tree already exist.
        let mut current: &mut Self = unsafe { &mut *node };
        let popped = match current.pop_deepest_leaf() {
            RemoveThis => {
                if current.parent().is_null() {
                    // The removal target is both the head of the tree and a leaf. Ownership of the
                    // target node is held outside the tree.
                    return None;
                }
                // The target node is a leaf; remove it directly from its parent node instead. Find
                // which side it's on and just take it out from there.
                // SAFETY: We can destroy and replace our mutable reference with that of its
                // parent here, as it is the only extant reference into the tree.
                current = unsafe { &mut *current.parent() };
                let [mut left_child, mut right_child] = current.children();
                let popped_target = match left_child.get() {
                    Some(child) => {
                        if child.deref() as *const Node == node {
                            child.into_owned()
                        } else {
                            // SAFETY: the left child is not the target node, so the right must be.
                            unsafe { right_child.get_unchecked() }.into_owned()
                        }
                    }
                    // SAFETY: the left child is not the target node, so the right must be.
                    None => unsafe { right_child.get_unchecked() }.into_owned(),
                };
                drop((left_child, right_child));
                // We just removed a child of this node; recompute its stats.
                *current.stat_mut() = current.make_stat();
                popped_target
            }
            Popped(mut popped, child_stats_updated) => {
                // Update the target node (current)'s stats if its child node's stats were updated
                if let Updated = child_stats_updated {
                    *current.stat_mut() = current.make_stat();
                }
                // Swap the value targeted for removal into the popped node
                swap(popped.value_mut(), current.value_mut());
                // Update stats of the ancestors, up to the root. Because current's value changed,
                // we always update at least the stats of the parent node.
                popped
            }
        };
        // Fix stats of ancestors traveling up the tree
        let mut updating = Updated;
        while let Updated = updating {
            if current.parent().is_null() {
                break; // reached the root
            }
            // SAFETY: We can destroy and replace our mutable reference with that of its
            // parent here, as it is the only extant reference into the tree.
            current = unsafe { &mut *current.parent() };
            updating = current.update_stat();
        }
        Some(popped)
    }
}

impl<V, Stat, Node> UpdateStat for Node
where
    Stat: ValueStatistic<V>,
    Node: BinaryNode + Value<Value = V> + Statistic<Stat = Stat>,
{
    fn make_stat(&self) -> Stat {
        Stat::combine(
            [self.left(), self.right()]
                .into_iter()
                .flatten()
                .map(Node::value),
            [self.left(), self.right()]
                .into_iter()
                .flatten()
                .map(Node::stat),
        )
    }

    fn update_stat(&mut self) -> StatUpdate {
        let new_stat = self.make_stat();
        return if self.stat() == &new_stat {
            NoChange
        } else {
            *self.stat_mut() = new_stat;
            Updated
        };
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct WithHeight<Stat: Default> {
    pub stat: Stat,
    height: u32,
}

impl<Stat: Default> Default for WithHeight<Stat> {
    fn default() -> Self {
        Self {
            stat: Default::default(),
            height: 0,
        }
    }
}

impl<Stat: Default> Height for WithHeight<Stat> {
    fn height(&self) -> u32 {
        self.height
    }
}

impl<V, Stat: ValueStatistic<V>> ValueStatistic<V> for WithHeight<Stat> {
    fn combine<'a, Vs, Ss>(values: Vs, stats: Ss) -> Self
    where
        V: 'a,
        Self: 'a,
        Vs: IntoIterator<Item = &'a V>,
        <Vs as IntoIterator>::IntoIter: Clone,
        Ss: IntoIterator<Item = &'a Self>,
        <Ss as IntoIterator>::IntoIter: Clone,
    {
        let stat_iter = stats.into_iter();
        Self {
            stat: Stat::combine(values, stat_iter.clone().map(|s| &s.stat)),
            height: stat_iter.fold(0, |most, s| max(most, s.height + 1)),
        }
    }
}

pub struct ConsumingBinaryTreeIter<Node>
where
    Node: MutableBinaryNode,
{
    stack: Vec<Node::Ownership>,
}

impl<Node> Iterator for ConsumingBinaryTreeIter<Node>
where
    Node: MutableBinaryNode,
{
    type Item = Node::Ownership;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|mut node| {
            self.stack.extend(
                node.children()
                    .into_iter()
                    .filter_map(Node::Child::into_owned),
            );
            node
        })
    }
}

pub struct ViewingBinaryTreeIter<'a, Node: BinaryNode> {
    stack: Vec<&'a Node>,
}

impl<'a, Node: BinaryNode> Iterator for ViewingBinaryTreeIter<'a, Node> {
    type Item = &'a Node;

    fn next(&mut self) -> Option<Self::Item> {
        self.stack.pop().map(|node| {
            self.stack
                .extend([node.left(), node.right()].into_iter().flatten());
            node
        })
    }
}

#[cfg(feature = "full_validation")]
impl<'a, Node> ViewingBinaryTreeIter<'a, Node>
where
    Node: BinaryNode + Ownable,
{
    pub fn new<'b: 'a, T>(tree: &'b T) -> Self
    where
        T: Tree<Node = Node>,
    {
        Self {
            stack: tree.as_ref().into_iter().collect(),
        }
    }
}

#[cfg(feature = "full_validation")]
pub fn validate_tree<T>(tree: &mut T)
where
    T: Tree,
    T::Node: ScapegoatKdNode + MutableBinaryNode,
    <T::Node as Statistic>::Stat: Debug + Eq,
    <T::Node as Value>::Value: KdValue,
{
    tree.validate();
    let Some(mut root) = tree.get() else { return };
    impl_scapegoat_node_validation(
        root.deref_mut(),
        0,
        vec![Default::default(); <T::Node as Value>::Value::DIMS],
    )
}

#[cfg(feature = "full_validation")]
#[derive(Clone)]
struct DimensionBound<D> {
    lower: Option<D>,
    upper: Option<D>,
}

#[cfg(feature = "full_validation")]
impl<D> Default for DimensionBound<D> {
    fn default() -> Self {
        Self {
            lower: None,
            upper: None,
        }
    }
}

#[cfg(feature = "full_validation")]
impl<D: TotalOrd> DimensionBound<D> {
    fn assert_contains(&self, v: &D) {
        assert!(self
            .lower
            .as_ref()
            .map_or(true, |low| low.total_cmp(v).is_le()));
        assert!(self
            .upper
            .as_ref()
            .map_or(true, |high| v.total_cmp(high).is_le()));
    }
}

#[cfg(feature = "full_validation")]
fn impl_scapegoat_node_validation<Node>(
    node: &mut Node,
    dim: usize,
    bounds: Vec<DimensionBound<<Node::Value as KdValue>::Dimension>>,
) where
    Node: ScapegoatKdNode + MutableBinaryNode,
    Node::Value: KdValue,
    Node::Stat: Debug + Eq,
{
    assert_eq!(bounds.len(), Node::Value::DIMS);
    assert_eq!(node.stat(), &node.make_stat());
    // Check the contained value falls within the allowed bounds.
    for (bound_dim, bound) in bounds.iter().enumerate() {
        bound.assert_contains(node.value().get_dimension(bound_dim).borrow());
    }
    let splitting_plane = node.discriminant(dim).borrow().clone();
    // Recurse with those bounds into children
    let next_dim = Node::Value::next_dim(dim);
    let [mut left_child, mut right_child] = node.children();
    left_child.validate();
    if let Some(mut child) = left_child.get() {
        let mut left_bounds = bounds.clone();
        left_bounds[dim].upper = Some(splitting_plane.clone());
        impl_scapegoat_node_validation(child.deref_mut(), next_dim, left_bounds);
    };
    right_child.validate();
    if let Some(mut child) = right_child.get() {
        let mut right_bounds = bounds;
        right_bounds[dim].lower = Some(splitting_plane);
        impl_scapegoat_node_validation(child.deref_mut(), next_dim, right_bounds);
    };
}

#[derive(Debug)]
pub struct KdBoxNodeParent<V: KdValue, Stat: ValueStatistic<V>> {
    left: Option<AliasableBox<Self>>,
    right: Option<AliasableBox<Self>>,
    parent: *mut Self,
    value: V,
    mid: V::Dimension,
    stat: Stat,
}

impl<V: KdValue, Stat: ValueStatistic<V>> Consumable for AliasableBox<KdBoxNodeParent<V, Stat>> {
    type Value = V;

    fn consume(self) -> V {
        AliasableBox::into_unique(self).value
    }
}

impl<V: KdValue, Stat: ValueStatistic<V>> Ownable for KdBoxNodeParent<V, Stat> {
    type Ownership = AliasableBox<Self>;
}

pub struct Takeable<'a, T: Ownable> {
    opt: &'a mut Option<T::Ownership>,
}

impl<'a, T: Ownable> Deref for Takeable<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        // SAFETY: self owns the &mut to the Option, which is always checked before construction.
        unsafe { self.opt.as_ref().unwrap_unchecked() }
    }
}

impl<'a, T: Ownable> DerefMut for Takeable<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: self owns the &mut to the Option, which is always checked before construction.
        unsafe { self.opt.as_mut().unwrap_unchecked() }
    }
}

impl<'a, T> IntoOwned for Takeable<'a, T>
where
    T: Ownable,
{
    fn into_owned(self) -> T::Ownership {
        // SAFETY: self owns the &mut to the Option, which is always checked before construction.
        unsafe { self.opt.take().unwrap_unchecked() }
    }
}

impl<'a, T: Ownable> Takeable<'a, T> {
    unsafe fn new(opt: &'a mut Option<T::Ownership>) -> Self {
        debug_assert!(opt.is_some());
        Self { opt }
    }
}

pub struct KdBoxNodeParentChild<'a, V, Stat>
where
    V: 'a + KdValue,
    Stat: 'a + ValueStatistic<V>,
{
    opt: &'a mut Option<AliasableBox<KdBoxNodeParent<V, Stat>>>,
    owner: *mut KdBoxNodeParent<V, Stat>,
}

impl<'a, V, Stat> IntoIterator for KdBoxNodeParentChild<'a, V, Stat>
where
    V: 'a + KdValue,
    Stat: 'a + ValueStatistic<V> + Height,
{
    type Item = AliasableBox<KdBoxNodeParent<V, Stat>>;
    type IntoIter = ConsumingBinaryTreeIter<KdBoxNodeParent<V, Stat>>;

    fn into_iter(self) -> Self::IntoIter {
        ConsumingBinaryTreeIter {
            stack: self.opt.take().into_iter().collect(),
        }
    }
}

impl<'a, V, Stat> Tree for KdBoxNodeParentChild<'a, V, Stat>
where
    V: 'a + KdValue,
    Stat: 'a + ValueStatistic<V> + Height,
{
    type Node = KdBoxNodeParent<V, Stat>;
    type Takeable<'b> = Takeable<'b, KdBoxNodeParent<V, Stat>>
    where
        Self: 'b;

    fn validate(&self) {
        if let Some(child) = self.opt.as_ref() {
            assert_eq!(child.parent, self.owner)
        }
    }

    fn as_ref(&self) -> Option<&KdBoxNodeParent<V, Stat>> {
        self.opt.as_ref().map(Deref::deref)
    }

    fn get(&mut self) -> Option<Self::Takeable<'_>> {
        if self.opt.is_some() {
            // SAFETY: We checked.
            Some(unsafe { self.get_unchecked() })
        } else {
            None
        }
    }

    unsafe fn get_unchecked(&mut self) -> Self::Takeable<'_> {
        unsafe { Takeable::new(self.opt) }
    }

    fn replace(
        &mut self,
        mut with: AliasableBox<KdBoxNodeParent<V, Stat>>,
    ) -> Option<AliasableBox<KdBoxNodeParent<V, Stat>>> {
        with.parent = self.owner;
        self.opt.replace(with)
    }

    fn take(&mut self) -> Option<AliasableBox<KdBoxNodeParent<V, Stat>>> {
        self.opt.take()
    }

    fn into_owned(self) -> Option<AliasableBox<KdBoxNodeParent<V, Stat>>> {
        self.opt.take()
    }
}

impl<V: KdValue, Stat: ValueStatistic<V> + Height> BinaryNode for KdBoxNodeParent<V, Stat> {
    fn left(&self) -> Option<&Self> {
        self.left.as_deref()
    }

    fn right(&self) -> Option<&Self> {
        self.right.as_deref()
    }

    fn is_leaf(&self) -> bool {
        self.stat.height() == 0
    }
}

impl<V: KdValue, Stat: ValueStatistic<V> + Height> MutableBinaryNode for KdBoxNodeParent<V, Stat> {
    type Child<'a> = KdBoxNodeParentChild<'a, V, Stat>
    where
        Self: 'a;

    fn left_child(&mut self) -> Self::Child<'_> {
        let owner: *mut Self = self;
        KdBoxNodeParentChild {
            opt: &mut self.left,
            owner,
        }
    }

    fn right_child(&mut self) -> Self::Child<'_> {
        let owner: *mut Self = self;
        KdBoxNodeParentChild {
            opt: &mut self.right,
            owner,
        }
    }

    fn children(&mut self) -> [Self::Child<'_>; 2] {
        let owner: *mut Self = self;
        [
            KdBoxNodeParentChild {
                opt: &mut self.left,
                owner,
            },
            KdBoxNodeParentChild {
                opt: &mut self.right,
                owner,
            },
        ]
    }
}

impl<V: KdValue, Stat: ValueStatistic<V>> Parent for KdBoxNodeParent<V, Stat> {
    fn parent(&self) -> *mut Self {
        self.parent
    }

    fn orphan(&mut self) {
        self.parent = null_mut();
    }
}

impl<V: KdValue, Stat: ValueStatistic<V>> Value for KdBoxNodeParent<V, Stat> {
    type Value = V;

    fn value(&self) -> &V {
        &self.value
    }

    fn value_mut(&mut self) -> &mut V {
        &mut self.value
    }
}

impl<V: KdValue, Stat: ValueStatistic<V>> Statistic for KdBoxNodeParent<V, Stat> {
    type Stat = Stat;

    fn stat(&self) -> &Stat {
        &self.stat
    }

    fn stat_mut(&mut self) -> &mut Stat {
        &mut self.stat
    }
}

#[derive(Debug)]
pub struct ParentTree<Node>
where
    Node: Ownable + Parent,
{
    head: Option<Node::Ownership>,
}

impl<Node> Default for ParentTree<Node>
where
    Node: Ownable + Parent,
{
    fn default() -> Self {
        Self { head: None }
    }
}

impl<Node> IntoIterator for ParentTree<Node>
where
    Node: MutableBinaryNode + Parent,
{
    type Item = Node::Ownership;
    type IntoIter = ConsumingBinaryTreeIter<Node>;

    fn into_iter(self) -> Self::IntoIter {
        ConsumingBinaryTreeIter {
            stack: self.into_owned().into_iter().collect(),
        }
    }
}

impl<Node> Tree for ParentTree<Node>
where
    Node: MutableBinaryNode + Parent,
{
    type Node = Node;
    type Takeable<'a> = Takeable<'a, Self::Node>
    where
        Self: 'a;

    fn validate(&self) {
        if let Some(root) = self.head.as_ref() {
            assert!(root.parent().is_null())
        }
    }

    fn as_ref(&self) -> Option<&Self::Node> {
        self.head.as_deref()
    }

    fn get(&mut self) -> Option<Self::Takeable<'_>> {
        if self.head.is_some() {
            Some(unsafe { self.get_unchecked() })
        } else {
            None
        }
    }

    unsafe fn get_unchecked(&mut self) -> Self::Takeable<'_> {
        Takeable {
            opt: &mut self.head,
        }
    }

    fn replace(&mut self, mut with: Node::Ownership) -> Option<Node::Ownership> {
        with.orphan();
        self.head.replace(with)
    }

    fn take(&mut self) -> Option<Node::Ownership> {
        self.head.take()
    }

    fn into_owned(mut self) -> Option<<Self::Node as Ownable>::Ownership> {
        self.head.take()
    }
}

impl<V, Stat> KdNode for KdBoxNodeParent<V, Stat>
where
    V: KdValue,
    Stat: ValueStatistic<V> + Height,
{
    type Tree = ParentTree<Self>;
    type Discriminant<'a> = &'a V::Dimension
    where
        Self: 'a;

    fn new(value: V, dim: usize) -> Self::Ownership {
        let mid = value.get_dimension(dim).borrow().clone();
        AliasableBox::from_unique(Box::new(Self {
            left: None,
            right: None,
            parent: null_mut(),
            value,
            mid,
            stat: Default::default(),
        }))
    }

    fn discriminant(&self, _dim: usize) -> &V::Dimension {
        &self.mid
    }
}

impl<V, Stat> MutableKdNode for KdBoxNodeParent<V, Stat>
where
    V: KdValue,
    Stat: ValueStatistic<V> + Height,
{
    fn set_dim(&mut self, dim: usize) {
        self.mid = self.value.get_dimension(dim).borrow().clone()
    }
}
