use std::collections::hash_map::Entry;
use std::collections::HashMap;

use kdtree::distance::squared_euclidean;
use kdtree::KdTree;

mod common;
use common::{colordeposit_main, Color, ColorMap};

#[derive(Copy, Clone, PartialOrd, PartialEq, Debug)]
struct ColorWithID(Color, usize);

impl AsRef<[f32]> for ColorWithID {
    fn as_ref(&self) -> &[f32] {
        self.0.as_ref()
    }
}

struct PopularKdMap {
    next_color_id: usize,
    // Associated data is the color id
    tree: KdTree<f32, u32, ColorWithID>,
    // Color and currently stored id by key
    table: HashMap<u32, ColorWithID>,
}

impl ColorMap for PopularKdMap {
    fn new() -> Self {
        Self {
            next_color_id: 0,
            tree: KdTree::new(3),
            table: HashMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.tree.size()
    }

    fn insert(&mut self, key: u32, value: Color) {
        match self.table.entry(key) {
            Entry::Occupied(mut entry) => {
                self.tree.remove(entry.get(), &key).unwrap();
                entry.get_mut().0 = value;
                self.tree.add(*entry.get(), key).unwrap();
            }
            Entry::Vacant(entry) => {
                // associate a unique id with this color to avoid a library bug where removing a
                // non-unique point causes an infinite loop
                let value = ColorWithID(value, self.next_color_id);
                self.next_color_id += 1;
                self.tree.add(value, key).unwrap();
                entry.insert(value);
            }
        }
    }

    fn nearest(&self, value: &Color) -> u32 {
        *self
            .tree
            .nearest(value.as_ref(), 1, &squared_euclidean)
            .unwrap()[0]
            .1
    }

    fn remove(&mut self, key: u32) {
        let value = self.table.remove(&key).unwrap();
        self.tree.remove(&value, &key).unwrap();
    }
}

fn main() {
    colordeposit_main::<PopularKdMap>();
}
