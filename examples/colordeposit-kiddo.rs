use std::collections::hash_map::Entry;
use std::collections::HashMap;

use kiddo::float::distance::SquaredEuclidean;
use kiddo::float::kdtree::KdTree;

mod common;
use common::{colordeposit_main, Color, ColorMap};

struct KiddoMap {
    tree: KdTree<f32, u32, 3, 64, u32>,
    table: HashMap<u32, Color>,
}

impl ColorMap for KiddoMap {
    fn new() -> Self {
        Self {
            tree: KdTree::new(),
            table: HashMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.tree.size() as usize
    }

    fn insert(&mut self, key: u32, value: Color) {
        match self.table.entry(key) {
            Entry::Occupied(mut entry) => {
                self.tree.remove(entry.get().as_ref(), key);
                *entry.get_mut() = value;
                self.tree.add(value.as_ref(), key);
            }
            Entry::Vacant(entry) => {
                // associate a unique id with this color to avoid a library bug where removing a
                // non-unique point causes an infinite loop
                self.tree.add(value.as_ref(), key);
                entry.insert(value);
            }
        }
    }

    fn nearest(&self, value: &Color) -> u32 {
        self.tree
            .nearest_one::<SquaredEuclidean>(value.as_ref())
            .item
    }

    fn remove(&mut self, key: u32) {
        let value = self.table.remove(&key).unwrap();
        self.tree.remove(value.as_ref(), key);
    }
}

fn main() {
    colordeposit_main::<KiddoMap>();
}
