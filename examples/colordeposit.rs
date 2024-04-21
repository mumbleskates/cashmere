use cashmere::search::Nearest;
use cashmere::value::{MetricLimits, SquaredEuclidean};
use cashmere::KdValueMap;

mod common;
use common::{colordeposit_main, Color, ColorMap};

type Distance = SquaredEuclidean<f64>;

struct CashmereColorMap(KdValueMap<u32, [f32; 3]>);

impl ColorMap for CashmereColorMap {
    fn new() -> Self {
        Self(KdValueMap::new())
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn insert(&mut self, key: u32, value: Color) {
        self.0.insert(key, value.0);
    }

    fn nearest(&self, value: &Color) -> u32 {
        self.0
            .search_values(Nearest::new(&value.0, Distance::infinity()))
            .into_nearest()
            .unwrap()
            .value
            .plus
    }

    fn remove(&mut self, key: u32) {
        self.0.remove(key);
    }

    #[cfg(feature = "full_validation")]
    fn fully_validate(&mut self) {
        self.0.fully_validate()
    }
}

fn main() {
    colordeposit_main::<CashmereColorMap>();
}
