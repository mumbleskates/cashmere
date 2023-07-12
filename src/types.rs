pub trait RatioT: Copy + Default {
    const NUM: isize;
    const DEN: usize;
    const VAL: f64 = (Self::NUM as f64) / (Self::DEN as f64);
    const VAL32: f32 = Self::VAL as f32;
}

#[derive(Copy, Clone, Debug)]
pub struct Ratio<const NUM: isize, const DEN: usize>;

impl<const NUM: isize, const DEN: usize> RatioT for Ratio<NUM, DEN> {
    const NUM: isize = NUM;
    const DEN: usize = DEN;
}

impl<const NUM: isize, const DEN: usize> Default for Ratio<NUM, DEN> {
    fn default() -> Self {
        debug_assert!(DEN > 0, "ratio must have a positive denominator");
        Self
    }
}

pub struct CountedIter<I: Iterator> {
    iter: I,
    remaining: usize,
}

impl<I: Iterator> CountedIter<I> {
    pub fn new(iter: I, remaining: usize) -> Self {
        Self { iter, remaining }
    }
}

impl<I: Iterator> Iterator for CountedIter<I> {
    type Item = I::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.remaining -= 1;
        self.iter.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}
