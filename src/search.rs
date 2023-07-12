use std::borrow::Borrow;
use std::cmp::Ordering;
use std::collections::binary_heap::PeekMut;
use std::collections::BinaryHeap;
use std::mem;

use crate::search::KdSearchGuide::SearchChildren;
use crate::search::OnlyOrBoth::{Both, Only};
use crate::value::{ByTotalOrd, KdValue, KdValueMetric, TotalOrd};

pub enum OnlyOrBoth {
    Only,
    Both,
}

pub enum KdSearchGuide {
    /// Do not visit either side.
    Skip,
    /// Search the left or right side first, either in isolation or including the other.
    SearchChildren(Ordering, OnlyOrBoth),
}

#[derive(Debug)]
pub struct StopSearch;

pub trait KdSearcher<V: KdValue, Stat> {
    fn visit(&mut self, val: &V) -> Result<(), StopSearch>;

    /// Given a splitting plane and its dimension number, plus an aggregated statistic for all the
    /// values on both sides, returns which side and in which order should be visited if any, or
    /// stops the search completely.
    fn guide_search<D>(&self, plane: D, dim: usize, stat: &Stat) -> KdSearchGuide
    where
        D: Borrow<V::Dimension>;

    /// Return true if this searcher has, or may have, changed what branches it is willing to search
    /// after visiting one or more values.
    fn recheck(&self) -> bool;
}

pub trait KdSearchable<V: KdValue, Stat> {
    fn search_with<S>(&self, searcher: &mut S)
    where
        S: KdSearcher<V, Stat>;

    fn search<S>(&self, mut searcher: S) -> S
    where
        S: KdSearcher<V, Stat>,
    {
        self.search_with(&mut searcher);
        searcher
    }
}

/// Ad-hoc k-d searcher, enabling easier transforms
pub struct AdhocKdSearcher<'a, S, VF, GF, RF> {
    pub searcher: &'a mut S,
    pub visitor: VF,
    pub guide: GF,
    pub recheck: RF,
}

impl<'a, V, Stat, S, VF, GF, RF> KdSearcher<V, Stat> for AdhocKdSearcher<'a, S, VF, GF, RF>
where
    V: KdValue,
    VF: FnMut(&mut S, &V) -> Result<(), StopSearch>,
    GF: Fn(&S, &V::Dimension, usize, &Stat) -> KdSearchGuide,
    RF: Fn(&S) -> bool,
{
    fn visit(&mut self, val: &V) -> Result<(), StopSearch> {
        (self.visitor)(self.searcher, val)
    }

    fn guide_search<D>(&self, plane: D, dim: usize, stat: &Stat) -> KdSearchGuide
    where
        D: Borrow<V::Dimension>,
    {
        (self.guide)(self.searcher, plane.borrow(), dim, stat)
    }

    fn recheck(&self) -> bool {
        (self.recheck)(self.searcher)
    }
}

#[derive(Debug)]
pub struct NearValue<Value: KdValue, Metric> {
    pub metric: Metric,
    pub value: Value,
}

impl<Value: KdValue, Metric: KdValueMetric<Value>> TotalOrd for NearValue<Value, Metric> {
    fn total_eq(&self, other: &Self) -> bool {
        self.metric.total_eq(&other.metric)
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        self.metric.total_cmp(&other.metric)
    }
}

#[derive(Clone, Debug)]
enum VecOrHeap<T: Ord> {
    Vec(Vec<T>),
    Heap(BinaryHeap<T>),
}

#[derive(Debug)]
pub struct NearestK<Target, Value, Metric, const K: usize>
where
    Target: Borrow<Value>,
    Value: KdValue,
    Metric: KdValueMetric<Value>,
{
    target: Target,
    heap: VecOrHeap<ByTotalOrd<NearValue<Value, Metric>>>,
}

impl<Target, Value, Metric, const K: usize> NearestK<Target, Value, Metric, K>
where
    Target: Borrow<Value>,
    Value: KdValue,
    Metric: KdValueMetric<Value>,
{
    pub fn new(target: Target) -> Self {
        assert!(K > 0, "nearest K=zero items is senseless");
        Self {
            target,
            heap: VecOrHeap::Vec(Vec::with_capacity(K)),
        }
    }

    pub fn into_nearest(self) -> Vec<NearValue<Value, Metric>> {
        match self.heap {
            VecOrHeap::Vec(mut vec) => {
                vec.sort();
                vec
            }
            VecOrHeap::Heap(heap) => heap.into_sorted_vec(),
        }
        .into_iter()
        .map(|bto| bto.0)
        .collect()
    }

    pub fn into_nearest_unsorted(self) -> Vec<NearValue<Value, Metric>> {
        match self.heap {
            VecOrHeap::Vec(vec) => vec,
            VecOrHeap::Heap(heap) => heap.into_vec(),
        }
        .into_iter()
        .map(|bto| bto.0)
        .collect()
    }
}

impl<Target, Value, Metric, Stat, const K: usize> KdSearcher<Value, Stat>
    for NearestK<Target, Value, Metric, K>
where
    Target: Borrow<Value>,
    Value: KdValue + Clone,
    Metric: KdValueMetric<Value>,
{
    fn visit(&mut self, val: &Value) -> Result<(), StopSearch> {
        assert!(K > 0);
        let this_distance = Metric::distance(self.target.borrow(), val);
        match self.heap {
            VecOrHeap::Vec(ref mut vec) => {
                vec.push(ByTotalOrd(NearValue {
                    metric: this_distance,
                    value: val.clone(),
                }));
                // Heapify when we visit the kth value
                if vec.len() == K {
                    self.heap = VecOrHeap::Heap(BinaryHeap::from(mem::take(vec)))
                }
                Ok(())
            }
            VecOrHeap::Heap(ref mut heap) => {
                // SAFETY: we never have an empty heap; we only heapify when we reach K items, and
                // K is asserted to be greater than zero.
                let head = unsafe { heap.peek_mut().unwrap_unchecked() };
                if this_distance.total_cmp(&head.0.metric).is_lt() {
                    let satiated = this_distance.is_zero();
                    PeekMut::pop(head);
                    heap.push(ByTotalOrd(NearValue {
                        metric: this_distance,
                        value: val.clone(),
                    }));
                    if satiated {
                        return Err(StopSearch); // We've found K values that cannot be closer.
                    }
                }
                Ok(())
            }
        }
    }

    fn guide_search<D>(&self, plane: D, dim: usize, _stat: &Stat) -> KdSearchGuide
    where
        D: Borrow<Value::Dimension>,
    {
        let _this_held = self.target.borrow().get_dimension(dim);
        let this = _this_held.borrow();
        let visiting = plane.borrow();
        SearchChildren(
            this.total_cmp(visiting),
            match self.heap {
                VecOrHeap::Vec(_) => Both,
                VecOrHeap::Heap(ref heap) => {
                    // SAFETY: we never have an empty heap.
                    if Metric::axial_distance(this, visiting, dim)
                        .total_cmp(&unsafe { heap.peek().unwrap_unchecked() }.0.metric)
                        .is_lt()
                    {
                        // Current Kth-best distance is greater than the distance to the plane, so
                        // we might find closer nodes on the other side of it.
                        Both
                    } else {
                        Only
                    }
                }
            },
        )
    }

    fn recheck(&self) -> bool {
        true // Search narrows as more nodes are visited
    }
}

#[derive(Debug)]
pub struct Nearest<'a, TargetValue, Value, Metric>
where
    Value: KdValue,
{
    target: &'a TargetValue,
    best_distance: Metric,
    best: Option<Value>,
}

impl<'a, TargetValue, Value, Metric> Nearest<'a, TargetValue, Value, Metric>
where
    Value: KdValue,
    TargetValue: KdValue<Dimension = Value::Dimension>,
    Metric: KdValueMetric<TargetValue>,
{
    pub fn new(target: &'a TargetValue, max_distance: Metric) -> Self {
        Self {
            target,
            best_distance: max_distance,
            best: None,
        }
    }

    pub fn nearest(&self) -> Option<&Value> {
        self.best.as_ref()
    }

    pub fn into_nearest(self) -> Option<NearValue<Value, Metric>> {
        self.best.map(|value| NearValue {
            metric: self.best_distance,
            value,
        })
    }
}

impl<'a, TargetValue, Value, Metric, Stat> KdSearcher<Value, Stat>
    for Nearest<'a, TargetValue, Value, Metric>
where
    Value: KdValue + Clone,
    TargetValue: KdValue<Dimension = Value::Dimension>,
    Metric: KdValueMetric<TargetValue>,
{
    fn visit(&mut self, val: &Value) -> Result<(), StopSearch> {
        let this_distance = Metric::distance(self.target, val);
        if this_distance.total_cmp(&self.best_distance).is_lt() {
            self.best_distance = this_distance;
            self.best.replace(val.clone());
            if self.best_distance.is_zero() {
                return Err(StopSearch); // Cannot improve distance
            }
        }
        Ok(())
    }

    fn guide_search<D>(&self, plane: D, dim: usize, _stat: &Stat) -> KdSearchGuide
    where
        D: Borrow<Value::Dimension>,
    {
        let _this_held = self.target.get_dimension(dim);
        let this = _this_held.borrow();
        let visiting = plane.borrow();
        SearchChildren(
            this.total_cmp(visiting),
            if Metric::axial_distance(this, visiting, dim)
                .total_cmp(&self.best_distance)
                .is_lt()
            {
                Both
            } else {
                Only
            },
        )
    }

    fn recheck(&self) -> bool {
        true // Search narrows as more nodes are visited
    }
}
