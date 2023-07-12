use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::Sub;

use num_traits::{AsPrimitive, Float};

use crate::node::ValueStatistic;

pub trait KdValue: Ord {
    const DIMS: usize;
    type Dimension: Ord + Clone;
    type ReturnedDimension<'a>: Borrow<Self::Dimension>
    where
        Self: 'a;

    fn get_dimension(&self, dim: usize) -> Self::ReturnedDimension<'_>;
}

pub(crate) trait CycleDim {
    fn next_dim(dim: usize) -> usize;
}

impl<V: KdValue> CycleDim for V {
    #[inline]
    fn next_dim(dim: usize) -> usize {
        if dim + 1 == Self::DIMS {
            0
        } else {
            dim + 1
        }
    }
}

impl<V: Ord + Clone, const N: usize> KdValue for [V; N] {
    const DIMS: usize = N;
    type Dimension = V;
    type ReturnedDimension<'a> = V
    where
        Self: 'a;

    #[inline]
    fn get_dimension(&self, dim: usize) -> V {
        self[dim].clone()
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Axis2<A, B>
where
    A: Ord + Clone,
    B: Ord + Clone,
{
    A(A),
    B(B),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Axis3<A, B, C>
where
    A: Ord + Clone,
    B: Ord + Clone,
    C: Ord + Clone,
{
    A(A),
    B(B),
    C(C),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Axis4<A, B, C, D>
where
    A: Ord + Clone,
    B: Ord + Clone,
    C: Ord + Clone,
    D: Ord + Clone,
{
    A(A),
    B(B),
    C(C),
    D(D),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Axis5<A, B, C, D, E>
where
    A: Ord + Clone,
    B: Ord + Clone,
    C: Ord + Clone,
    D: Ord + Clone,
    E: Ord + Clone,
{
    A(A),
    B(B),
    C(C),
    D(D),
    E(E),
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Axis6<A, B, C, D, E, F>
where
    A: Ord + Clone,
    B: Ord + Clone,
    C: Ord + Clone,
    D: Ord + Clone,
    E: Ord + Clone,
    F: Ord + Clone,
{
    A(A),
    B(B),
    C(C),
    D(D),
    E(E),
    F(F),
}

impl<A, B> KdValue for (A, B)
where
    A: Ord + Clone,
    B: Ord + Clone,
{
    const DIMS: usize = 2;
    type Dimension = Axis2<A, B>;
    type ReturnedDimension<'a> = Self::Dimension
    where
        Self: 'a;

    #[inline]
    fn get_dimension(&self, dim: usize) -> Self::Dimension {
        match dim {
            0 => Axis2::A(self.0.clone()),
            1 => Axis2::B(self.1.clone()),
            _ => panic!("invalid dimension"),
        }
    }
}

impl<A, B, C> KdValue for (A, B, C)
where
    A: Ord + Clone,
    B: Ord + Clone,
    C: Ord + Clone,
{
    const DIMS: usize = 3;
    type Dimension = Axis3<A, B, C>;
    type ReturnedDimension<'a> = Self::Dimension
    where
        Self: 'a;

    #[inline]
    fn get_dimension(&self, dim: usize) -> Self::Dimension {
        match dim {
            0 => Axis3::A(self.0.clone()),
            1 => Axis3::B(self.1.clone()),
            2 => Axis3::C(self.2.clone()),
            _ => panic!("invalid dimension"),
        }
    }
}

impl<A, B, C, D> KdValue for (A, B, C, D)
where
    A: Ord + Clone,
    B: Ord + Clone,
    C: Ord + Clone,
    D: Ord + Clone,
{
    const DIMS: usize = 4;
    type Dimension = Axis4<A, B, C, D>;
    type ReturnedDimension<'a> = Self::Dimension
    where
        Self: 'a;

    #[inline]
    fn get_dimension(&self, dim: usize) -> Self::Dimension {
        match dim {
            0 => Axis4::A(self.0.clone()),
            1 => Axis4::B(self.1.clone()),
            2 => Axis4::C(self.2.clone()),
            3 => Axis4::D(self.3.clone()),
            _ => panic!("invalid dimension"),
        }
    }
}

impl<A, B, C, D, E> KdValue for (A, B, C, D, E)
where
    A: Ord + Clone,
    B: Ord + Clone,
    C: Ord + Clone,
    D: Ord + Clone,
    E: Ord + Clone,
{
    const DIMS: usize = 5;
    type Dimension = Axis5<A, B, C, D, E>;
    type ReturnedDimension<'a> = Self::Dimension
    where
        Self: 'a;

    #[inline]
    fn get_dimension(&self, dim: usize) -> Self::Dimension {
        match dim {
            0 => Axis5::A(self.0.clone()),
            1 => Axis5::B(self.1.clone()),
            2 => Axis5::C(self.2.clone()),
            3 => Axis5::D(self.3.clone()),
            4 => Axis5::E(self.4.clone()),
            _ => panic!("invalid dimension"),
        }
    }
}

impl<A, B, C, D, E, F> KdValue for (A, B, C, D, E, F)
where
    A: Ord + Clone,
    B: Ord + Clone,
    C: Ord + Clone,
    D: Ord + Clone,
    E: Ord + Clone,
    F: Ord + Clone,
{
    const DIMS: usize = 6;
    type Dimension = Axis6<A, B, C, D, E, F>;
    type ReturnedDimension<'a> = Self::Dimension
    where
        Self: 'a;

    #[inline]
    fn get_dimension(&self, dim: usize) -> Self::Dimension {
        match dim {
            0 => Axis6::A(self.0.clone()),
            1 => Axis6::B(self.1.clone()),
            2 => Axis6::C(self.2.clone()),
            3 => Axis6::D(self.3.clone()),
            4 => Axis6::E(self.4.clone()),
            5 => Axis6::F(self.5.clone()),
            _ => panic!("invalid dimension"),
        }
    }
}

/// KdValue wrapper that attaches extra data which is ignored for purposes of equality and
/// ordering.
#[derive(Clone, Debug)]
pub struct KdValuePlus<V: KdValue, Plus> {
    pub val: V,
    pub plus: Plus,
}

impl<V: KdValue, Plus> Eq for KdValuePlus<V, Plus> {}

impl<V: KdValue, Plus> Ord for KdValuePlus<V, Plus> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.val.cmp(&other.val)
    }
}

impl<V: KdValue, Plus> PartialOrd for KdValuePlus<V, Plus> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.val.partial_cmp(&other.val)
    }
}

impl<V: KdValue, Plus> PartialEq<Self> for KdValuePlus<V, Plus> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.val.eq(&other.val)
    }
}

impl<V: KdValue, Plus> KdValue for KdValuePlus<V, Plus> {
    const DIMS: usize = V::DIMS;
    type Dimension = V::Dimension;
    type ReturnedDimension<'a> = V::ReturnedDimension<'a>
    where
        Self: 'a;

    #[inline]
    fn get_dimension(&self, dim: usize) -> Self::ReturnedDimension<'_> {
        self.val.get_dimension(dim)
    }
}

#[derive(Clone, PartialEq, Eq, Default, Debug)]
pub struct ValueStatisticPlus<Stat: Default> {
    stat: Stat,
}

impl<Stat: Default> From<Stat> for ValueStatisticPlus<Stat> {
    fn from(stat: Stat) -> Self {
        Self { stat }
    }
}

impl<V, Plus, Stat> ValueStatistic<KdValuePlus<V, Plus>> for ValueStatisticPlus<Stat>
where
    V: KdValue,
    Stat: ValueStatistic<V>,
{
    #[inline]
    fn combine<'a, Vs, Ss>(values: Vs, stats: Ss) -> Self
    where
        V: 'a,
        Plus: 'a,
        Self: 'a,
        Vs: IntoIterator<Item = &'a KdValuePlus<V, Plus>>,
        <Vs as IntoIterator>::IntoIter: Clone,
        Ss: IntoIterator<Item = &'a Self>,
        <Ss as IntoIterator>::IntoIter: Clone,
    {
        Stat::combine(
            values.into_iter().map(|v| &v.val),
            stats.into_iter().map(|s| &s.stat),
        )
        .into()
    }
}

pub trait MetricLimits {
    /// Create a maximum metric, such that any two values are closer together than this value.
    fn infinity() -> Self;

    /// Returns whether this distance metric is "zero" and cannot possibly be improved.
    fn is_zero(&self) -> bool;
}

pub trait KdValueMetric<V: KdValue>: Ord + MetricLimits {
    /// Measure distance between any two values.
    fn distance<V2>(a: &V, b: &V2) -> Self
    where
        V2: KdValue<Dimension = V::Dimension>;

    /// Measure distance between two points (or planes) along a specific axis indicated by `dim`.
    fn axial_distance(a: &V::Dimension, b: &V::Dimension, dim: usize) -> Self;
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct SquaredEuclidean<Holder: Ord> {
    distance: Holder,
}

impl<Holder> MetricLimits for SquaredEuclidean<Holder>
where
    Holder: Ord + Float,
{
    fn infinity() -> Self {
        Self {
            distance: Holder::infinity(),
        }
    }

    fn is_zero(&self) -> bool {
        self.distance.is_zero()
    }
}

impl<Holder, V> KdValueMetric<V> for SquaredEuclidean<Holder>
where
    Holder: 'static + Float + Ord,
    V: KdValue,
    V::Dimension: Sub,
    <V::Dimension as Sub>::Output: AsPrimitive<Holder>,
{
    #[inline]
    fn distance<V2>(a: &V, b: &V2) -> Self
    where
        V2: KdValue<Dimension = V::Dimension>,
    {
        assert_eq!(V::DIMS, V2::DIMS);
        let mut distance = Holder::zero();
        for dim in 0..V::DIMS {
            let diff: Holder = (a.get_dimension(dim).borrow().clone()
                - b.get_dimension(dim).borrow().clone())
            .as_();
            distance = distance + diff * diff;
        }
        Self { distance }
    }

    #[inline]
    fn axial_distance(a: &V::Dimension, b: &V::Dimension, _dim: usize) -> Self {
        let diff: Holder = a.clone().sub(b.clone()).as_();
        Self {
            distance: diff * diff,
        }
    }
}
