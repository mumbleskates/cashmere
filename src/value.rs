use std::borrow::Borrow;
use std::cmp::Ordering;
use std::fmt::Debug;
use std::ops::Sub;

use num_traits::{AsPrimitive, Float};

use crate::node::ValueStatistic;

pub trait TotalOrd {
    fn total_eq(&self, other: &Self) -> bool;
    fn total_cmp(&self, other: &Self) -> Ordering;
}

impl<T: TotalOrd> TotalOrd for &T {
    fn total_eq(&self, other: &Self) -> bool {
        (*self).total_eq(*other)
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        (*self).total_cmp(*other)
    }
}

impl<V: TotalOrd, const N: usize> TotalOrd for [V; N] {
    fn total_eq(&self, other: &Self) -> bool {
        self.iter().zip(other).all(|(a, b)| a.total_eq(b))
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        self.iter()
            .map(ByTotalOrd)
            .cmp(other.iter().map(ByTotalOrd))
    }
}

impl<V: TotalOrd> TotalOrd for [V] {
    fn total_eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(a, b)| a.total_eq(b))
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        self.iter()
            .map(ByTotalOrd)
            .cmp(other.iter().map(ByTotalOrd))
    }
}

impl TotalOrd for f32 {
    fn total_eq(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        f32::total_cmp(self, other)
    }
}

impl TotalOrd for f64 {
    fn total_eq(&self, other: &Self) -> bool {
        self.to_bits() == other.to_bits()
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        f64::total_cmp(self, other)
    }
}

macro_rules! define_total_ord {
    ($T:ty) => {
        impl TotalOrd for $T {
            fn total_eq(&self, other: &Self) -> bool {
                self == other
            }

            fn total_cmp(&self, other: &Self) -> Ordering {
                self.cmp(other)
            }
        }
    };
}

define_total_ord!(u8);
define_total_ord!(u16);
define_total_ord!(u32);
define_total_ord!(u64);
define_total_ord!(u128);
define_total_ord!(i8);
define_total_ord!(i16);
define_total_ord!(i32);
define_total_ord!(i64);
define_total_ord!(i128);
define_total_ord!(char);
define_total_ord!(bool);

define_total_ord!(str);
define_total_ord!(String);

define_total_ord!(std::time::Duration);
define_total_ord!(std::time::Instant);
define_total_ord!(std::time::SystemTime);

#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct ByTotalOrd<T: TotalOrd>(pub T);

impl<T: TotalOrd> Eq for ByTotalOrd<T> {}

impl<T: TotalOrd> PartialEq for ByTotalOrd<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.total_eq(&other.0)
    }
}

impl<T: TotalOrd> PartialOrd for ByTotalOrd<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.total_cmp(&other.0))
    }
}

impl<T: TotalOrd> Ord for ByTotalOrd<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

// TODO(widders): relax TotalOrd for KdValue? or keep it because it gives equality?
pub trait KdValue: TotalOrd {
    const DIMS: usize;
    type Dimension: Clone + TotalOrd;
    // TODO(widders): TotalOrd + ToOwned<Self::Dimension> & Self::Dimension: Borrow<this one>?
    //  it should be possible to compare via a ref & discriminant instead of cloning eagerly (so we
    //  can use AxisN<&'a A, &'a B, ...>)
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

impl<V: Clone + TotalOrd, const N: usize> KdValue for [V; N] {
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

macro_rules! axis_type {
    ($ty:ident $(, $generics:ident)+) => {
        #[derive(Clone, Debug)]
        pub enum $ty<$($generics),+> {
            $($generics($generics),)+
        }

        impl<$($generics),+> TotalOrd for $ty<$($generics),+>
        where
            $($generics: TotalOrd,)+
        {
            fn total_eq(&self, other: &Self) -> bool {
                match (self, other) {
                    $(($ty::$generics(s), $ty::$generics(o)) => s.total_eq(o),)+
                    _ => false,
                }
            }

            fn total_cmp(&self, other: &Self) -> Ordering {
                match (self, other) {
                    $(($ty::$generics(s), $ty::$generics(o)) => s.total_cmp(o),)+
                    _ => panic!("compared different dims"),
                }
            }
        }
    };
}

axis_type!(Axis2, A, B);
axis_type!(Axis3, A, B, C);
axis_type!(Axis4, A, B, C, D);
axis_type!(Axis5, A, B, C, D, E);
axis_type!(Axis6, A, B, C, D, E, F);

impl<A, B> TotalOrd for (A, B)
where
    A: TotalOrd,
    B: TotalOrd,
{
    fn total_eq(&self, other: &Self) -> bool {
        self.0.total_eq(&other.0) && self.1.total_eq(&other.1)
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| self.1.total_cmp(&other.1))
    }
}

impl<A, B> KdValue for (A, B)
where
    A: Clone + TotalOrd,
    B: Clone + TotalOrd,
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

impl<A, B, C> TotalOrd for (A, B, C)
where
    A: TotalOrd,
    B: TotalOrd,
    C: TotalOrd,
{
    fn total_eq(&self, other: &Self) -> bool {
        self.0.total_eq(&other.0) && self.1.total_eq(&other.1) && self.2.total_eq(&other.2)
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| self.1.total_cmp(&other.1))
            .then_with(|| self.2.total_cmp(&other.2))
    }
}

impl<A, B, C> KdValue for (A, B, C)
where
    A: Clone + TotalOrd,
    B: Clone + TotalOrd,
    C: Clone + TotalOrd,
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

impl<A, B, C, D> TotalOrd for (A, B, C, D)
where
    A: TotalOrd,
    B: TotalOrd,
    C: TotalOrd,
    D: TotalOrd,
{
    fn total_eq(&self, other: &Self) -> bool {
        self.0.total_eq(&other.0)
            && self.1.total_eq(&other.1)
            && self.2.total_eq(&other.2)
            && self.3.total_eq(&other.3)
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| self.1.total_cmp(&other.1))
            .then_with(|| self.2.total_cmp(&other.2))
            .then_with(|| self.3.total_cmp(&other.3))
    }
}

impl<A, B, C, D> KdValue for (A, B, C, D)
where
    A: Clone + TotalOrd,
    B: Clone + TotalOrd,
    C: Clone + TotalOrd,
    D: Clone + TotalOrd,
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

impl<A, B, C, D, E> TotalOrd for (A, B, C, D, E)
where
    A: TotalOrd,
    B: TotalOrd,
    C: TotalOrd,
    D: TotalOrd,
    E: TotalOrd,
{
    fn total_eq(&self, other: &Self) -> bool {
        self.0.total_eq(&other.0)
            && self.1.total_eq(&other.1)
            && self.2.total_eq(&other.2)
            && self.3.total_eq(&other.3)
            && self.4.total_eq(&other.4)
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| self.1.total_cmp(&other.1))
            .then_with(|| self.2.total_cmp(&other.2))
            .then_with(|| self.3.total_cmp(&other.3))
            .then_with(|| self.4.total_cmp(&other.4))
    }
}

impl<A, B, C, D, E> KdValue for (A, B, C, D, E)
where
    A: Clone + TotalOrd,
    B: Clone + TotalOrd,
    C: Clone + TotalOrd,
    D: Clone + TotalOrd,
    E: Clone + TotalOrd,
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

impl<A, B, C, D, E, F> TotalOrd for (A, B, C, D, E, F)
where
    A: TotalOrd,
    B: TotalOrd,
    C: TotalOrd,
    D: TotalOrd,
    E: TotalOrd,
    F: TotalOrd,
{
    fn total_eq(&self, other: &Self) -> bool {
        self.0.total_eq(&other.0)
            && self.1.total_eq(&other.1)
            && self.2.total_eq(&other.2)
            && self.3.total_eq(&other.3)
            && self.4.total_eq(&other.4)
            && self.5.total_eq(&other.5)
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        self.0
            .total_cmp(&other.0)
            .then_with(|| self.1.total_cmp(&other.1))
            .then_with(|| self.2.total_cmp(&other.2))
            .then_with(|| self.3.total_cmp(&other.3))
            .then_with(|| self.4.total_cmp(&other.4))
            .then_with(|| self.5.total_cmp(&other.5))
    }
}

impl<A, B, C, D, E, F> KdValue for (A, B, C, D, E, F)
where
    A: Clone + TotalOrd,
    B: Clone + TotalOrd,
    C: Clone + TotalOrd,
    D: Clone + TotalOrd,
    E: Clone + TotalOrd,
    F: Clone + TotalOrd,
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

impl<V: KdValue, Plus> TotalOrd for KdValuePlus<V, Plus> {
    fn total_eq(&self, other: &Self) -> bool {
        self.val.total_eq(&other.val)
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        self.val.total_cmp(&other.val)
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

pub trait KdValueMetric<V: KdValue>: TotalOrd + MetricLimits {
    /// Measure distance between any two values.
    fn distance<V2>(a: &V, b: &V2) -> Self
    where
        V2: KdValue<Dimension = V::Dimension>;

    /// Measure distance between two points (or planes) along a specific axis indicated by `dim`.
    fn axial_distance(a: &V::Dimension, b: &V::Dimension, dim: usize) -> Self;
}

#[derive(Copy, Clone, Debug)]
pub struct SquaredEuclidean<Holder: TotalOrd + Float> {
    distance: Holder,
}

impl<Holder> MetricLimits for SquaredEuclidean<Holder>
where
    Holder: TotalOrd + Float,
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

impl<Holder> TotalOrd for SquaredEuclidean<Holder>
where
    Holder: 'static + Float + TotalOrd,
{
    fn total_eq(&self, other: &Self) -> bool {
        self.distance.total_eq(&other.distance)
    }

    fn total_cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

impl<Holder, V> KdValueMetric<V> for SquaredEuclidean<Holder>
where
    Holder: 'static + Float + TotalOrd,
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
