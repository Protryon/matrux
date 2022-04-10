use std::{ops::{Mul, Div, Add, Sub, Neg}, iter::Sum, fmt::{Debug, Display}};

use half::f16;

pub trait Scalar: Clone + Copy + Default + Mul<Self, Output=Self> + Div<Self, Output=Self> + Add<Self, Output=Self> + Sub<Self, Output=Self> + Sum + Neg<Output=Self> + Display + Debug + PartialOrd + 'static {
    const ONE: Self;

    fn from_f64(from: f64) -> Self;

    fn is_nan(self) -> bool;

    fn power(self, exponent: Self) -> Self;
}

impl Scalar for f16 {
    const ONE: Self = f16::ONE;

    fn from_f64(from: f64) -> Self {
        f16::from_f64(from)
    }

    fn is_nan(self) -> bool {
        self.is_nan()
    }

    fn power(self, exponent: Self) -> Self {
        f16::from_f32(self.to_f32().powf(exponent.to_f32()))
    }
}

impl Scalar for f32 {
    const ONE: Self = 1.0;

    fn from_f64(from: f64) -> Self {
        from as f32
    }

    fn is_nan(self) -> bool {
        self.is_nan()
    }

    fn power(self, exponent: Self) -> Self {
        self.powf(exponent)
    }
}

impl Scalar for f64 {
    const ONE: Self = 1.0;

    fn from_f64(from: f64) -> Self {
        from
    }

    fn is_nan(self) -> bool {
        self.is_nan()
    }

    fn power(self, exponent: Self) -> Self {
        self.powf(exponent)
    }
}
