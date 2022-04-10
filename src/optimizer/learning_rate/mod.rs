use crate::Scalar;

mod exp;
pub use exp::*;

pub trait LearningRate<I: Scalar>: 'static {
    fn rate(&self, step: usize) -> I;
}

impl<I: Scalar> LearningRate<I> for I {
    fn rate(&self, _step: usize) -> I {
        *self
    }
}
