use crate::{Scalar, Matrix};

pub mod learning_rate;
pub use learning_rate::LearningRate;

mod sgd;
pub use sgd::*;

pub trait Optimizer<I: Scalar>: 'static {
    fn optimize(&mut self, weights: Matrix<I>, gradient: Matrix<I>, step: usize) -> Matrix<I>;
}
