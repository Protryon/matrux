use crate::Scalar;

use super::LearningRate;

pub struct ExponentialLearningRate<I: Scalar> {
    initial: I,
    decay_steps: I,
    decay_rate: I,
}

impl<I: Scalar> LearningRate<I> for ExponentialLearningRate<I> {
    fn rate(&self, step: usize) -> I {
        self.initial * self.decay_rate.power(I::from_f64(step as f64) / self.decay_steps)
    }
}