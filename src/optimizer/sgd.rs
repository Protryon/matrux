use crate::{Scalar, Optimizer, Matrix};

use super::LearningRate;

pub struct StochasticGradientDescent<I: Scalar, L: LearningRate<I>> {
    learning_rate: L,
    minimum: Option<I>,
    maximum: Option<I>,
}

impl<I: Scalar, L: LearningRate<I>> StochasticGradientDescent<I, L> {
    pub fn new(learning_rate: L) -> Self {
        Self {
            learning_rate,
            minimum: None,
            maximum: None,
        }
    }

    pub fn set_minimum(&mut self, minimum: I) -> &mut Self {
        self.minimum = Some(minimum);
        self
    }

    pub fn set_maximum(&mut self, maximum: I) -> &mut Self {
        self.maximum = Some(maximum);
        self
    }

    pub fn norm(&mut self) -> &mut Self {
        self.maximum = Some(I::ONE);
        self.minimum = Some(-I::ONE);
        self
    }
}

impl<I: Scalar, L: LearningRate<I>> Optimizer<I> for StochasticGradientDescent<I, L> {
    fn optimize(&mut self, weights: Matrix<I>, gradient: Matrix<I>, step: usize) -> Matrix<I> {
        let gradient = gradient.scale(self.learning_rate.rate(step));
        if gradient.has_nan() {
            eprintln!("NaN in gradients, skipping SGD application");
            return weights;
        }
        let mut out = weights - gradient;
        if let Some(maximum) = self.maximum {
            out = out.min(maximum);
        }
        if let Some(minimum) = self.minimum {
            out = out.max(minimum);
        }
        out
    }
}
