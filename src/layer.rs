use std::collections::HashMap;

use crate::{Scalar, MatrixPlan, Matrix, Activation};

pub trait Layer<I: Scalar> {
    fn input_shape(&self) -> (usize, usize);

    fn output_shape(&self) -> (usize, usize);

    fn prepare_input(&mut self, id: usize);

    fn assign_input(&self, output: &mut HashMap<String, Matrix<I>>);

    fn set_weights(&mut self, weights: Matrix<I>);

    fn get_weights(&self) -> Option<&Matrix<I>>;

    fn forward(&self, from: Matrix<I>) -> Matrix<I>;

    fn forward_plan(&self, from: MatrixPlan<I>) -> MatrixPlan<I>;

    fn backward_plan(&self, prior: MatrixPlan<I>, layer_value: MatrixPlan<I>, lower_layer_value: MatrixPlan<I>) -> (MatrixPlan<I>, MatrixPlan<I>);
}

pub struct DenseLayer<I: Scalar, A: Activation> {
    weights: Matrix<I>,
    id: Option<usize>,
    input: Option<MatrixPlan<I>>,
    activation: A,
}

impl<I: Scalar, A: Activation> DenseLayer<I, A> {
    pub fn new(weights: Matrix<I>, activation: A) -> Self {
        Self {
            weights,
            id: None,
            input: None,
            activation,
        }
    }
}

impl<I: Scalar, A: Activation> Layer<I> for DenseLayer<I, A> {

    fn input_shape(&self) -> (usize, usize) {
        (self.weights.cols(), 1)
    }

    fn output_shape(&self) -> (usize, usize) {
        (self.weights.rows(), 1)
    }

    fn prepare_input(&mut self, id: usize) {
        self.id = Some(id);
        self.input = Some(MatrixPlan::input(self.weights.rows(), self.weights.cols(), format!("weights_{}", self.id.unwrap())));
    }

    fn assign_input(&self, output: &mut HashMap<String, Matrix<I>>) {
        assert!(self.id.is_some());
        output.insert(format!("weights_{}", self.id.unwrap()), self.weights.clone());
    }

    fn set_weights(&mut self, weights: Matrix<I>) {
        self.weights = weights;
    }

    fn get_weights(&self) -> Option<&Matrix<I>> {
        Some(&self.weights)
    }

    fn forward(&self, from: Matrix<I>) -> Matrix<I> {
        self.activation.forward(self.weights.clone() * from)
    }

    fn forward_plan(&self, from: MatrixPlan<I>) -> MatrixPlan<I> {
        assert!(self.input.is_some());
        self.activation.forward_plan(self.input.clone().unwrap() * from)
    }

    fn backward_plan(&self, prior: MatrixPlan<I>, layer_value: MatrixPlan<I>, lower_layer_value: MatrixPlan<I>) -> (MatrixPlan<I>, MatrixPlan<I>) {
        assert!(self.input.is_some());

        let sigma = prior.hadamard_mul(self.activation.derivative(layer_value));
        (self.input.clone().unwrap().transpose() * &sigma, sigma * lower_layer_value.transpose())
    }
}