use std::collections::HashMap;

use crate::{Scalar, MatrixPlan, Matrix, Layer, Activation, DenseLayer, Optimizer};

#[derive(Default)]
pub struct NeuralNetworkBuilder<I: Scalar> {
    plan: Option<MatrixPlan<I>>,
    layers: Vec<Box<dyn Layer<I>>>,
    inputs: usize,
    trained_steps: usize,
}

impl<I: Scalar> NeuralNetworkBuilder<I> {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn input(mut self, count: usize) -> Self {
        assert!(self.plan.is_none());
        self.plan = Some(MatrixPlan::input(count, 1, "input"));
        self.inputs = count;
        self
    }

    pub fn add_dense_layer<A: Activation>(self, count: usize, activation: A) -> Self {
        let last_count = if self.layers.is_empty() {
            self.inputs
        } else {
            self.layers.last().unwrap().output_shape().0
        };
        let weights = Matrix::new(count, last_count).fill(I::from_f64(0.5));

        self.add_dense_layer_weighted(weights, activation)
    }

    pub fn add_dense_layer_weighted<A: Activation>(mut self, weights: Matrix<I>, activation: A) -> Self {
        assert!(self.plan.is_some());
        let mut layer = DenseLayer::new(
            weights,
            activation,
        );
        layer.prepare_input(self.layers.len());

        self.plan = Some(layer.forward_plan(self.plan.take().unwrap()));
        self.layers.push(Box::new(layer));
        self
    }

    // pub fn layers(&self) -> &[Layer<I>] {
    //     &self.layers[..]
    // }

    pub fn hidden_layers(&self) -> usize {
        self.layers.len()
    }

    // pub fn set_layer_weights(&mut self, layer: usize, weights: &[I]) {
    //     assert!(layer < self.layers.len());
    //     assert_eq!(self.layers[layer].weights.rows(), weights.len());
    //     for (i, weight) in weights.iter().copied().enumerate() {
    //         self.layers[layer].weights[i][0] = weight;
    //     }
    // }

    pub fn fill_plan_weights(&self, output: &mut HashMap<String, Matrix<I>>) {
        for layer in self.layers.iter() {
            layer.assign_input(output);
        }
    }

    pub fn eval(&self, inputs: &[I]) -> Vec<I> {
        assert!(self.plan.is_some());

        let matrix = Matrix::from_col(inputs.iter().copied());

        let mut inputs = HashMap::new();
        inputs.insert("input".to_string(), matrix);
        self.fill_plan_weights(&mut inputs);
        let (outputs, _) = self.plan.as_ref().unwrap().execute_cpu(&inputs);
        assert_eq!(outputs.cols(), 1);
        outputs.col(0).collect()
    }

    pub fn apply_backprop<O: Optimizer<I>>(&mut self, optimizer: &mut O, gradients: Vec<Matrix<I>>) {
        assert_eq!(self.layers.len(), gradients.len());
        self.layers.iter_mut().zip(gradients.into_iter()).for_each(|(current, gradient)| {
            //TODO: weight-less layers
            let weights = current.get_weights().unwrap().clone();
            current.set_weights(optimizer.optimize(weights, gradient, self.trained_steps));
        });
        self.trained_steps += 1;
    }

    pub fn plan_backprop(&self, batch_size: usize) -> MatrixPlan<I> {
        assert!(!self.layers.is_empty());

        let targets = MatrixPlan::<I>::input(self.plan.as_ref().unwrap().rows(), batch_size, "targets");
        let inputs = MatrixPlan::<I>::input(self.inputs, batch_size, "inputs");

        let mut state = inputs;
        let mut layer_values = vec![state.clone()];

        for layer in &self.layers {
            state = layer.forward_plan(state);
            layer_values.push(state.clone());
        }

        let outputs = layer_values.last().cloned().unwrap().output("outputs");

        let diff = layer_values.last().cloned().unwrap() - targets;

        let mut prior = diff;
        let mut output = vec![];
        for ((layer, layer_value), lower_layer_value) in self.layers.iter().rev()
            .zip(layer_values.iter().rev())
            .zip(layer_values.iter().rev().skip(1)) {
            let (new_prior, out) = layer.backward_plan(prior, layer_value.clone(), lower_layer_value.clone());
            prior = new_prior;
            output.push(out);
        }

        output.reverse();
        output.iter_mut().enumerate().for_each(|(i, sigma)| {
            let old = std::mem::take(sigma);
            *sigma = old.output(format!("gradient_{}", i));
        });
        println!("output = {:#?}", output);
        output.push(outputs);

        MatrixPlan::merge_outputs(output)
    }
}
