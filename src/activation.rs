use std::collections::HashMap;

use crate::{MatrixPlan, Scalar, Matrix};


pub trait Activation: 'static {
    fn forward<I: Scalar>(&self, from: Matrix<I>) -> Matrix<I> {
        let input: HashMap<String, Matrix<I>> = HashMap::new();
        let (output, _) = self.forward_plan(MatrixPlan::constant(from)).execute_cpu(&input);
        output
    }

    fn forward_plan<I: Scalar>(&self, from: MatrixPlan<I>) -> MatrixPlan<I>;

    fn derivative<I: Scalar>(&self, from: MatrixPlan<I>) -> MatrixPlan<I>;
}

pub struct Linear;

impl Activation for Linear {
    fn forward_plan<I: Scalar>(&self, from: MatrixPlan<I>) -> MatrixPlan<I> {
        from
    }

    fn derivative<I: Scalar>(&self, from: MatrixPlan<I>) -> MatrixPlan<I> {
        MatrixPlan::constant(Matrix::new(from.rows(), from.cols()).fill(I::ONE))
    }
}

pub struct Relu;

impl Activation for Relu {
    fn forward_plan<I: Scalar>(&self, from: MatrixPlan<I>) -> MatrixPlan<I> {
        from.max(I::default())
    }

    fn derivative<I: Scalar>(&self, from: MatrixPlan<I>) -> MatrixPlan<I> {
        from.sign().max(I::default())
    }
}

pub struct Sigmoid;

impl Activation for Sigmoid {
    fn forward_plan<I: Scalar>(&self, from: MatrixPlan<I>) -> MatrixPlan<I> {
        from.sigmoid()
    }

    fn derivative<I: Scalar>(&self, from: MatrixPlan<I>) -> MatrixPlan<I> {
        let one = MatrixPlan::constant(Matrix::new(from.rows(), from.cols()).fill(I::ONE));

        let sigmoid = from.sigmoid();
        let one_minus_sigmoid = one - &sigmoid;
        sigmoid.hadamard_mul(one_minus_sigmoid)
    }
}