use crate::{Scalar, MatrixPlan, Matrix};


#[derive(Clone, Debug)]
pub enum MatrixOp<I: Scalar> {
    Input {
        name: String,
    },
    Output {
        name: String,
        matrix: MatrixPlan<I>,
    },
    Constant {
        matrix: Matrix<I>,
    },
    Scale {
        matrix: MatrixPlan<I>,
        scalar: I,
    },
    Max {
        matrix: MatrixPlan<I>,
        scalar: I,
    },
    Neg {
        matrix: MatrixPlan<I>,
    },
    Transpose {
        matrix: MatrixPlan<I>,
    },
    Sign {
        matrix: MatrixPlan<I>,
    },
    Sigmoid {
        matrix: MatrixPlan<I>,
    },
    Mul {
        left: MatrixPlan<I>,
        right: MatrixPlan<I>,
    },
    HadamardMul {
        left: MatrixPlan<I>,
        right: MatrixPlan<I>,
    },
    Add {
        left: MatrixPlan<I>,
        right: MatrixPlan<I>,
    },
    Sub {
        left: MatrixPlan<I>,
        right: MatrixPlan<I>,
    },
    Combine {
        inner: Vec<MatrixPlan<I>>,
    },
}
