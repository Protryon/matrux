use std::{ops::{Mul, Add, Neg, Sub}, sync::Arc, collections::HashMap};

use half::f16;

use crate::{Scalar, Matrix};

mod op;
use op::MatrixOp;

mod cpu_eval;

#[derive(Clone, Debug)]
pub struct MatrixPlan<I: Scalar> {
    rows: usize,
    cols: usize,
    source: Arc<MatrixOp<I>>,
}

impl<I: Scalar> Default for MatrixPlan<I> {
    fn default() -> Self {
        Self { rows: 0, cols: 0, source: Arc::new(MatrixOp::Input { name: String::new() }) }
    }
}

impl<I: Scalar> MatrixPlan<I> {
    pub fn input(rows: usize, cols: usize, name: impl AsRef<str>) -> Self {
        Self {
            rows,
            cols,
            source: Arc::new(MatrixOp::Input { name: name.as_ref().to_string() }),
        }
    }

    pub fn output(self, name: impl AsRef<str>) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            source: Arc::new(MatrixOp::Output { name: name.as_ref().to_string(), matrix: self }),
        }
    }

    pub fn constant(matrix: Matrix<I>) -> Self {
        Self {
            rows: matrix.rows(),
            cols: matrix.cols(),
            source: Arc::new(MatrixOp::Constant {
                matrix,
            }),
        }
    }

    /// Copys outputs from all internal plans, but throws away anonymous outputs
    pub fn merge_outputs(inner: impl IntoIterator<Item=MatrixPlan<I>>) -> MatrixPlan<I> {
        Self {
            rows: 0,
            cols: 0,
            source: Arc::new(MatrixOp::Combine { inner: inner.into_iter().collect() }),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn hadamard_mul<M: AsRef<MatrixPlan<I>>>(self, rhs: M) -> Self {
        let rhs = rhs.as_ref();
        assert_eq!(self.cols, rhs.cols);
        assert_eq!(self.rows, rhs.rows);
        MatrixPlan {
            rows: self.rows,
            cols: self.cols,
            source: Arc::new(MatrixOp::HadamardMul {
                left: self,
                right: rhs.clone(),
            }),
        }

    }

    pub fn transpose(self) -> Self {
        MatrixPlan {
            rows: self.cols,
            cols: self.rows,
            source: Arc::new(MatrixOp::Transpose {
                matrix: self,
            }),
        }
    }

    pub fn sign(self) -> Self {
        MatrixPlan {
            rows: self.rows,
            cols: self.cols,
            source: Arc::new(MatrixOp::Sign {
                matrix: self,
            }),
        }
    }

    pub fn sigmoid(self) -> Self {
        MatrixPlan {
            rows: self.rows,
            cols: self.cols,
            source: Arc::new(MatrixOp::Sigmoid {
                matrix: self,
            }),
        }
    }

    fn inputs_recur<'a>(&'a self, out: &mut Vec<(&'a str, (usize, usize))>) {
        match &*self.source {
            MatrixOp::Input { name } => {
                out.push((name, (self.rows, self.cols)));
            },
            MatrixOp::Constant { .. } => (),
            MatrixOp::Output { matrix, .. } |
            MatrixOp::Scale { matrix, .. } |
            MatrixOp::Max { matrix, .. } |
            MatrixOp::Neg { matrix } |
            MatrixOp::Transpose { matrix } |
            MatrixOp::Sigmoid { matrix } |
            MatrixOp::Sign { matrix } => {
                matrix.inputs_recur(out);
            },
            MatrixOp::Add { left, right } |
            MatrixOp::Sub { left, right } |
            MatrixOp::HadamardMul { left, right } |
            MatrixOp::Mul { left, right } => {
                left.inputs_recur(out);
                right.inputs_recur(out);
            },
            MatrixOp::Combine { inner } => {
                for inner in inner {
                    inner.inputs_recur(out);
                }
            },
        }
    }

    pub fn inputs(&self) -> Vec<(&str, (usize, usize))> {
        let mut out = vec![];
        self.inputs_recur(&mut out);
        out
    }

    pub fn execute_cpu(&self, inputs: &HashMap<impl AsRef<str>, impl AsRef<Matrix<I>>>) -> (Matrix<I>, HashMap<String, Matrix<I>>) {
        let inputs = inputs.iter().map(|(k, v)| (k.as_ref(), v.as_ref())).collect::<HashMap<_, _>>();
        cpu_eval::MatrixPlanCPUContext::execute(self, &inputs)
    }

    pub fn scale(self, rhs: I) -> Self {
        MatrixPlan {
            rows: self.rows,
            cols: self.cols,
            source: Arc::new(MatrixOp::Scale { matrix: self, scalar: rhs }),
        }
    }

    pub fn max(self, rhs: I) -> Self {
        MatrixPlan {
            rows: self.rows,
            cols: self.cols,
            source: Arc::new(MatrixOp::Max { matrix: self, scalar: rhs }),
        }
    }
}

impl<I: Scalar> AsRef<MatrixPlan<I>> for MatrixPlan<I> {
    fn as_ref(&self) -> &MatrixPlan<I> {
        self
    }
}

macro_rules! mul_impl {
    ($scalar:ident) => {
        impl Mul<$scalar> for MatrixPlan<$scalar> {
            type Output = MatrixPlan<$scalar>;
        
            fn mul(self, rhs: $scalar) -> Self::Output {
                MatrixPlan {
                    rows: self.rows,
                    cols: self.cols,
                    source: Arc::new(MatrixOp::Scale { matrix: self, scalar: rhs }),
                }
            }
        }

        impl Mul<MatrixPlan<$scalar>> for $scalar {
            type Output = MatrixPlan<$scalar>;

            fn mul(self, rhs: MatrixPlan<$scalar>) -> Self::Output {
                rhs * self
            }
        }
    };
}

mul_impl!(f16);
mul_impl!(f32);
mul_impl!(f64);

impl<I: Scalar, M: AsRef<MatrixPlan<I>>> Mul<M> for MatrixPlan<I> {
    type Output = MatrixPlan<I>;

    fn mul(self, rhs: M) -> Self::Output {
        let rhs = rhs.as_ref();
        if self.cols != rhs.rows {
            panic!("cannot multiply _x{} by {}x_ matrix", self.cols, rhs.rows);
        }
        MatrixPlan {
            rows: self.rows,
            cols: rhs.cols,
            source: Arc::new(MatrixOp::Mul {
                left: self,
                right: rhs.clone(),
            }),
        }
    }
}

impl<I: Scalar, M: AsRef<MatrixPlan<I>>> Add<M> for MatrixPlan<I> {
    type Output = MatrixPlan<I>;

    fn add(self, rhs: M) -> Self::Output {
        let rhs = rhs.as_ref();
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        MatrixPlan {
            rows: self.rows,
            cols: self.cols,
            source: Arc::new(MatrixOp::Add {
                left: self,
                right: rhs.clone(),
            }),
        }
    }

}

impl<I: Scalar> Neg for MatrixPlan<I> {
    type Output = MatrixPlan<I>;

    fn neg(self) -> Self::Output {
        MatrixPlan {
            rows: self.rows,
            cols: self.cols,
            source: Arc::new(MatrixOp::Neg {
                matrix: self,
            }),
        }
    }
}

impl<I: Scalar, M: AsRef<MatrixPlan<I>>> Sub<M> for MatrixPlan<I> {
    type Output = MatrixPlan<I>;

    fn sub(self, rhs: M) -> Self::Output {
        let rhs = rhs.as_ref();
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        MatrixPlan {
            rows: self.rows,
            cols: self.cols,
            source: Arc::new(MatrixOp::Sub {
                left: self,
                right: rhs.clone(),
            }),
        }
    }
}
