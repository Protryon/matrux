use std::{collections::HashMap, sync::Arc};

use crate::{MatrixPlan, Scalar, plan::op::MatrixOp, Matrix};

pub struct MatrixPlanCPUContext<'b, I: Scalar> {
    inputs: &'b HashMap<&'b str, &'b Matrix<I>>,
    outputs: HashMap<String, Matrix<I>>,
    cache: HashMap<u64, Matrix<I>>,
}

impl<'b, I: Scalar> MatrixPlanCPUContext<'b, I> {

    pub fn execute(plan: &MatrixPlan<I>, inputs: &'b HashMap<&'b str, &'b Matrix<I>>) -> (Matrix<I>, HashMap<String, Matrix<I>>) {
        // let inputs = inputs.iter().map(|(k, v)| (k.as_ref(), v.as_ref())).collect::<HashMap<_, _>>();
        let mut self_ = Self {
            inputs,
            outputs: HashMap::new(),
            cache: HashMap::new(),
        };

        let base_output = self_.execute_cpu_recur(plan);
        // println!("cache entries = {}", self_.cache.len());

        (base_output, self_.outputs)
    }

    fn execute_cpu_recur(&mut self, plan: &MatrixPlan<I>) -> Matrix<I> {
        let ptr = Arc::as_ptr(&plan.source) as u64;
        match self.cache.get(&ptr) {
            Some(cached) => {
                // println!("cache hit");
                cached.clone()
            },
            None => {
                let output = self.execute_cpu_recur_uncached(plan);
                if Arc::strong_count(&plan.source) > 1 {
                    self.cache.insert(ptr, output.clone());
                }
                output
            },
        }
    }

    fn execute_cpu_recur_uncached(&mut self, plan: &MatrixPlan<I>) -> Matrix<I> {
        let output = match &*plan.source {
            MatrixOp::Input { name } => {
                (*self.inputs.get(&**name).expect(&*format!("missing input for '{}'", name))).clone()
            },
            MatrixOp::Output { name, matrix } => {
                let matrix = self.execute_cpu_recur(matrix);
                self.outputs.insert(name.clone(), matrix.clone());
                matrix
            },
            MatrixOp::Constant { matrix } => {
                matrix.clone()
            },
            MatrixOp::Scale { matrix, scalar } => {
                self.execute_cpu_recur(matrix).scale(*scalar)
            },
            MatrixOp::Max { matrix, scalar } => {
                self.execute_cpu_recur(matrix).max(*scalar)
            },
            MatrixOp::Neg { matrix } => {
                -self.execute_cpu_recur(matrix)
            },
            MatrixOp::Transpose { matrix } => {
                self.execute_cpu_recur(matrix).transpose()
            },
            MatrixOp::Sign { matrix } => {
                self.execute_cpu_recur(matrix).sign()
            },
            MatrixOp::Sigmoid { matrix } => {
                self.execute_cpu_recur(matrix).sigmoid()
            },
            MatrixOp::Mul { left, right } => {
                self.execute_cpu_recur(left) * self.execute_cpu_recur(right)
            },
            MatrixOp::HadamardMul { left, right } => {
                self.execute_cpu_recur(left).hadamard_mul(self.execute_cpu_recur(right))
            },
            MatrixOp::Add { left, right } => {
                self.execute_cpu_recur(left) + self.execute_cpu_recur(right)
            },
            MatrixOp::Sub { left, right } => {
                self.execute_cpu_recur(left) - self.execute_cpu_recur(right)
            },
            MatrixOp::Combine { inner } => {
                for inner in inner {
                    self.execute_cpu_recur(inner);
                }
                Matrix::default()
            },
        };
        assert_eq!(plan.rows(), output.rows());
        assert_eq!(plan.cols(), output.cols());
        output
    }
    
}
