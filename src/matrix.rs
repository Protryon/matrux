use core::fmt;
use std::{ops::{Index, IndexMut, Mul, Add, Neg, Sub}, fmt::{Display, Debug}};

use half::f16;

use crate::Scalar;

#[derive(Clone, Debug, Default)]
pub struct Matrix<I: Scalar> {
    data: Vec<I>,
    rows: usize,
    cols: usize,
}

impl<I: Scalar> Matrix<I> {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![I::default(); rows * cols],
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn from_col(col: impl IntoIterator<Item=I>) -> Self {
        let data: Vec<I> = col.into_iter().collect();
        Self {
            rows: data.len(),
            cols: 1,
            data,
        }
    }

    pub fn col<'a>(&'a self, column: usize) -> impl Iterator<Item=I> + 'a {
        struct ColIter<'a, I: Scalar> {
            matrix: &'a Matrix<I>,
            col: usize,
            row: usize,
        }

        impl<'a, I: Scalar> Iterator for ColIter<'a, I> {
            type Item = I;

            fn next(&mut self) -> Option<Self::Item> {
                if self.row >= self.matrix.rows {
                    return None;
                }
                let out = Some(self.matrix[self.row][self.col]);
                self.row += 1;
                out
            }
        }

        ColIter {
            matrix: self,
            col: column,
            row: 0,
        }
    }

    pub fn scale(mut self, rhs: I) -> Self {
        for component in self.as_mut() {
            *component = *component * rhs;
        }
        self
    }

    pub fn max(mut self, rhs: I) -> Self {
        for component in self.as_mut() {
            *component = if *component > rhs {
                *component
            } else {
                rhs
            };
        }
        self
    }

    pub fn min(mut self, rhs: I) -> Self {
        for component in self.as_mut() {
            *component = if *component < rhs {
                *component
            } else {
                rhs
            };
        }
        self
    }

    pub fn sigmoid(mut self) -> Self {
        for component in self.as_mut() {
            *component = I::ONE / (I::ONE + I::from_f64(std::f64::consts::E).power(-*component));
        }
        self
    }

    /// sets each component to -1, 0, or 1
    pub fn sign(mut self) -> Self {
        for component in self.as_mut() {
            *component = if *component > I::default() {
                I::ONE
            } else if *component < I::default() {
                -I::ONE
            } else {
                I::default()
            };
        }
        self
    }

    pub fn hadamard_mul<M: AsRef<Matrix<I>>>(mut self, rhs: M) -> Self {
        let rhs = rhs.as_ref();
        assert_eq!(self.cols, rhs.cols);
        assert_eq!(self.rows, rhs.rows);
        for row in 0..self.rows {
            let cols = self.cols;
            let self_left_row = &mut self[row];
            let rhs_left_row = &rhs[row];
            
            for col in 0..cols {
                self_left_row[col] = self_left_row[col] * rhs_left_row[col];
            }
        }
        self
    }

    pub fn transpose(&self) -> Self {
        let mut out = Matrix::new(self.cols, self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                out[col][row] = self[row][col];
            }
        }
        out
    }

    pub fn fill(mut self, with: I) -> Self {
        self.data.iter_mut().for_each(|x| *x = with);
        self
    }

    pub fn has_nan(&self) -> bool {
        self.data.iter().any(|x| x.is_nan())
    }
}

impl<I: Scalar> Display for Matrix<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for row in 0..self.rows() {
            if row > 0 {
                write!(f, " ")?;
            }
            write!(f, "[")?;
            for col in 0..self.cols() {
                write!(f, "{}", self[row][col])?;
                if col < self.cols() - 1 {
                    write!(f, ",")?;
                }
            }
            write!(f, "]")?;
            if row < self.rows() - 1 {
                writeln!(f)?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl<I: Scalar> AsRef<Matrix<I>> for Matrix<I> {
    fn as_ref(&self) -> &Matrix<I> {
        self
    }
}

impl<I: Scalar> AsRef<[I]> for Matrix<I> {
    fn as_ref(&self) -> &[I] {
        &self.data[..]
    }
}

impl<I: Scalar> AsMut<[I]> for Matrix<I> {
    fn as_mut(&mut self) -> &mut [I] {
        &mut self.data[..]
    }
}

impl<I: Scalar> Index<(usize, usize)> for Matrix<I> {
    type Output = I;

    fn index(&self, (row, col): (usize, usize)) -> &Self::Output {
        &self.data[row * self.cols + col]
    }
}

impl<I: Scalar> IndexMut<(usize, usize)> for Matrix<I> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut Self::Output {
        &mut self.data[row * self.cols + col]
    }
}

impl<I: Scalar> Index<usize> for Matrix<I> {
    type Output = [I];

    fn index(&self, row: usize) -> &Self::Output {
        &self.data[row * self.cols..(row + 1) * self.cols]
    }
}

impl<I: Scalar> IndexMut<usize> for Matrix<I> {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        &mut self.data[row * self.cols..(row + 1) * self.cols]
    }
}

macro_rules! mul_impl {
    ($scalar:ident) => {
        impl Mul<$scalar> for Matrix<$scalar> {
            type Output = Matrix<$scalar>;
        
            fn mul(mut self, rhs: $scalar) -> Self::Output {
                for component in self.as_mut() {
                    *component = *component * rhs;
                }
                self
            }
        }

        impl Mul<Matrix<$scalar>> for $scalar {
            type Output = Matrix<$scalar>;

            fn mul(self, rhs: Matrix<$scalar>) -> Self::Output {
                rhs * self
            }
        }
    };
}

mul_impl!(f16);
mul_impl!(f32);
mul_impl!(f64);

impl<I: Scalar, M: AsRef<Matrix<I>>> Mul<M> for Matrix<I> {
    type Output = Matrix<I>;

    fn mul(self, rhs: M) -> Self::Output {
        let rhs = rhs.as_ref();
        if self.cols != rhs.rows {
            panic!("cannot multiply _x{} by {}x_ matrix", self.cols, rhs.rows);
        }
        let mut output = Matrix::<I>::new(self.rows, rhs.cols);
        for row in 0..self.rows {
            let left_row = &self[row];
            
            for col in 0..rhs.cols {
                output[row][col] = left_row.iter().copied().zip(rhs.col(col)).map(|(left, right)| left * right).sum::<I>();
            }
        }
        output
    }
}

impl<I: Scalar, M: AsRef<Matrix<I>>> Add<M> for Matrix<I> {
    type Output = Matrix<I>;

    fn add(mut self, rhs: M) -> Self::Output {
        let rhs = rhs.as_ref();
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let rhs: &[I] = rhs.as_ref();
        self.as_mut().iter_mut().zip(rhs.iter().copied()).for_each(|(target, source)| *target = *target + source);
        self
    }

}

impl<I: Scalar> Neg for Matrix<I> {
    type Output = Matrix<I>;

    fn neg(mut self) -> Self::Output {
        self.as_mut().iter_mut().for_each(|x| *x = -*x);
        self
    }
}

impl<I: Scalar, M: AsRef<Matrix<I>>> Sub<M> for Matrix<I> {
    type Output = Matrix<I>;

    fn sub(mut self, rhs: M) -> Self::Output {
        let rhs = rhs.as_ref();
        assert_eq!(self.rows, rhs.rows);
        assert_eq!(self.cols, rhs.cols);
        let rhs: &[I] = rhs.as_ref();
        self.as_mut().iter_mut().zip(rhs.iter().copied()).for_each(|(target, source)| *target = *target - source);
        self
    }
}
