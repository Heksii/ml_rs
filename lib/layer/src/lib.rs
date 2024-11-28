use std::io::Error;

use matrix::Matrix;
use num_traits::{Float, Zero};

#[derive(Clone, Copy, Debug)]
pub struct Shape {
    input_size: usize,
    output_size: usize,
}

impl Shape {
    pub fn parse(input_size: usize, output_size: usize) -> Result<Self, Error> {
        if input_size == 0 || output_size == 0 {
            return Err(Error::other(format!("layer shape dimensions cannot be zero, input_size({input_size}) or output_size({output_size}) was zero")));
        }

        Ok(Self {
            input_size,
            output_size,
        })
    }
}

pub struct Layer<F: Float + Zero> {
    shape: Shape,
    weights: Matrix<F>,
    biases: Matrix<F>,
}

impl<F: Float + Zero> Layer<F> {
    pub fn zeros(input_size: usize, output_size: usize) -> Result<Self, Error> {
        let shape = Shape::parse(input_size, output_size)?;

        Ok(Self {
            shape,
            weights: Matrix::zeros(shape.input_size, shape.output_size)?,
            biases: Matrix::zeros(shape.output_size, 1)?,
        })
    }
}
