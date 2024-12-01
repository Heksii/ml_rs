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
            biases: Matrix::zeros(1, shape.output_size)?,
        })
    }

    pub fn randomize(mut self) -> Result<Self, Error> {
        use rand::{thread_rng, Rng};
        let mut rng = thread_rng();

        for j in 0..self.shape.output_size {
            for i in 0..self.shape.input_size {
                self.weights
                    .set(i, j, F::from(rng.gen_range(-1f64..1f64)).unwrap())?;
            }

            self.biases
                .set(0, j, F::from(rng.gen_range(-1f64..1f64)).unwrap())?
        }

        Ok(self)
    }

    pub fn forward(&self, in_data: Matrix<F>) -> Result<Matrix<F>, Error> {
        Ok(in_data.dot(&self.weights)?.add(&self.biases)?)
    }
}
