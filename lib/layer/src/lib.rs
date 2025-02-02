use std::io::Error;

use matrix::Matrix;
use num_traits::{Float, Zero};

#[derive(Clone, Copy, Debug)]
pub struct LayerShape {
    input_size: usize,
    output_size: usize,
}

impl LayerShape {
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
    pub shape: LayerShape,
    weights: Matrix<F>,
    biases: Matrix<F>,
    last_in: Option<Matrix<F>>,
    last_out: Option<Matrix<F>>,
}

impl<F: Float + Zero> Layer<F> {
    pub fn zeros(input_size: usize, output_size: usize) -> Result<Self, Error> {
        let shape = LayerShape::parse(input_size, output_size)?;

        Ok(Self {
            shape,
            weights: Matrix::zeros(shape.input_size, shape.output_size)?,
            biases: Matrix::zeros(1, shape.output_size)?,
            last_in: None,
            last_out: None,
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

    pub fn forward(&mut self, in_data: Matrix<F>) -> Result<Matrix<F>, Error> {
        self.last_in = Some(in_data.clone());

        let activation = in_data.dot(&self.weights)?.add(&self.biases)?;
        self.last_out = Some(activation.clone());

        Ok(activation)
    }

    pub fn apply_gradients(
        &mut self,
        weight_gradients: Matrix<F>,
        bias_gradients: Matrix<F>,
        learning_rate: F,
    ) -> Result<(), Error> {
        self.weights = self.weights.sub(&weight_gradients.scale(learning_rate)?)?;
        self.biases = self.biases.sub(&bias_gradients.scale(learning_rate)?)?;

        Ok(())
    }
}
