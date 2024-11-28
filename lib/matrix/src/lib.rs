use num_traits::{Float, Zero};
use rand::Rng;
use std::{io::Error, iter};

#[derive(Debug)]
pub struct Shape {
    pub rows: usize,
    pub cols: usize,
}

impl Shape {
    fn parse(rows: usize, cols: usize) -> Result<Self, Error> {
        if rows == 0 || cols == 0 {
            return Err(Error::other(format!(
                "matrix shape dimensions cannot be zero: rows({rows}), cols({cols})",
            )));
        }

        Ok(Self { rows, cols })
    }

    fn validate_equal(shape_a: &Shape, shape_b: &Shape) -> Result<Self, Error> {
        if shape_a != shape_b {
            return Err(Error::other(format!(
                "mismatched shapes: ({} x {}), ({} x {})",
                shape_a.rows, shape_a.cols, shape_b.rows, shape_b.cols
            )));
        }

        Self::parse(shape_a.rows, shape_a.cols)
    }

    fn validate_for_dot_product(shape_a: &Shape, shape_b: &Shape) -> Result<Self, Error> {
        if shape_a.cols != shape_b.rows {
            return Err(Error::other(format!(
                "matrix A columns ({}) does not equal matrix B rows({})",
                shape_a.cols, shape_b.rows
            )));
        }

        Self::parse(shape_a.rows, shape_b.cols)
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols
    }
}

#[derive(Debug)]
pub struct Matrix<F: Float + Zero> {
    pub shape: Shape,
    elements: Vec<F>,
}

impl<F: Float + Zero> Matrix<F> {
    pub fn new(rows: usize, cols: usize, elements: Vec<F>) -> Result<Self, Error> {
        if elements.len() == 0 {
            return Err(Error::other(format!("elements array is empty")));
        }

        Ok(Self {
            shape: Shape::parse(rows, cols)?,
            elements,
        })
    }

    pub fn zeros(rows: usize, cols: usize) -> Result<Self, Error> {
        Ok(Self {
            shape: Shape::parse(rows, cols)?,
            elements: iter::repeat(F::zero()).take(rows * cols).collect(),
        })
    }

    pub fn randomize(mut self) -> Result<Self, Error> {
        use rand::thread_rng;
        let mut rng = thread_rng();

        for i in 0..self.shape.rows {
            for j in 0..self.shape.cols {
                self.set(i, j, F::from(rng.gen_range(-1f64..1f64)).unwrap())?;
            }
        }

        Ok(self)
    }

    fn get_index(&self, row: usize, col: usize) -> Result<usize, Error> {
        if row >= self.shape.rows {
            return Err(Error::other(format!(
                "row({row}) is out of bounds, limit is {}",
                self.shape.rows - 1
            )));
        }

        if col >= self.shape.cols {
            return Err(Error::other(format!(
                "col({col}) is out of bounds, limit is {}",
                self.shape.cols - 1
            )));
        }

        Ok(self.shape.cols * row + col)
    }

    pub fn get(&self, row: usize, col: usize) -> Result<F, Error> {
        let index = self.get_index(row, col)?;

        Ok(self.elements[index])
    }

    pub fn set(&mut self, row: usize, col: usize, value: F) -> Result<(), Error> {
        let index = self.get_index(row, col)?;

        self.elements[index] = value;
        Ok(())
    }

    pub fn add(&self, other: &Self) -> Result<Self, Error> {
        let shape = Shape::validate_equal(&self.shape, &other.shape)?;

        let output_elements = self
            .elements
            .iter()
            .zip(other.elements.iter())
            .map(|(a, b)| *a + *b)
            .collect();

        Ok(Matrix {
            shape,
            elements: output_elements,
        })
    }

    pub fn sub(&self, other: &Self) -> Result<Self, Error> {
        let shape = Shape::validate_equal(&self.shape, &other.shape)?;

        let output_elements = self
            .elements
            .iter()
            .zip(other.elements.iter())
            .map(|(a, b)| *a - *b)
            .collect();

        Ok(Matrix {
            shape,
            elements: output_elements,
        })
    }

    pub fn mul(&self, other: &Self) -> Result<Self, Error> {
        let shape = Shape::validate_equal(&self.shape, &other.shape)?;

        let output_elements = self
            .elements
            .iter()
            .zip(other.elements.iter())
            .map(|(a, b)| *a * *b)
            .collect();

        Ok(Matrix {
            shape,
            elements: output_elements,
        })
    }

    pub fn transpose(&self) -> Result<Self, Error> {
        let shape = Shape::parse(self.shape.rows, self.shape.cols)?;
        let mut output = Self::zeros(shape.cols, shape.rows)?;

        for row in 0..self.shape.rows {
            for col in 0..self.shape.cols {
                output.set(col, row, self.get(row, col)?)?;
            }
        }

        Ok(output)
    }

    pub fn dot(&self, other: &Self) -> Result<Self, Error> {
        let shape = Shape::validate_for_dot_product(&self.shape, &other.shape)?;
        let mut output = Self::zeros(shape.rows, shape.cols)?;

        for i in 0..shape.rows {
            for j in 0..shape.cols {
                output.set(
                    i,
                    j,
                    (0..other.shape.rows).fold(F::zero(), |acc: F, k| {
                        // Use unwrap since Results are returned inside the closure,
                        // potential for panicking but iter::fold is way faster than a for loop

                        acc + self.get(i, k).unwrap() * other.get(k, j).unwrap()
                    }),
                )?;
            }
        }

        Ok(output)
    }
}
