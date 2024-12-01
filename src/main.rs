use layer::Layer;
use matrix::Matrix;
use std::io::Error;

fn main() -> Result<(), Error> {
    let l1 = Layer::<f64>::zeros(3, 6)?.randomize()?;
    let l2 = Layer::<f64>::zeros(6, 4)?.randomize()?;
    let m1 = Matrix::<f64>::zeros(1, 3)?.randomize()?;

    let m2 = l1.forward(m1)?;
    let m3 = l2.forward(m2)?;

    println!("{:?}", m3);

    Ok(())
}
