use matrix::Matrix;
use std::io::Error;

fn main() -> Result<(), Error> {
    let m1 = Matrix::<f64>::zeros(10, 16)?.randomize()?;
    let m2 = Matrix::<f64>::zeros(16, 3)?.randomize()?;

    let m3 = m1.dot(&m2)?;
    println!("{:?}", m3);

    Ok(())
}
