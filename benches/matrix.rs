use criterion::{criterion_group, criterion_main, Criterion};
use matrix::Matrix;

fn bench_4x4_matrix(c: &mut Criterion) {
    let m1 = Matrix::<f64>::zeros(4, 4).unwrap();
    let m2 = Matrix::<f64>::zeros(4, 4).unwrap();

    c.bench_function("matrix_add_4x4", |b| {
        b.iter(|| {
            m1.add(&m2).unwrap();
        });
    });

    c.bench_function("matrix_dot_4x4", |b| {
        b.iter(|| {
            m1.dot(&m2).unwrap();
        });
    });
}

fn bench_64x64_matrix(c: &mut Criterion) {
    let m1 = Matrix::<f64>::zeros(64, 64).unwrap();
    let m2 = Matrix::<f64>::zeros(64, 64).unwrap();

    c.bench_function("matrix_add_64x64", |b| {
        b.iter(|| {
            m1.add(&m2).unwrap();
        });
    });

    c.bench_function("matrix_dot_64x64", |b| {
        b.iter(|| {
            m1.dot(&m2).unwrap();
        });
    });
}

fn bench_256x256_matrix(c: &mut Criterion) {
    let m1 = Matrix::<f64>::zeros(256, 256).unwrap();
    let m2 = Matrix::<f64>::zeros(256, 256).unwrap();

    c.bench_function("matrix_add_256x256", |b| {
        b.iter(|| {
            m1.add(&m2).unwrap();
        });
    });

    c.bench_function("matrix_dot_256x256", |b| {
        b.iter(|| {
            m1.dot(&m2).unwrap();
        });
    });
}

fn bench_1024x1024_matrix(c: &mut Criterion) {
    let m1 = Matrix::<f64>::zeros(1024, 1024).unwrap();
    let m2 = Matrix::<f64>::zeros(1024, 1024).unwrap();

    c.bench_function("matrix_add_1024x1024", |b| {
        b.iter(|| {
            m1.add(&m2).unwrap();
        });
    });

    c.bench_function("matrix_dot_1024x1024", |b| {
        b.iter(|| {
            m1.dot(&m2).unwrap();
        });
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(50);
    targets = bench_4x4_matrix, bench_64x64_matrix, bench_256x256_matrix, bench_1024x1024_matrix
);
criterion_main!(benches);
