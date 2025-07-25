use criterion::{criterion_group, criterion_main, Criterion};
use keplerian_sim::sinhcosh;
use std::{hint::black_box, time::Duration};

const MIN_RANGE: f64 = -1000.0;
const MAX_RANGE: f64 = 1000.0;
const STEP_SIZE: f64 = 0.001;

fn benchmark(c: &mut Criterion) {
    c.measurement_time(Duration::from_secs(15));
    c.sample_size(500);

    c.bench_function("sinhcosh naive", |b| {
        b.iter(|| {
            let mut x = MIN_RANGE;
            while x < MAX_RANGE {
                black_box(x.sinh());
                black_box(x.cosh());
                x += STEP_SIZE;
            }
        })
    });

    c.bench_function("sinhcosh together", |b| {
        b.iter(|| {
            let mut x = MIN_RANGE;
            while x < MAX_RANGE {
                black_box(x.sinh());
                black_box(x.cosh());
                x += STEP_SIZE;
            }
        })
    });
}

criterion_group!(sinhcosh_bench, benchmark);
criterion_main!(sinhcosh_bench);
