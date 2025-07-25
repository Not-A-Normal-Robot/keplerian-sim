use criterion::{criterion_group, criterion_main, Criterion};
use keplerian_sim::sinhcosh;
use std::{hint::black_box, time::Duration};

const MIN_RANGE: f64 = -1000.0;
const MAX_RANGE: f64 = 1000.0;
const STEP_SIZE: f64 = 0.01;

fn benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sinhcosh");

    group
        .significance_level(0.1)
        .sample_size(1000)
        .warm_up_time(Duration::from_secs(8))
        .measurement_time(Duration::from_secs(20));

    group.bench_function("sinhcosh naive", |b| {
        b.iter(|| {
            let mut x = MIN_RANGE;
            while x < MAX_RANGE {
                black_box((x.sinh(), x.cosh()));
                x += STEP_SIZE;
            }
        })
    });

    group.bench_function("sinhcosh together", |b| {
        b.iter(|| {
            let mut x = MIN_RANGE;
            while x < MAX_RANGE {
                black_box(sinhcosh(x));
                x += STEP_SIZE;
            }
        })
    });
}

criterion_group!(sinhcosh_bench, benchmark);
criterion_main!(sinhcosh_bench);
