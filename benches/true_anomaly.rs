use criterion::{criterion_group, criterion_main, Criterion};
use keplerian_sim::{CompactOrbit, Orbit, OrbitTrait};
use std::hint::black_box;

const POLL_ANGLES: usize = 1024;

#[inline(always)]
fn poll_true_cached(orbit: &Orbit) {
    let multiplier = std::f64::consts::TAU / POLL_ANGLES as f64;

    for i in 0..POLL_ANGLES {
        let angle = i as f64 * multiplier;
        black_box(orbit.get_true_anomaly(black_box(angle)));
    }
}

#[inline(always)]
fn poll_true_compact(orbit: &CompactOrbit) {
    let multiplier = std::f64::consts::TAU / POLL_ANGLES as f64;

    for i in 0..POLL_ANGLES {
        let angle = i as f64 * multiplier;
        black_box(orbit.get_true_anomaly(black_box(angle)));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let orbit = Orbit::default();

    c.bench_function("true poll cached", |b| {
        b.iter(|| poll_true_cached(black_box(&orbit)))
    });

    let compact: CompactOrbit = orbit.into();

    c.bench_function("true poll compact", |b| {
        b.iter(|| poll_true_compact(black_box(&compact)))
    });

    // hyperbolic
    let orbit = Orbit::new(2.4, 1.0, 0.98, 3.01, 1.01, 2.55, 1.0);

    c.bench_function("true poll hyp cached", |b| {
        b.iter(|| poll_true_cached(black_box(&orbit)))
    });

    let compact: CompactOrbit = orbit.into();

    c.bench_function("true poll hyp compact", |b| {
        b.iter(|| poll_true_compact(black_box(&compact)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
