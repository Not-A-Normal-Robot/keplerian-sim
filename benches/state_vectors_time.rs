use criterion::{criterion_group, criterion_main, Criterion};
use keplerian_sim::{CompactOrbit, Orbit, OrbitTrait};
use std::hint::black_box;

const POLL_ANGLES: usize = 1024;
const MULTIPLIER: f64 = 1.0 / POLL_ANGLES as f64;

#[inline(always)]
fn poll_sv(orbit: &impl OrbitTrait) {
    for i in 0..POLL_ANGLES {
        let angle = i as f64 * MULTIPLIER;
        black_box(orbit.get_state_vectors_at_time(black_box(angle)));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let orbit = Orbit::with_apoapsis(
        1.52097597e11,
        1.47098450e11,
        0.00005_f64.to_radians(),
        114.20783_f64.to_radians(),
        -11.26064_f64.to_radians(),
        358.617_f64.to_radians(),
        1.0,
    );

    c.bench_function("sv time poll cached", |b| {
        b.iter(|| poll_sv(black_box(&orbit)))
    });

    let compact: CompactOrbit = orbit.into();

    c.bench_function("sv time poll compact", |b| {
        b.iter(|| poll_sv(black_box(&compact)))
    });

    let orbit = Orbit::new(
        2.0,
        1.47098450e11,
        0.00005_f64.to_radians(),
        114.20783_f64.to_radians(),
        -11.26064_f64.to_radians(),
        358.617_f64.to_radians(),
        1.0,
    );

    c.bench_function("sv time poll hyp cached", |b| {
        b.iter(|| poll_sv(black_box(&orbit)))
    });

    let compact: CompactOrbit = orbit.into();

    c.bench_function("sv time poll hyp compact", |b| {
        b.iter(|| poll_sv(black_box(&compact)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
