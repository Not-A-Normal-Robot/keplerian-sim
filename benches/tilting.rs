use criterion::{criterion_group, criterion_main, Criterion};
use keplerian_sim::{CompactOrbit, Orbit, OrbitTrait};
use std::hint::black_box;

const POLL_ANGLES: usize = 1024;

#[inline(always)]
fn poll_tilt(orbit: &impl OrbitTrait, points: &[(f64, f64)]) {
    for point in points {
        black_box(orbit.tilt_flat_position(black_box(point.0), black_box(point.1)));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let orbit = Orbit::new_default();
    let points = (0..POLL_ANGLES)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (POLL_ANGLES as f64);
            (angle.cos(), angle.sin())
        })
        .collect::<Vec<_>>();
    let points = black_box(points);

    c.bench_function("tilt poll cached", |b| {
        b.iter(|| poll_tilt(black_box(&orbit), black_box(&points)))
    });

    let compact: CompactOrbit = orbit.into();

    c.bench_function("tilt poll compact", |b| {
        b.iter(|| poll_tilt(black_box(&compact), black_box(&points)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
