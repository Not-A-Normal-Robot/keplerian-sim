use criterion::{criterion_group, criterion_main, Criterion};
use keplerian_sim::{CompactOrbit, Orbit, OrbitTrait};
use std::hint::black_box;

const POLL_ANGLES: usize = 1024;

#[inline(always)]
fn poll_vel(orbit: &impl OrbitTrait) {
    let multiplier = std::f64::consts::TAU / POLL_ANGLES as f64;

    for i in 0..POLL_ANGLES {
        let angle = i as f64 * multiplier;
        black_box(orbit.get_velocity_at_eccentric_anomaly(black_box(angle)));
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

    c.bench_function("vel poll cached", |b| {
        b.iter(|| poll_vel(black_box(&orbit)))
    });

    let compact: CompactOrbit = orbit.into();

    c.bench_function("vel poll compact", |b| {
        b.iter(|| poll_vel(black_box(&compact)))
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

    c.bench_function("vel poll hyp cached", |b| {
        b.iter(|| poll_vel(black_box(&orbit)))
    });

    let compact: CompactOrbit = orbit.into();

    c.bench_function("vel poll hyp compact", |b| {
        b.iter(|| poll_vel(black_box(&compact)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
