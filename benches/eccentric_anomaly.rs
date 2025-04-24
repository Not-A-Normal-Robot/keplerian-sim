use criterion::{criterion_group, criterion_main, Criterion};
use keplerian_sim::{body_presets::MASS_SUN, CompactOrbit, Orbit, OrbitTrait};
use std::hint::black_box;

const POLL_ANGLES: usize = 1024;

#[inline(always)]
fn poll_ecc_cached(orbit: &Orbit) {
    let multiplier = std::f64::consts::TAU / POLL_ANGLES as f64;

    for i in 0..POLL_ANGLES {
        let angle = i as f64 * multiplier;
        black_box(orbit.get_eccentric_anomaly(black_box(angle)));
    }
}

#[inline(always)]
fn poll_ecc_compact(orbit: &CompactOrbit) {
    let multiplier = std::f64::consts::TAU / POLL_ANGLES as f64;

    for i in 0..POLL_ANGLES {
        let angle = i as f64 * multiplier;
        black_box(orbit.get_eccentric_anomaly(black_box(angle)));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let orbit = Orbit::default();

    c.bench_function("ecc poll cached", |b| {
        b.iter(|| poll_ecc_cached(black_box(&orbit)))
    });

    let compact: CompactOrbit = orbit.into();

    c.bench_function("ecc poll compact", |b| {
        b.iter(|| poll_ecc_compact(black_box(&compact)))
    });

    // hyperbolic orbit
    let orbit = Orbit::new(2.9, 1.0, 2.19, 0.44, 0.61, 0.98, MASS_SUN);

    c.bench_function("ecc poll hyp cached", |b| {
        b.iter(|| poll_ecc_cached(black_box(&orbit)))
    });

    let compact: CompactOrbit = orbit.into();

    c.bench_function("ecc poll hyp compact", |b| {
        b.iter(|| poll_ecc_compact(black_box(&compact)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
