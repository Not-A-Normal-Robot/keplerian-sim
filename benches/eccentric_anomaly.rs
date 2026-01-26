use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use keplerian_sim::{CompactOrbit, CompactOrbit2D, Orbit, Orbit2D, OrbitTrait, OrbitTrait2D};
use std::hint::black_box;

const POLL_ITERS: u64 = 1024;
const MULTIPLIER: f64 = std::f64::consts::TAU / POLL_ITERS as f64;

#[inline(always)]
fn poll_ea_3d(orbit: &impl OrbitTrait) {
    for i in 0..POLL_ITERS {
        let angle = i as f64 * MULTIPLIER;
        black_box(orbit.get_eccentric_anomaly_at_mean_anomaly(black_box(angle)));
    }
}

#[inline(always)]
fn poll_ea_2d(orbit: &impl OrbitTrait2D) {
    for i in 0..POLL_ITERS {
        let angle = i as f64 * MULTIPLIER;
        black_box(orbit.get_eccentric_anomaly_at_mean_anomaly(black_box(angle)));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let orbit = Orbit::default();
    let orbit2d = Orbit2D::default();
    let compact = CompactOrbit::from(orbit.clone());
    let compact2d = CompactOrbit2D::from(orbit2d.clone());

    let hyperbolic = Orbit::new(2.9, 1.0, 2.19, 0.44, 0.61, 0.98, 1.0);
    let hyperbolic2d = Orbit2D::new(2.9, 1.0, 0.44, 0.98, 1.0);
    let compact_hyperbolic = CompactOrbit::from(hyperbolic.clone());
    let compact_hyperbolic2d = CompactOrbit2D::from(hyperbolic2d.clone());

    let mut group = c.benchmark_group("eccentric_anomaly@mean_anomaly");
    group.throughput(Throughput::Elements(POLL_ITERS));

    group.bench_function("3d elliptic cached", |b| {
        b.iter(|| poll_ea_3d(black_box(&orbit)))
    });
    group.bench_function("3d elliptic compact", |b| {
        b.iter(|| poll_ea_3d(black_box(&compact)))
    });
    group.bench_function("3d hyperbolic cached", |b| {
        b.iter(|| poll_ea_3d(black_box(&hyperbolic)))
    });
    group.bench_function("3d hyperbolic compact", |b| {
        b.iter(|| poll_ea_3d(black_box(&compact_hyperbolic)))
    });

    group.bench_function("2d elliptic cached", |b| {
        b.iter(|| poll_ea_2d(black_box(&orbit2d)))
    });
    group.bench_function("2d elliptic compact", |b| {
        b.iter(|| poll_ea_2d(black_box(&compact2d)))
    });
    group.bench_function("2d hyperbolic cached", |b| {
        b.iter(|| poll_ea_2d(black_box(&hyperbolic2d)))
    });
    group.bench_function("2d hyperbolic compact", |b| {
        b.iter(|| poll_ea_2d(black_box(&compact_hyperbolic2d)))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
