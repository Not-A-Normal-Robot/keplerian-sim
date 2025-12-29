use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use glam::DVec2;
use keplerian_sim::{CompactOrbit, CompactOrbit2D, Orbit, Orbit2D, OrbitTrait, OrbitTrait2D};
use std::hint::black_box;

const POLL_ITERS: u64 = 1024;

#[inline(always)]
fn poll_tf_3d(orbit: &impl OrbitTrait, points: &[DVec2]) {
    for &point in points {
        black_box(orbit.transform_pqw_vector(black_box(point)));
    }
}

#[inline(always)]
fn poll_tf_2d(orbit: &impl OrbitTrait2D, points: &[DVec2]) {
    for &point in points {
        black_box(orbit.transform_pqw_vector(black_box(point)));
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let orbit = Orbit::default();
    let compact = CompactOrbit::from(orbit.clone());
    let orbit2d = Orbit2D::default();
    let compact2d = CompactOrbit2D::from(orbit2d.clone());

    let points = (0..POLL_ITERS)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (POLL_ITERS as f64);
            DVec2::new(angle.cos(), angle.sin())
        })
        .collect::<Box<_>>();
    let points = black_box(points);

    let mut group = c.benchmark_group("pqw_transformation");
    group.throughput(Throughput::Elements(POLL_ITERS));

    group.bench_function("3d cached", |b| {
        b.iter(|| poll_tf_3d(black_box(&orbit), black_box(&points)))
    });
    group.bench_function("3d compact", |b| {
        b.iter(|| poll_tf_3d(black_box(&compact), black_box(&points)))
    });

    group.bench_function("2d cached", |b| {
        b.iter(|| poll_tf_2d(black_box(&orbit2d), black_box(&points)))
    });
    group.bench_function("3d compact", |b| {
        b.iter(|| poll_tf_2d(black_box(&compact2d), black_box(&points)))
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
