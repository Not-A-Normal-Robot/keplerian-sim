use core::f64::consts::TAU;

use crate::{CompactOrbit, CompactOrbit2D};

#[allow(dead_code)]
pub(super) fn random_mult() -> f64 {
    if rand::random_bool(0.5) {
        // Lower
        rand::random_range(0.1f64..0.9f64)
    } else {
        // Higher
        rand::random_range(1.1f64..5.0f64)
    }
}

pub(super) fn random_circular() -> CompactOrbit {
    CompactOrbit::new(
        0.0,
        rand::random_range(0.01..1e6),
        if rand::random_bool(0.5) {
            rand::random_range(-TAU..TAU)
        } else {
            0.0
        },
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_circular_2d() -> CompactOrbit2D {
    CompactOrbit2D::new(
        0.0,
        rand::random_range(0.01..1e6),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_elliptic() -> CompactOrbit {
    CompactOrbit::new(
        rand::random_range(0.01..0.99),
        rand::random_range(0.01..1e6),
        if rand::random_bool(0.5) {
            rand::random_range(-TAU..TAU)
        } else {
            0.0
        },
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_elliptic_2d() -> CompactOrbit2D {
    CompactOrbit2D::new(
        rand::random_range(0.01..0.99),
        rand::random_range(0.01..1e6),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_near_parabolic() -> CompactOrbit {
    CompactOrbit::new(
        rand::random_range(0.99..0.9999),
        rand::random_range(0.01..1e6),
        if rand::random_bool(0.5) {
            rand::random_range(-TAU..TAU)
        } else {
            0.0
        },
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_near_parabolic_2d() -> CompactOrbit2D {
    CompactOrbit2D::new(
        rand::random_range(0.99..0.9999),
        rand::random_range(0.01..1e6),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_parabolic() -> CompactOrbit {
    CompactOrbit::new(
        1.0,
        rand::random_range(0.01..1e6),
        if rand::random_bool(0.5) {
            rand::random_range(-TAU..TAU)
        } else {
            0.0
        },
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_parabolic_2d() -> CompactOrbit2D {
    CompactOrbit2D::new(
        1.0,
        rand::random_range(0.01..1e6),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_hyperbolic() -> CompactOrbit {
    CompactOrbit::new(
        rand::random_range(1.01..3.0),
        rand::random_range(0.01..1e6),
        if rand::random_bool(0.5) {
            rand::random_range(-TAU..TAU)
        } else {
            0.0
        },
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_hyperbolic_2d() -> CompactOrbit2D {
    CompactOrbit2D::new(
        rand::random_range(1.01..3.0),
        rand::random_range(0.01..1e6),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_very_hyperbolic() -> CompactOrbit {
    CompactOrbit::new(
        rand::random_range(5.0..15.0),
        rand::random_range(0.01..1e6),
        if rand::random_bool(0.5) {
            rand::random_range(-TAU..TAU)
        } else {
            0.0
        },
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_very_hyperbolic_2d() -> CompactOrbit2D {
    CompactOrbit2D::new(
        rand::random_range(5.0..15.0),
        rand::random_range(0.01..1e6),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_extremely_hyperbolic() -> CompactOrbit {
    CompactOrbit::new(
        rand::random_range(80.0..150.0),
        rand::random_range(0.01..1e6),
        if rand::random_bool(0.5) {
            rand::random_range(-TAU..TAU)
        } else {
            0.0
        },
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_extremely_hyperbolic_2d() -> CompactOrbit2D {
    CompactOrbit2D::new(
        rand::random_range(80.0..150.0),
        rand::random_range(0.01..1e6),
        rand::random_range(-TAU..TAU),
        rand::random_range(-TAU..TAU),
        rand::random_range(0.01..1e6),
    )
}

pub(super) fn random_nonparabolic() -> CompactOrbit {
    const FNS: &[fn() -> CompactOrbit] = &[
        random_circular,
        random_elliptic,
        random_near_parabolic,
        random_hyperbolic,
        random_very_hyperbolic,
        random_extremely_hyperbolic,
    ];

    let i = rand::random_range(0..FNS.len());

    FNS[i]()
}

pub(super) fn random_nonparabolic_iter(iters: usize) -> impl Iterator<Item = CompactOrbit> {
    (0..iters).into_iter().map(|_| random_nonparabolic())
}

pub(super) fn random_any() -> CompactOrbit {
    const FNS: &[fn() -> CompactOrbit] = &[
        random_circular,
        random_elliptic,
        random_near_parabolic,
        random_parabolic,
        random_hyperbolic,
        random_very_hyperbolic,
        random_extremely_hyperbolic,
    ];

    let i = rand::random_range(0..FNS.len());

    FNS[i]()
}

pub(super) fn random_any_iter(iters: usize) -> impl Iterator<Item = CompactOrbit> {
    (0..iters).into_iter().map(|_| random_any())
}

pub(super) fn random_any_2d() -> CompactOrbit2D {
    const FNS: &[fn() -> CompactOrbit2D] = &[
        random_circular_2d,
        random_elliptic_2d,
        random_near_parabolic_2d,
        random_parabolic_2d,
        random_hyperbolic_2d,
        random_very_hyperbolic_2d,
        random_extremely_hyperbolic_2d,
    ];

    let i = rand::random_range(0..FNS.len());

    FNS[i]()
}

pub(super) fn random_any_2d_iter(iters: usize) -> impl Iterator<Item = CompactOrbit2D> {
    (0..iters).map(|_| random_any_2d())
}
