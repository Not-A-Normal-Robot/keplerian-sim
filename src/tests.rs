#![cfg(test)]

extern crate std;

use crate::{CompactOrbit, Orbit, OrbitTrait};
use std::f64::consts::{PI, TAU};

type Vec3 = (f64, f64, f64);
type Vec2 = (f64, f64);

const ALMOST_EQ_TOLERANCE: f64 = 1e-6;
const ORBIT_POLL_ANGLES: usize = 4096;

fn assert_almost_eq(a: f64, b: f64, what: &str) {
    let dist = (a - b).abs();
    let msg = format!(
        "Almost-eq assertion failed for '{what}'!\n\
        {a} and {b} has distance {dist}, which is more than max of {ALMOST_EQ_TOLERANCE}"
    );

    assert!(dist < ALMOST_EQ_TOLERANCE, "{msg}");
}

fn vec3_len(v: Vec3) -> f64 {
    return (v.0 * v.0 + v.1 * v.1 + v.2 * v.2).sqrt();
}

fn assert_eq_vec3(a: Vec3, b: Vec3, what: &str) {
    assert_eq!(a.0.to_bits(), b.0.to_bits(), "X coord of {what}");
    assert_eq!(a.1.to_bits(), b.1.to_bits(), "Y coord of {what}");
    assert_eq!(a.2.to_bits(), b.2.to_bits(), "Z coord of {what}");
}

fn assert_eq_vec2(a: Vec2, b: Vec2, what: &str) {
    assert_eq!(a.0.to_bits(), b.0.to_bits(), "X coord of {what}");
    assert_eq!(a.1.to_bits(), b.1.to_bits(), "Y coord of {what}");
}

fn assert_almost_eq_vec3(a: Vec3, b: Vec3, what: &str) {
    assert_almost_eq(a.0, b.0, &("X coord of ".to_string() + what));
    assert_almost_eq(a.1, b.1, &("Y coord of ".to_string() + what));
    assert_almost_eq(a.2, b.2, &("Z coord of ".to_string() + what));
}

fn assert_almost_eq_vec2(a: Vec2, b: Vec2, what: &str) {
    assert_almost_eq(a.0, b.0, &("X coord of ".to_string() + what));
    assert_almost_eq(a.1, b.1, &("Y coord of ".to_string() + what));
}

fn assert_orbit_positions_3d(orbit: &impl OrbitTrait, tests: &[(&str, f64, Vec3)]) {
    for (what, angle, expected) in tests.iter() {
        let pos = orbit.get_position_at_angle(*angle);
        assert_almost_eq_vec3(pos, *expected, what);
    }
}

fn assert_orbit_positions_2d(orbit: &impl OrbitTrait, tests: &[(&str, f64, Vec2)]) {
    for (what, angle, expected) in tests.iter() {
        let pos = orbit.get_flat_position_at_angle(*angle);
        assert_almost_eq_vec2(pos, *expected, what);
    }
}

fn poll_orbit(orbit: &impl OrbitTrait) -> Vec<Vec3> {
    let mut vec: Vec<Vec3> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
        vec.push(orbit.get_position_at_angle(angle));
    }

    return vec;
}
fn poll_flat(orbit: &impl OrbitTrait) -> Vec<Vec2> {
    let mut vec: Vec<Vec2> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
        vec.push(orbit.get_flat_position_at_angle(angle));
    }

    return vec;
}
fn poll_transform(orbit: &impl OrbitTrait) -> Vec<Vec3> {
    let mut vec: Vec<Vec3> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
        vec.push(orbit.tilt_flat_position(1.0 * angle.cos(), 1.0 * angle.sin()));
    }

    return vec;
}
fn poll_eccentric_anomaly(orbit: &impl OrbitTrait) -> Vec<f64> {
    let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
        vec.push(orbit.get_eccentric_anomaly(angle));
    }

    return vec;
}
fn poll_speed(orbit: &impl OrbitTrait) -> Vec<f64> {
    (0..ORBIT_POLL_ANGLES)
        .into_iter()
        .map(|i| (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64))
        .map(|t| orbit.get_speed_at_time(t))
        .collect()
}
fn poll_flat_vel(orbit: &impl OrbitTrait) -> Vec<Vec2> {
    (0..ORBIT_POLL_ANGLES)
        .into_iter()
        .map(|i| (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64))
        .map(|t| orbit.get_flat_velocity_at_time(t))
        .collect()
}
fn poll_vel(orbit: &impl OrbitTrait) -> Vec<Vec3> {
    (0..ORBIT_POLL_ANGLES)
        .into_iter()
        .map(|i| (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64))
        .map(|t| orbit.get_velocity_at_time(t))
        .collect()
}
fn unit_orbit() -> Orbit {
    return Orbit::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
}

mod random_orbit {
    use super::*;
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

    pub(super) fn random_circular() -> Orbit {
        Orbit::new(
            0.0,
            rand::random_range(0.01..1e6),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(0.01..1e6),
        )
    }

    pub(super) fn random_elliptic() -> Orbit {
        Orbit::new(
            rand::random_range(0.01..0.99),
            rand::random_range(0.01..1e6),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(0.01..1e6),
        )
    }

    pub(super) fn random_near_parabolic() -> Orbit {
        Orbit::new(
            rand::random_range(0.99..0.9999),
            rand::random_range(0.01..1e6),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(0.01..1e6),
        )
    }

    pub(super) fn random_parabolic() -> Orbit {
        Orbit::new(
            1.0,
            rand::random_range(0.01..1e6),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(0.01..1e6),
        )
    }

    pub(super) fn random_hyperbolic() -> Orbit {
        Orbit::new(
            rand::random_range(1.01..3.0),
            rand::random_range(0.01..1e6),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(0.01..1e6),
        )
    }

    pub(super) fn random_very_hyperbolic() -> Orbit {
        Orbit::new(
            rand::random_range(5.0..15.0),
            rand::random_range(0.01..1e6),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(0.01..1e6),
        )
    }

    pub(super) fn random_extremely_hyperbolic() -> Orbit {
        Orbit::new(
            rand::random_range(80.0..150.0),
            rand::random_range(0.01..1e6),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(-TAU..TAU),
            rand::random_range(0.01..1e6),
        )
    }

    pub(super) fn random_nonparabolic() -> Orbit {
        const FNS: &[fn() -> Orbit] = &[
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

    pub(super) fn random_nonparabolic_iter(iters: usize) -> impl Iterator<Item = Orbit> {
        (0..iters).into_iter().map(|_| random_nonparabolic())
    }

    pub(super) fn random_any() -> Orbit {
        const FNS: &[fn() -> Orbit] = &[
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

    pub(super) fn random_any_iter(iters: usize) -> impl Iterator<Item = Orbit> {
        (0..iters).into_iter().map(|_| random_any())
    }
}

use random_orbit::*;

#[test]
fn unit_orbit_angle_3d() {
    let orbit = unit_orbit();

    assert_orbit_positions_3d(
        &orbit,
        &[
            ("unit orbit 1", 0.0 * PI, (1.0, 0.0, 0.0)),
            ("unit orbit 2", 0.5 * PI, (0.0, 1.0, 0.0)),
            ("unit orbit 3", 1.0 * PI, (-1.0, 0.0, 0.0)),
            ("unit orbit 4", 1.5 * PI, (0.0, -1.0, 0.0)),
            ("unit orbit 5", 2.0 * PI, (1.0, 0.0, 0.0)),
        ],
    );
}

#[test]
fn unit_orbit_angle_2d() {
    let orbit = unit_orbit();

    assert_orbit_positions_2d(
        &orbit,
        &[
            ("unit orbit 1", 0.0 * PI, (1.0, 0.0)),
            ("unit orbit 2", 0.5 * PI, (0.0, 1.0)),
            ("unit orbit 3", 1.0 * PI, (-1.0, 0.0)),
            ("unit orbit 4", 1.5 * PI, (0.0, -1.0)),
            ("unit orbit 5", 2.0 * PI, (1.0, 0.0)),
        ],
    );
}

#[test]
fn unit_orbit_transformation() {
    // Test how the inclination and LAN tilts points in the orbit.
    // Since inclination is zero, it should not do anything.
    let orbit = unit_orbit();

    let tests = [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)];

    for point in tests {
        let transformed = orbit.tilt_flat_position(point.0, point.1);

        assert_eq!(transformed.0, point.0);
        assert_eq!(transformed.1, point.1);
        assert_eq!(transformed.2, 0.0);
    }
}

#[test]
fn tilted_equidistant() {
    let orbit = Orbit::new(
        0.0,
        1.0,
        2.848915582093,
        1.9520945821,
        2.1834987325,
        0.69482153021,
        1.0,
    );

    // Test for equidistance
    let points = poll_orbit(&orbit);

    for point in points {
        let distance = vec3_len(point);

        assert_almost_eq(distance, 1.0, "Distance");
    }
}

#[test]
fn tilted_90deg() {
    let orbit = Orbit::new(0.0, 1.0, PI / 2.0, 0.0, 0.0, 0.0, 1.0);

    // Transform test
    let tests = [
        // Before and after transformation
        (("Vector 1"), (1.0, 0.0), (1.0, 0.0, 0.0)),
        (("Vector 2"), (0.0, 1.0), (0.0, 0.0, 1.0)),
        (("Vector 3"), (-1.0, 0.0), (-1.0, 0.0, 0.0)),
        (("Vector 4"), (0.0, -1.0), (0.0, 0.0, -1.0)),
    ];

    for (what, point, expected) in tests.iter() {
        let transformed = orbit.tilt_flat_position(point.0, point.1);

        assert_almost_eq_vec3(transformed, *expected, what);
    }
}

#[test]
fn apoapsis_of_two() {
    let orbit = Orbit::with_apoapsis(2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let point_at_apoapsis = orbit.get_position_at_angle(PI);
    let point_at_periapsis = orbit.get_position_at_angle(0.0);

    assert_almost_eq_vec3(point_at_apoapsis, (-2.0, 0.0, 0.0), "Ap");
    assert_almost_eq_vec3(point_at_periapsis, (1.0, 0.0, 0.0), "Pe");
}

#[test]
fn huge_apoapsis() {
    let orbit = Orbit::with_apoapsis(10000.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let point_at_apoapsis = orbit.get_position_at_angle(PI);
    let point_at_periapsis = orbit.get_position_at_angle(0.0);

    assert_almost_eq_vec3(point_at_apoapsis, (-10000.0, 0.0, 0.0), "Ap");
    assert_almost_eq_vec3(point_at_periapsis, (1.0, 0.0, 0.0), "Pe");
}

const JUST_BELOW_ONE: f64 = 0.9999999999999999;

#[test]
fn almost_parabolic() {
    let orbit = Orbit::new(
        // The largest f64 below 1
        JUST_BELOW_ONE,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    );

    let eccentric_anomalies = poll_eccentric_anomaly(&orbit);

    for ecc in eccentric_anomalies {
        assert!(
            ecc.is_finite(),
            "Eccentric anomaly algorithm instability at near-parabolic edge case"
        );
    }

    let positions = poll_flat(&orbit);

    for pos in positions {
        assert!(
            pos.0.is_finite() && pos.1.is_finite(),
            "2D position algorithm instability at near-parabolic edge case"
        );
    }

    let positions = poll_orbit(&orbit);

    for pos in positions {
        assert!(
            pos.0.is_finite() && pos.1.is_finite() && pos.2.is_finite(),
            "3D position algorithm instability at near-parabolic edge case"
        );
    }

    let position_at_periapsis = orbit.get_position_at_angle(TAU);

    assert_almost_eq_vec3(position_at_periapsis, (1.0, 0.0, 0.0), "Periapsis");

    let position_at_apoapsis = orbit.get_position_at_angle(PI);

    assert!(
        position_at_apoapsis.0.abs() > 1e12,
        "Apoapsis is not far enough"
    );
}

#[test]
fn parabolic() {
    let orbit = Orbit::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let point_near_infinity = orbit.get_position_at_angle(PI - 1e-7);
    let point_at_periapsis = orbit.get_position_at_angle(0.0);

    assert!(
        vec3_len(point_near_infinity) > 1e9,
        "Point near infinity is not far enough"
    );
    assert!(
        point_near_infinity.1.abs() > 0.0,
        "Y coord near infinity should move a little"
    );
    assert_almost_eq(
        point_near_infinity.2,
        0.0,
        "Point near infinity should be flat",
    );
    assert_almost_eq_vec3(point_at_periapsis, (1.0, 0.0, 0.0), "Pe");

    let point_at_asymptote = orbit.get_position_at_angle(PI);

    assert!(
        point_at_asymptote.0.is_nan(),
        "X at asymptote should be undefined"
    );
    assert!(
        point_at_asymptote.1.is_nan(),
        "Y at asymptote should be undefined"
    );
    assert!(
        point_at_asymptote.2.is_nan(),
        "Z at asymptote should be undefined"
    );
}

fn orbit_conversion_base_test(orbit: Orbit, what: &str) {
    let compact_orbit = CompactOrbit::from(orbit.clone());
    let reexpanded_orbit = Orbit::from(compact_orbit.clone());

    let compact_message = format!("Original / Compact ({what})");
    let reexpanded_message = format!("Compact /  Reexpanded ({what})");

    {
        let original_transforms = poll_transform(&orbit);
        let compact_transforms = poll_transform(&compact_orbit);
        let reexpanded_transforms = poll_transform(&reexpanded_orbit);

        let compact_message = format!("{compact_message} (transform)");
        let reexpanded_message = format!("{reexpanded_message} (transform)");

        for i in 0..original_transforms.len() {
            assert_eq_vec3(
                original_transforms[i],
                compact_transforms[i],
                &compact_message,
            );
            assert_eq_vec3(
                original_transforms[i],
                reexpanded_transforms[i],
                &reexpanded_message,
            );
        }
    }
    {
        let original_ecc = poll_eccentric_anomaly(&orbit);
        let compact_ecc = poll_eccentric_anomaly(&compact_orbit);
        let reexpanded_ecc = poll_eccentric_anomaly(&reexpanded_orbit);

        for i in 0..original_ecc.len() {
            assert_eq!(
                original_ecc[i], compact_ecc[i],
                "{compact_message} (eccentric anomaly)"
            );
            assert_eq!(
                original_ecc[i], reexpanded_ecc[i],
                "{reexpanded_message} (eccentric anomaly)"
            );
        }
    }
    {
        let original_true = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let true_anomaly = orbit.get_true_anomaly(angle);
                vec.push(true_anomaly);
            }

            vec
        };
        let compact_true = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let true_anomaly = compact_orbit.get_true_anomaly(angle);
                vec.push(true_anomaly);
            }

            vec
        };
        let reexpanded_true = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                vec.push(reexpanded_orbit.get_true_anomaly(angle));
            }

            vec
        };

        for i in 0..original_true.len() {
            assert_eq!(
                original_true[i].to_bits(),
                compact_true[i].to_bits(),
                "{compact_message} (true anomaly) (i={i})"
            );
            assert_eq!(
                original_true[i].to_bits(),
                reexpanded_true[i].to_bits(),
                "{reexpanded_message} (true anomaly) (i={i})"
            );
        }
    }
    {
        let original_eccentricity = orbit.get_eccentricity();
        let compact_eccentricity = compact_orbit.eccentricity;
        let reexpanded_eccentricity = reexpanded_orbit.get_eccentricity();

        assert_eq!(
            original_eccentricity, compact_eccentricity,
            "{compact_message} (eccentricity)"
        );
        assert_eq!(
            original_eccentricity, reexpanded_eccentricity,
            "{reexpanded_message} (eccentricity)"
        );
    }
    {
        let original_flat = poll_flat(&orbit);
        let compact_flat = poll_flat(&compact_orbit);
        let reexpanded_flat = poll_flat(&reexpanded_orbit);

        let compact_message = format!("{compact_message} (flat)");
        let reexpanded_message = format!("{reexpanded_message} (flat)");

        for i in 0..original_flat.len() {
            assert_eq_vec2(original_flat[i], compact_flat[i], &compact_message);
            assert_eq_vec2(original_flat[i], reexpanded_flat[i], &reexpanded_message);
        }
    }
    {
        let original_positions = poll_orbit(&orbit);
        let compact_positions = poll_orbit(&compact_orbit);
        let reexpanded_positions = poll_orbit(&reexpanded_orbit);

        let compact_message = format!("{compact_message} (position)");
        let reexpanded_message = format!("{reexpanded_message} (position)");

        for i in 0..original_positions.len() {
            assert_eq_vec3(
                original_positions[i],
                compact_positions[i],
                &compact_message,
            );
            assert_eq_vec3(
                original_positions[i],
                reexpanded_positions[i],
                &reexpanded_message,
            );
        }
    }
    {
        let original_altitudes = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let altitude = orbit.get_altitude_at_angle(angle);
                vec.push(altitude);
            }

            vec
        };
        let compact_altitudes = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let altitude = compact_orbit.get_altitude_at_angle(angle);
                vec.push(altitude);
            }

            vec
        };
        let reexpanded_altitudes = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let altitude = reexpanded_orbit.get_altitude_at_angle(angle);
                vec.push(altitude);
            }

            vec
        };

        let compact_message = format!("{compact_message} (altitude)");
        let reexpanded_message = format!("{reexpanded_message} (altitude)");

        for i in 0..original_altitudes.len() {
            assert_eq!(
                original_altitudes[i].to_bits(),
                compact_altitudes[i].to_bits(),
                "{compact_message}"
            );
            assert_eq!(
                original_altitudes[i].to_bits(),
                reexpanded_altitudes[i].to_bits(),
                "{reexpanded_message}"
            );
        }
    }
    {
        let original_slr = orbit.get_semi_latus_rectum();
        let compact_slr = compact_orbit.get_semi_latus_rectum();
        let reexpanded_slr = reexpanded_orbit.get_semi_latus_rectum();

        let compact_message = format!("{compact_message} (semi-latus rectum)");
        let reexpanded_message = format!("{reexpanded_message} (semi-latus rectum)");

        assert_eq!(
            original_slr.to_bits(),
            compact_slr.to_bits(),
            "{compact_message}"
        );
        assert_eq!(
            original_slr.to_bits(),
            reexpanded_slr.to_bits(),
            "{reexpanded_message}"
        );
    }
    {
        let original_apoapsis = orbit.get_apoapsis();
        let compact_apoapsis = compact_orbit.get_apoapsis();
        let reexpanded_apoapsis = reexpanded_orbit.get_apoapsis();

        let compact_message = format!("{compact_message} (apoapsis getter)");
        let reexpanded_message = format!("{reexpanded_message} (apoapsis getter)");

        assert_eq!(
            original_apoapsis.to_bits(),
            compact_apoapsis.to_bits(),
            "{compact_message}"
        );
        assert_eq!(
            original_apoapsis.to_bits(),
            reexpanded_apoapsis.to_bits(),
            "{reexpanded_message}"
        );
    }
    {
        for i in 0..31 {
            let true_anomaly = i as f64 * 0.1;

            let original_ecc_anom = orbit.get_eccentric_anomaly_at_true_anomaly(true_anomaly);
            let compact_ecc_anom =
                compact_orbit.get_eccentric_anomaly_at_true_anomaly(true_anomaly);
            let reexpanded_ecc_anom =
                reexpanded_orbit.get_eccentric_anomaly_at_true_anomaly(true_anomaly);

            let compact_message = format!("{compact_message} (true anom -> ecc anom)");
            let reexpanded_message = format!("{reexpanded_message} (true anom -> ecc anom)");

            assert_eq!(
                original_ecc_anom.to_bits(),
                compact_ecc_anom.to_bits(),
                "{compact_message}"
            );
            assert_eq!(
                original_ecc_anom.to_bits(),
                reexpanded_ecc_anom.to_bits(),
                "{reexpanded_message}"
            );
        }
    }
    {
        let original_speeds = poll_speed(&orbit);
        let compact_speeds = poll_speed(&compact_orbit);
        let reexpanded_speeds = poll_speed(&reexpanded_orbit);

        let compact_message = format!("{compact_message} (speed)");
        let reexpanded_message = format!("{reexpanded_message} (speed)");

        assert_eq!(
            original_speeds
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            compact_speeds
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            "{compact_message}",
        );
        assert_eq!(
            original_speeds
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            reexpanded_speeds
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            "{reexpanded_message}",
        );
    }
    {
        let original_fvels = poll_flat_vel(&orbit);
        let compact_fvels = poll_flat_vel(&compact_orbit);
        let reexpanded_fvels = poll_flat_vel(&reexpanded_orbit);

        let compact_message = format!("{compact_message} (flat velocity)");
        let reexpanded_message = format!("{reexpanded_message} (flat velocity)");

        assert_eq!(
            original_fvels
                .iter()
                .map(|(x, y)| (x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>(),
            compact_fvels
                .iter()
                .map(|(x, y)| (x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>(),
            "{compact_message}",
        );
        assert_eq!(
            original_fvels
                .iter()
                .map(|(x, y)| (x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>(),
            reexpanded_fvels
                .iter()
                .map(|(x, y)| (x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>(),
            "{reexpanded_message}",
        );
    }
    {
        let original_vels = poll_vel(&orbit);
        let compact_vels = poll_vel(&compact_orbit);
        let reexpanded_vels = poll_vel(&reexpanded_orbit);

        let compact_message = format!("{compact_message} (velocity)");
        let reexpanded_message = format!("{reexpanded_message} (velocity)");

        assert_eq!(
            original_vels
                .iter()
                .map(|(x, y, z)| (x.to_bits(), y.to_bits(), z.to_bits()))
                .collect::<Vec<_>>(),
            compact_vels
                .iter()
                .map(|(x, y, z)| (x.to_bits(), y.to_bits(), z.to_bits()))
                .collect::<Vec<_>>(),
            "{compact_message}",
        );
        assert_eq!(
            original_vels
                .iter()
                .map(|(x, y, z)| (x.to_bits(), y.to_bits(), z.to_bits()))
                .collect::<Vec<_>>(),
            reexpanded_vels
                .iter()
                .map(|(x, y, z)| (x.to_bits(), y.to_bits(), z.to_bits()))
                .collect::<Vec<_>>(),
            "{reexpanded_message}",
        );
    }
}

fn speed_velocity_base_test(orbit: &impl OrbitTrait, what: &str) {
    for i in 0..ORBIT_POLL_ANGLES {
        let t = (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64);

        let speed = orbit.get_speed_at_time(t);
        let flat_vel = orbit.get_flat_velocity_at_time(t);
        let vel = orbit.get_velocity_at_time(t);

        let flat_vel_mag = (flat_vel.0.powi(2) + flat_vel.1.powi(2)).sqrt();
        let vel_mag = (vel.0.powi(2) + vel.1.powi(2) + vel.2.powi(2)).sqrt();

        assert_almost_eq(
            speed,
            flat_vel_mag,
            &format!("Speed and flat velocity magnitude on orbit {what} at i={i}, t={t}"),
        );

        assert_almost_eq(
            speed,
            vel_mag,
            &format!("Speed and velocity magnitude on orbit {what} at i={i}, t={t}"),
        );
    }
}

fn naive_speed_correlation_base_test(orbit: &impl OrbitTrait, what: &str) {
    let mut coeff = None;

    fn cosine_similarity(v1: Vec3, v2: Vec3) -> f64 {
        let dot_product = v1.0 * v2.0 + v1.1 * v2.1 + v1.2 * v2.2;
        let v1_mag = (v1.0.powi(2) + v1.1.powi(2) + v1.2.powi(2)).sqrt();
        let v2_mag = (v2.0.powi(2) + v2.1.powi(2) + v2.2.powi(2)).sqrt();

        dot_product / (v1_mag * v2_mag)
    }

    for i in 0..ORBIT_POLL_ANGLES {
        let t = (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64);

        let offset = match orbit.get_eccentricity() {
            x if x > 0.95 && x < 1.05 => (x - 1.0).abs().powi(2) + 0.001,
            _ => 0.01,
        };

        let t2 = (i as f64 + offset) * TAU / (ORBIT_POLL_ANGLES as f64);

        let pos1 = orbit.get_position_at_time(t);
        let pos2 = orbit.get_position_at_time(t2);
        let diff = (pos2.0 - pos1.0, pos2.1 - pos1.1, pos2.2 - pos1.2);

        let vel = orbit.get_velocity_at_time(t);

        let direction_similarity = cosine_similarity(vel, diff);

        const SIMILARITY_THRESHOLD: f64 = 0.99;

        assert!(
            direction_similarity > SIMILARITY_THRESHOLD,
            "Velocity and position difference direction similarity is {direction_similarity}, \
            which is below threshold of {SIMILARITY_THRESHOLD}\n\
            On orbit {what}\n\
            At i={i}, t={t}, t2={t2}\n\n\
            pos1 = {pos1:?}\npos2 = {pos2:?}\ndiff = {diff:?}\nvel = {vel:?}"
        );

        let diff_mag = (diff.0.powi(2) + diff.1.powi(2) + diff.2.powi(2)).sqrt();
        let vel_mag = (vel.0.powi(2) + vel.1.powi(2) + vel.2.powi(2)).sqrt();
        let new_coeff = offset * diff_mag / vel_mag;

        match coeff {
            Some(coeff) => {
                match orbit.get_eccentricity() {
                    e if (e - 1.0).abs() > 0.01 => assert_almost_eq(
                        coeff,
                        new_coeff,
                        &format!("coefficient between diff_mag and vel_mag on orbit {what} at i={i}, t={t}, t2={t2}"),
                    ),
                    _e => {
                        // println!("coefficient between diff_mag and vel_mag check skipped (e = {_e})");
                    }
                }
            }
            None if new_coeff != 0.0 => {
                coeff = Some(new_coeff);
            }
            None => (),
        }
    }
}

// TODO: Add a unit test for this when the feature is implemented
fn _orbit_mu_setter_base_test(orbit: impl OrbitTrait + Clone) {
    for _ in 0..1024 {
        let _after = {
            let mut o = orbit.clone();
            o.set_gravitational_parameter(
                orbit.get_gravitational_parameter() * random_mult(),
                crate::MuSetterMode::KeepElements,
            );
            o
        };
    }

    for i in 0..1024 {
        let time = i as f64 * 0.15f64;
        let mut o = orbit.clone();
        let pos_before = orbit.get_position_at_time(time);
        o.set_gravitational_parameter(
            orbit.get_gravitational_parameter() * random_mult(),
            crate::MuSetterMode::KeepPositionAtTime(time),
        );
        let pos_after = orbit.get_position_at_time(time);
        assert_almost_eq_vec3(
            pos_before,
            pos_after,
            "Positions before and after mu setter",
        );
    }

    for i in 0..1024 {
        let time = i as f64 * 0.15f64;
        let mut o = orbit.clone();
        let pos_before = orbit.get_position_at_time(time);
        let vel_before = orbit.get_velocity_at_time(time);
        o.set_gravitational_parameter(
            orbit.get_gravitational_parameter() * random_mult(),
            crate::MuSetterMode::KeepPositionAndVelocityAtTime(time),
        );
        let pos_after = orbit.get_position_at_time(time);
        let vel_after = orbit.get_velocity_at_time(time);
        assert_almost_eq_vec3(
            pos_before,
            pos_after,
            "Positions before and after mu setter",
        );
        assert_almost_eq_vec3(
            vel_before,
            vel_after,
            "Velocities before and after mu setter",
        )
    }

    for i in 0..1024 {
        let angle = i as f64 * 0.15f64;
        let mut o = orbit.clone();
        let pos_before = orbit.get_position_at_angle(angle);
        o.set_gravitational_parameter(
            o.get_gravitational_parameter() * random_mult(),
            crate::MuSetterMode::KeepPositionAtAngle(angle),
        );
        let pos_after = orbit.get_position_at_angle(angle);

        assert_almost_eq_vec3(
            pos_before,
            pos_after,
            "Positions before and after mu setter",
        );
    }

    for i in 0..1024 {
        let angle = i as f64 * 0.15f64;
        let mut o = orbit.clone();
        let pos_before = orbit.get_position_at_angle(angle);
        let vel_before = orbit.get_velocity_at_angle(angle);
        o.set_gravitational_parameter(
            orbit.get_gravitational_parameter() * random_mult(),
            crate::MuSetterMode::KeepPositionAndVelocityAtAngle(angle),
        );
        let pos_after = orbit.get_position_at_angle(angle);
        let vel_after = orbit.get_velocity_at_angle(angle);
        assert_almost_eq_vec3(
            pos_before,
            pos_after,
            "Positions before and after mu setter",
        );
        assert_almost_eq_vec3(
            vel_before,
            vel_after,
            "Velocities before and after mu setter",
        );
    }
    todo!("Orbit mu setter test");
}

#[test]
fn orbit_conversions() {
    let orbits = [
        ("Unit orbit", Orbit::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        (
            "Mildly eccentric orbit",
            Orbit::new(0.39, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Very eccentric orbit",
            Orbit::new(0.99, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Just below parabolic orbit",
            Orbit::new(JUST_BELOW_ONE, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Parabolic trajectory",
            Orbit::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Barely hyperbolic trajectory",
            Orbit::new(1.01, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Very hyperbolic trajectory",
            Orbit::new(9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Extremely hyperbolic trajectory",
            Orbit::new(100.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Tilted orbit",
            Orbit::new(
                0.0,
                1.0,
                2.848915582093,
                1.9520945821,
                2.1834987325,
                0.69482153021,
                1.0,
            ),
        ),
        (
            "Tilted eccentric",
            Orbit::new(
                0.39,
                1.0,
                2.848915582093,
                1.9520945821,
                2.1834987325,
                0.69482153021,
                1.0,
            ),
        ),
        (
            "Tilted near-parabolic",
            Orbit::new(
                JUST_BELOW_ONE,
                1.0,
                2.848915582093,
                1.9520945821,
                2.1834987325,
                0.69482153021,
                1.0,
            ),
        ),
        (
            "Tilted parabolic",
            Orbit::new(
                1.0,
                1.0,
                2.848915582093,
                1.9520945821,
                2.1834987325,
                0.69482153021,
                1.0,
            ),
        ),
        (
            "Tilted hyperbolic",
            Orbit::new(
                1.8,
                1.0,
                2.848915582093,
                1.9520945821,
                2.1834987325,
                0.69482153021,
                1.0,
            ),
        ),
    ];

    for (what, orbit) in orbits.iter() {
        orbit_conversion_base_test(orbit.clone(), what);
    }

    for orbit in random_any_iter(1000) {
        let message = &format!("Random orbit ({orbit:?})");
        orbit_conversion_base_test(orbit, message);
    }
}

#[test]
fn test_sinh_approx_lt5() {
    use crate::generated_sinh_approximator::sinh_approx_lt5;
    use std::fs;

    const OUTPUT_CSV_PATH: &str = "out/test_sinh_approx_lt5.csv";
    const GRANULARITY: f64 = 0.001;
    const ITERATIONS: usize = (5.0f64 / GRANULARITY) as usize;
    const MAX_ACCEPTABLE_DIFF: f64 = 1e-5;
    // The paper says that the relative error for
    // F in [0, 5) is "significantly small", saying
    // that its maximum is around 5.14e-8. It is unclear
    // whether this is using infinite precision or double precision.
    const MAX_ACCEPTABLE_ERR: f64 = 6e-8;

    let mut approx_data = Vec::with_capacity(ITERATIONS);
    let mut real_data = Vec::with_capacity(ITERATIONS);
    let mut diff_data = Vec::with_capacity(ITERATIONS);
    let mut err_data = Vec::with_capacity(ITERATIONS);

    let iterator = (0..ITERATIONS).map(|i| i as f64 * GRANULARITY);

    for x in iterator {
        let approx = sinh_approx_lt5(x);
        let real = x.sinh();

        let diff = (approx - real).abs();
        let relative_err = diff / real;

        approx_data.push(approx);
        real_data.push(real);
        diff_data.push(diff);
        err_data.push(relative_err);
    }

    let mut csv = String::new();

    csv += "x,approx,real,diff,err\n";

    for i in 0..ITERATIONS {
        csv += &format!(
            "{x},{approx},{real},{diff},{err}\n",
            x = i as f64 * GRANULARITY,
            approx = approx_data[i],
            real = real_data[i],
            diff = diff_data[i],
            err = err_data[i],
        );
    }

    let res = fs::write(OUTPUT_CSV_PATH, csv);

    if let Err(e) = res {
        println!("Failed to write CSV: {e}");
    }

    for i in 0..ITERATIONS {
        assert!(
            diff_data[i] < MAX_ACCEPTABLE_DIFF,
            "Approx of sinh strayed too far at x={x} \
            (approx={approx}, real={real}), with distance {diff}\n\n\
            A CSV file has been written to '{OUTPUT_CSV_PATH}'",
            x = i as f64 * GRANULARITY,
            approx = approx_data[i],
            real = real_data[i],
            diff = diff_data[i]
        );
        assert!(
            real_data[i] == 0.0 || err_data[i] < MAX_ACCEPTABLE_ERR,
            "Approx of sinh strayed too far at x={x} \n\
            (approx={approx}, real={real}), with error {err}\n\n\
            A CSV file has been written to '{OUTPUT_CSV_PATH}'",
            x = i as f64 * GRANULARITY,
            approx = approx_data[i],
            real = real_data[i],
            err = err_data[i],
        );
    }
}

fn keplers_equation_hyperbolic(
    mean_anomaly: f64,
    eccentric_anomaly: f64,
    eccentricity: f64,
) -> f64 {
    return eccentricity * eccentric_anomaly.sinh() - eccentric_anomaly - mean_anomaly;
}

/// Use binary search to get the real hyperbolic eccentric anomaly instead of Newton's method.
///
/// We can do this because the hyperbolic KE is a monotonic function.
fn slowly_get_real_hyperbolic_eccentric_anomaly(orbit: &Orbit, mean_anomaly: f64) -> f64 {
    use keplers_equation_hyperbolic as ke;
    let eccentricity = orbit.get_eccentricity();

    let mut low = -1.0;
    let mut high = 1.0;

    {
        // Expand bounds until ke(low) < 0 and ke(high) > 0
        loop {
            let low_value = ke(mean_anomaly, low, eccentricity);

            if low_value < 0.0 {
                break;
            }

            low *= 2.0;
        }

        loop {
            let high_value = ke(mean_anomaly, high, eccentricity);

            if high_value > 0.0 {
                break;
            }

            high *= 2.0;
        }
    }

    let mut iters = 0u64;
    let max_expected_iters = 4096u64;

    loop {
        iters += 1;

        let midpoint = 0.5 * (low + high);
        let midvalue = ke(mean_anomaly, midpoint, eccentricity);

        if midvalue > 0.0 {
            high = midpoint;
        } else {
            low = midpoint;
        }

        let midpoint = 0.5 * (low + high);

        if midpoint == 0.0 || midpoint == low || midpoint == high {
            if midvalue.abs() > 1e-6 {
                panic!(
                    "Binary search failed to converge to a solution.\n\
                ... Orbit: {orbit:?}\n\
                ... Mean anomaly: {mean_anomaly}\n\
                ... Low bound: {low}\n\
                ... High bound: {high}\n\
                ... Midpoint: {midpoint}\n\
                ... Value: {midvalue}"
                );
            }

            return midpoint;
        }

        if iters > max_expected_iters {
            panic!(
                "Too many iterations. There might be a error in the binary search algorithm.\n\
            ... Orbit: {orbit:?}\n\
            ... Mean anomaly: {mean_anomaly}\n\
            ... Low bound: {low}\n\
            ... High bound: {high}"
            );
        }
    }
}

#[test]
fn test_hyperbolic_eccentric_anomaly() {
    let orbits = [
        (
            "Normal parabolic",
            Orbit::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Normal hyperbolic",
            Orbit::new(1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Extreme hyperbolic",
            Orbit::new(100.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
    ];

    struct Situation {
        orbit_type: &'static str,
        iteration_num: usize,
        angle: f64,
        deviation: f64,
        value: f64,
        expected: f64,
    }

    let mut situation_at_max_deviation = Situation {
        orbit_type: "",
        iteration_num: 0,
        angle: 0.0,
        deviation: 0.0,
        value: 0.0,
        expected: 0.0,
    };

    const ORBIT_POLL_ANGLES: usize = 128;

    for (what, orbit) in orbits.iter() {
        println!("Testing orbit type: {}", what);
        for i in 0..ORBIT_POLL_ANGLES {
            let angle =
                (i as f64 - 0.5 * ORBIT_POLL_ANGLES as f64) * TAU / (ORBIT_POLL_ANGLES as f64);

            let angle = 4.0 * angle.powi(3); // Test a wider range of angles

            let ecc_anom = orbit.get_eccentric_anomaly(angle);

            let expected = slowly_get_real_hyperbolic_eccentric_anomaly(orbit, angle);

            let deviation = (ecc_anom - expected).abs();

            if deviation > situation_at_max_deviation.deviation {
                situation_at_max_deviation = Situation {
                    orbit_type: what,
                    iteration_num: i,
                    angle,
                    deviation,
                    value: ecc_anom,
                    expected,
                };
            }
        }
    }

    assert!(
        situation_at_max_deviation.deviation < 1e-6,
        "Hyp. ecc. anom. deviates too much from stable amount \
        at iteration {}, {} rad\n\
        ... Orbit type: {}\n\
        ... Deviation: {}\n\
        ... Value: {}\n\
        ... Expected: {}",
        situation_at_max_deviation.iteration_num,
        situation_at_max_deviation.angle,
        situation_at_max_deviation.orbit_type,
        situation_at_max_deviation.deviation,
        situation_at_max_deviation.value,
        situation_at_max_deviation.expected,
    );

    println!(
        "Test success!\n\
        Max deviation: {:?}\n\
        ... Orbit type: {}\n\
        ... Iteration number: {}",
        situation_at_max_deviation.deviation,
        situation_at_max_deviation.orbit_type,
        situation_at_max_deviation.iteration_num,
    );
}

fn test_true_anom_to_ecc_anom_base(what: &str, orbit: &impl OrbitTrait) {
    for i in -100..100 {
        let true_anomaly = i as f64 * 0.1;

        let ecc_anom = orbit.get_eccentric_anomaly_at_true_anomaly(true_anomaly);

        if ecc_anom.is_nan() && orbit.get_eccentricity() >= 1.0 {
            // Some true anomalies just aren't possible in hyperbolic trajectories
            continue;
        }

        let reconverted_true_anomaly = orbit.get_true_anomaly_at_eccentric_anomaly(ecc_anom);

        let message = format!("True -> Ecc -> True anomaly conversion for {what}, at iter {i} and angle {true_anomaly}");
        assert_almost_eq(
            true_anomaly.rem_euclid(TAU),
            reconverted_true_anomaly.rem_euclid(TAU),
            message.as_str(),
        );
    }
}

#[test]
fn test_true_anom_to_ecc_anom() {
    let orbits = [
        ("Unit orbit", Orbit::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)),
        (
            "Mildly eccentric orbit",
            Orbit::new(0.39, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Very eccentric orbit",
            Orbit::new(0.99, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Just below parabolic orbit",
            Orbit::new(JUST_BELOW_ONE, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        // TODO: POST-PARABOLIC SUPPORT: Uncomment this when parabolic support is properly implemented
        // (
        //     "Parabolic trajectory",
        //     Orbit::new(
        //         1.0,
        //         1.0,
        //         0.0,
        //         0.0,
        //         0.0,
        //         0.0,1.0
        //     )
        // ),
        (
            "Barely hyperbolic trajectory",
            Orbit::new(1.01, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Very hyperbolic trajectory",
            Orbit::new(9.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
        (
            "Extremely hyperbolic trajectory",
            Orbit::new(100.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        ),
    ];

    for (what, orbit) in orbits.iter() {
        test_true_anom_to_ecc_anom_base(what, orbit);
    }
}

#[test]
fn test_semi_latus_rectum() {
    let orbits = [
        (
            "Circular",
            Orbit::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            1.0,
        ),
        (
            "Large Circular",
            Orbit::new(0.0, 192.168001001, 0.0, 0.0, 0.0, 0.0, 1.0),
            192.168001001,
        ),
        (
            "Elliptic",
            Orbit::new(0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            1.5,
        ),
        (
            "Large Elliptic",
            Orbit::new(0.5, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            150.0,
        ),
        (
            "Parabolic",
            Orbit::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            2.0,
        ),
        (
            "Large Parabolic",
            Orbit::new(1.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            200.0,
        ),
        (
            "Hyperbolic",
            Orbit::new(2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            3.0,
        ),
        (
            "Large Hyperbolic",
            Orbit::new(2.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            300.0,
        ),
    ];

    for (kind, orbit, expected) in orbits {
        let semi_latus_rectum = orbit.get_semi_latus_rectum();
        assert_almost_eq(
            semi_latus_rectum,
            expected,
            &format!("Semi-latus rectum of {} orbit", kind),
        );
    }
}

#[test]
fn test_altitude() {
    let orbits = [
        (
            "Circular",
            Orbit::new(
                0.0,
                6.1378562542109725,
                1.4755776307550952,
                0.3267764431411045,
                0.6031097400880272,
                0.7494917839241119,
                1.0,
            ),
        ),
        (
            "Elliptic",
            Orbit::new(
                0.8136083012245382,
                2.8944806908277103,
                0.6401863568023209,
                3.0362713144982374,
                1.918498022946511,
                5.8051565948396,
                1.0,
            ),
        ),
        (
            "Parabolic",
            Orbit::new(
                1.0,
                6.209509865315525,
                5.118096184019639,
                3.981150762118136,
                3.3940481449565048,
                3.736718306390939,
                1.0,
            ),
        ),
        (
            "Hyperbolic",
            Orbit::new(
                2.826628243687278,
                0.3257767961832889,
                5.182279515397755,
                6.212669269522696,
                1.990603413825992,
                6.145132647429473,
                1.0,
            ),
        ),
    ];

    for (kind, orbit) in orbits {
        for i in 0..ORBIT_POLL_ANGLES {
            let angle = (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64);

            let pos = orbit.get_position_at_angle(angle);
            let flat_pos = orbit.get_flat_position_at_angle(angle);
            let altitude = orbit.get_altitude_at_angle(angle);

            let pos_alt = (pos.0.powi(2) + pos.1.powi(2) + pos.2.powi(2)).sqrt();

            let pos_flat = (flat_pos.0.powi(2) + flat_pos.1.powi(2)).sqrt();

            if !altitude.is_finite() {
                continue;
            }

            assert_almost_eq(
                pos_alt,
                altitude,
                &format!("Dist of tilted point (orbit {kind}, angle {angle})",),
            );

            assert_almost_eq(
                pos_flat,
                altitude,
                &format!("Dist of flat point (orbit {kind}, angle {angle})",),
            );
        }
    }
}

#[test]
fn test_velocity() {
    let orbits = [
        (
            "Circular",
            Orbit::new(
                0.0,
                6.1378562542109725,
                1.4755776307550952,
                0.3267764431411045,
                0.6031097400880272,
                0.7494917839241119,
                1.0,
            ),
        ),
        (
            "Elliptic",
            Orbit::new(
                0.8136083012245382,
                2.8944806908277103,
                0.6401863568023209,
                3.0362713144982374,
                1.918498022946511,
                5.8051565948396,
                1.0,
            ),
        ),
        (
            "Hyperbolic",
            Orbit::new(
                2.826628243687278,
                0.3257767961832889,
                5.182279515397755,
                6.212669269522696,
                1.990603413825992,
                6.145132647429473,
                1.0,
            ),
        ),
    ];

    for (what, orbit) in orbits {
        speed_velocity_base_test(&orbit, what);
        naive_speed_correlation_base_test(&orbit, what);
    }

    // TODO: POST-PARABOLA SUPPORT: Change to all-random instead of just nonparabolic
    for mut orbit in random_nonparabolic_iter(128) {
        orbit.set_gravitational_parameter(100.0 / orbit.get_semi_major_axis().abs(), crate::MuSetterMode::KeepElements);
        let what = &format!("random ({orbit:?})");
        speed_velocity_base_test(&orbit, what);
        // We purposely leave out naive speed correlation because some fuzzed orbits
        // are just too extreme for the naive method to be accurate
    }
}

mod monotone_cubic_solver {
    use crate::solve_monotone_cubic;

    #[test]
    fn test_monotone_increasing() {
        // x^3 + 3x^2 + 3x + 1 = 0
        // Monotonic increasing, one real root
        let root = solve_monotone_cubic(1.0, 3.0, 3.0, 1.0);
        let expected = -1.0; // Root is exactly -1
        assert!(
            (root - expected).abs() < 1e-6,
            "Expected root close to -1, got {root}"
        );
    }

    #[test]
    fn test_edge_case_triple_root() {
        // x^3 = 0
        // Real root: x = 0
        let root = solve_monotone_cubic(1.0, 0.0, 0.0, 0.0);
        assert!(
            (root - 0.0).abs() < 1e-6,
            "Expected root close to 0, got {root}"
        );
    }

    #[test]
    fn test_large_coefficients() {
        // 1000x^3 - 3000x^2 + 3000x - 1000 = 0
        // Should handle large coefficients accurately
        let root = solve_monotone_cubic(1000.0, -3000.0, 3000.0, -1000.0);
        let expected = 1.0; // Root is exactly 1
        assert!(
            (root - expected).abs() < 1e-6,
            "Expected root close to 1, got {root}"
        );
    }
}

mod sinh_approx {
    use crate::generated_sinh_approximator::sinh_approx_lt5;
    use std::hint::black_box;
    use std::time::Instant;

    // Run this benchmark:
    // `RUST_TEST_NOCAPTURE=1 cargo test -r bench_sinh` (bash)
    #[test]
    fn bench_sinh() {
        let base_start_time = Instant::now();
        for i in 0..5000000 {
            let f = i as f64 * 1e-6;
            black_box(black_box(f).sinh());
        }
        let base_duration = base_start_time.elapsed();

        let approx_start_time = Instant::now();
        for i in 0..5000000 {
            let f = i as f64 * 1e-6;
            black_box(sinh_approx_lt5(black_box(f)));
        }
        let approx_duration = approx_start_time.elapsed();

        eprintln!("Real sinh call: {:?}", base_duration);
        eprintln!("Approximation: {:?}", approx_duration);
        let speed_factor = base_duration.as_nanos() as f64 / approx_duration.as_nanos() as f64;
        eprintln!("\nThe approximation is {speed_factor}x the speed of the real sinh call.",);
        assert!(
            speed_factor > 1.0,
            "The approximation is slower than the real sinh call."
        );
    }
}
