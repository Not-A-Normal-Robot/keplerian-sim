#![cfg(test)]

extern crate std;

use glam::{DVec2, DVec3};

use crate::{CompactOrbit, Orbit, OrbitTrait, StateVectors};
use std::f64::consts::{PI, TAU};

const ALMOST_EQ_TOLERANCE: f64 = 1e-6;
const ORBIT_POLL_ANGLES: usize = 4096;

fn dvec3_to_bits(v: DVec3) -> (u64, u64, u64) {
    (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
}

fn assert_almost_eq(a: f64, b: f64, what: &str) {
    if a.is_nan() && b.is_nan() {
        return;
    }

    let dist = (a - b).abs();
    let msg = format!(
        "Almost-eq assertion failed for '{what}'!\n\
        {a} and {b} has distance {dist}, which is more than max of {ALMOST_EQ_TOLERANCE}"
    );

    assert!(dist < ALMOST_EQ_TOLERANCE, "{msg}");
}

fn assert_almost_eq_orbit(a: &impl OrbitTrait, b: &impl OrbitTrait, what: &str) {
    assert_almost_eq(
        a.get_gravitational_parameter(),
        b.get_gravitational_parameter(),
        &format!("gravitational parameter of {what}"),
    );
    assert_almost_eq(
        a.get_eccentricity(),
        b.get_eccentricity(),
        &format!("eccentricity of {what}"),
    );
    assert_almost_eq(
        a.get_periapsis(),
        b.get_periapsis(),
        &format!("periapsis of {what}"),
    );

    const TIMES: [f64; 3] = [0.0, -1.0, 1.0];

    for t in TIMES {
        let mean_anom_a = a.get_mean_anomaly_at_time(t);
        let ecc_anom_a = a.get_eccentric_anomaly_at_time(t);
        let true_anom_a = a.get_true_anomaly_at_time(t);
        let mean_anom_b = b.get_mean_anomaly_at_time(t);
        let ecc_anom_b = b.get_eccentric_anomaly_at_time(t);
        let true_anom_b = b.get_true_anomaly_at_time(t);
        let a_sv = a.get_state_vectors_at_time(t);
        let b_sv = b.get_state_vectors_at_time(t);

        assert_almost_eq_vec3(
            a_sv.position.normalize(),
            b_sv.position.normalize(),
            &format!("Normalized positions ({} vs {}) at t = {t} (Ma={mean_anom_a:?}/Mb={mean_anom_b:?}/Ea={ecc_anom_a:?}/Eb={ecc_anom_b:?}/fa={true_anom_a:?}/fb={true_anom_b:?}) for {what}",
                a_sv.position, b_sv.position),
        );
        assert_almost_eq(
            a_sv.position.length().log10(),
            b_sv.position.length().log10(),
            &format!("Log of position ({} vs {}) magnitudes at t = {t} (Ma={mean_anom_a:?}/Mb={mean_anom_b:?}/Ea={ecc_anom_a:?}/Eb={ecc_anom_b:?}/fa={true_anom_a:?}/fb={true_anom_b:?}) for {what}",
                a_sv.position, b_sv.position),
        );
        assert_almost_eq_vec3(
            a_sv.velocity.normalize(),
            b_sv.velocity.normalize(),
            &format!("Normalized velocities ({} vs {}) at t = {t} (Ma={mean_anom_a:?}/Mb={mean_anom_b:?}/Ea={ecc_anom_a:?}/Eb={ecc_anom_b:?}/fa={true_anom_a:?}/fb={true_anom_b:?}) for {what}",
                a_sv.velocity, b_sv.velocity),
        );
        assert_almost_eq(
            a_sv.velocity.length().log10(),
            b_sv.velocity.length().log10(),
            &format!("Log of velocity ({} vs {}) magnitudes at t = {t} (Ma={mean_anom_a:?}/Mb={mean_anom_b:?}/Ea={ecc_anom_a:?}/Eb={ecc_anom_b:?}/fa={true_anom_a:?}/fb={true_anom_b:?}) for {what}",
                a_sv.velocity, b_sv.velocity),
        );
    }

    // Only test true anomaly SVs for inclined, non-circular orbits
    // We cannot rely on true anomaly-based getters in non-inclined, non-circular orbits
    // as the argument of periapsis can be wild sometimes, and we have to rely on the
    // time-based getters
    if a.get_eccentricity() > ALMOST_EQ_TOLERANCE
        && a.get_eccentricity() < 1.5
        && a.get_inclination().rem_euclid(TAU).abs() > ALMOST_EQ_TOLERANCE
    {
        const TRUE_ANOMALIES: [f64; 3] = [0.0, PI, -PI];

        for theta in TRUE_ANOMALIES {
            let a_sv = a.get_state_vectors_at_true_anomaly(theta);
            let b_sv = b.get_state_vectors_at_true_anomaly(theta);

            assert_almost_eq_vec3(
                a_sv.position.normalize(),
                b_sv.position.normalize(),
                &format!("Positions at f = {theta} for {what}"),
            );
            assert_almost_eq_vec3(
                a_sv.velocity.normalize(),
                b_sv.velocity.normalize(),
                &format!("Velocities at f = {theta} for {what}"),
            );
        }
    }

    if a.get_eccentricity() > 0.25 {
        let a_p = a.transform_pqw_vector(DVec2::new(1.0, 0.0));
        let a_q = a.transform_pqw_vector(DVec2::new(0.0, 1.0));
        let b_p = b.transform_pqw_vector(DVec2::new(1.0, 0.0));
        let b_q = b.transform_pqw_vector(DVec2::new(0.0, 1.0));

        let p_angle_diff = a_p.angle_between(b_p);
        let q_angle_diff = a_q.angle_between(b_q);

        const ANGULAR_TOLERANCE: f64 = 1e-5;

        assert!(
            p_angle_diff < ANGULAR_TOLERANCE,
            "P basis vector differs by {p_angle_diff} rad (> tolerance of {ANGULAR_TOLERANCE} rad) for {what}"
        );
        assert!(
            q_angle_diff < ANGULAR_TOLERANCE,
            "Q basis vector differs by {q_angle_diff} rad (> tolerance of {ANGULAR_TOLERANCE} rad) for {what}"
        );
    }
}

fn assert_eq_vec3(a: DVec3, b: DVec3, what: &str) {
    let desc = format!("{a:?} vs {b:?}; {what}");
    assert_eq!(a.x.to_bits(), b.x.to_bits(), "X coord of {desc}");
    assert_eq!(a.y.to_bits(), b.y.to_bits(), "Y coord of {desc}");
    assert_eq!(a.z.to_bits(), b.z.to_bits(), "Z coord of {desc}");
}

fn assert_eq_vec2(a: DVec2, b: DVec2, what: &str) {
    let desc = format!("{a:?} vs {b:?}; {what}");
    assert_eq!(a.x.to_bits(), b.x.to_bits(), "X coord of {desc}");
    assert_eq!(a.y.to_bits(), b.y.to_bits(), "Y coord of {desc}");
}

fn assert_almost_eq_vec3(a: DVec3, b: DVec3, what: &str) {
    let desc = format!("{a:?} vs {b:?}; {what}");
    assert_almost_eq(a.x, b.x, &format!("X coord of {desc}"));
    assert_almost_eq(a.y, b.y, &format!("Y coord of {desc}"));
    assert_almost_eq(a.z, b.z, &format!("Z coord of {desc}"));
}

fn assert_almost_eq_rescale(a: f64, b: f64, what: &str) {
    assert_eq!(
        a.signum(),
        b.signum(),
        "sign of given params not the same: {what}"
    );

    let a_scale = a.abs().log2();
    let b_scale = b.abs().log2();

    assert_almost_eq(a_scale, b_scale, &format!("logarithmic scale of {what}"));
}

fn assert_almost_eq_vec3_rescale(a: DVec3, b: DVec3, what: &str) {
    let desc = format!("{a:?} vs {b:?}; {what}");
    let a_norm = a.normalize();
    let b_norm = b.normalize();
    let a_scale = a.length().log2();
    let b_scale = b.length().log2();

    assert_almost_eq(a_scale, b_scale, &format!("logarithmic scale of {desc}"));
    if a_scale.is_finite() {
        assert_almost_eq(a_norm.x, b_norm.x, &format!("rescaled X coord of {desc}"));
        assert_almost_eq(a_norm.y, b_norm.y, &format!("rescaled Y coord of {desc}"));
        assert_almost_eq(a_norm.z, b_norm.z, &format!("rescaled Z coord of {desc}"));
    }
}

fn assert_almost_eq_vec2(a: DVec2, b: DVec2, what: &str) {
    let desc = format!("{a:?} vs {b:?}; {what}");
    assert_almost_eq(a.x, b.x, &("X coord of ".to_string() + &desc));
    assert_almost_eq(a.y, b.y, &("Y coord of ".to_string() + &desc));
}

fn assert_orbit_positions_3d(orbit: &impl OrbitTrait, tests: &[(&str, f64, DVec3)]) {
    for (what, angle, expected) in tests.iter() {
        let pos = orbit.get_position_at_true_anomaly(*angle);
        assert_almost_eq_vec3(pos, *expected, what);
    }
}

fn assert_orbit_positions_2d(orbit: &impl OrbitTrait, tests: &[(&str, f64, DVec2)]) {
    for (what, angle, expected) in tests.iter() {
        let pos = orbit.get_pqw_position_at_true_anomaly(*angle);
        assert_almost_eq_vec2(pos, *expected, what);
    }
}

fn poll_orbit(orbit: &impl OrbitTrait) -> Vec<DVec3> {
    let mut vec: Vec<DVec3> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let time = if orbit.get_eccentricity() < 1.0 {
            (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
        } else {
            (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
        };
        vec.push(orbit.get_position_at_time(time));
    }

    return vec;
}
fn poll_flat(orbit: &impl OrbitTrait) -> Vec<DVec2> {
    let mut vec: Vec<DVec2> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let angle = (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64);
        vec.push(orbit.get_pqw_position_at_true_anomaly(angle));
    }

    return vec;
}
fn poll_transform(orbit: &impl OrbitTrait) -> Vec<DVec3> {
    let mut vec: Vec<DVec3> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let angle = (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64);
        vec.push(orbit.transform_pqw_vector(DVec2::new(1.0 * angle.cos(), 1.0 * angle.sin())));
    }

    return vec;
}
fn poll_eccentric_anomaly(orbit: &impl OrbitTrait) -> Vec<f64> {
    let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let time = if orbit.get_eccentricity() < 1.0 {
            (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
        } else {
            (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
        };

        vec.push(orbit.get_eccentric_anomaly_at_time(time));
    }

    return vec;
}
fn poll_speed(orbit: &impl OrbitTrait) -> Vec<f64> {
    (0..ORBIT_POLL_ANGLES)
        .into_iter()
        .map(|i| {
            if orbit.get_eccentricity() < 1.0 {
                (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
            } else {
                (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
            }
        })
        .map(|t| orbit.get_speed_at_time(t))
        .collect()
}
fn poll_flat_vel(orbit: &impl OrbitTrait) -> Vec<DVec2> {
    (0..ORBIT_POLL_ANGLES)
        .into_iter()
        .map(|i| {
            if orbit.get_eccentricity() < 1.0 {
                (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
            } else {
                (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
            }
        })
        .map(|t| orbit.get_pqw_velocity_at_time(t))
        .collect()
}
fn poll_vel(orbit: &impl OrbitTrait) -> Vec<DVec3> {
    (0..ORBIT_POLL_ANGLES)
        .into_iter()
        .map(|i| {
            if orbit.get_eccentricity() < 1.0 {
                (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
            } else {
                (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
            }
        })
        .map(|t| orbit.get_velocity_at_time(t))
        .collect()
}
fn poll_sv(orbit: &impl OrbitTrait) -> Vec<StateVectors> {
    (0..ORBIT_POLL_ANGLES)
        .into_iter()
        .map(|i| {
            if orbit.get_eccentricity() < 1.0 {
                (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
            } else {
                (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
            }
        })
        .map(|t| orbit.get_state_vectors_at_time(t))
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

    pub(super) fn random_elliptic() -> Orbit {
        Orbit::new(
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

    pub(super) fn random_near_parabolic() -> Orbit {
        Orbit::new(
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

    pub(super) fn random_parabolic() -> Orbit {
        Orbit::new(
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

    pub(super) fn random_hyperbolic() -> Orbit {
        Orbit::new(
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

    pub(super) fn random_very_hyperbolic() -> Orbit {
        Orbit::new(
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

    pub(super) fn random_extremely_hyperbolic() -> Orbit {
        Orbit::new(
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
            ("unit orbit 1", 0.0 * PI, DVec3::new(1.0, 0.0, 0.0)),
            ("unit orbit 2", 0.5 * PI, DVec3::new(0.0, 1.0, 0.0)),
            ("unit orbit 3", 1.0 * PI, DVec3::new(-1.0, 0.0, 0.0)),
            ("unit orbit 4", 1.5 * PI, DVec3::new(0.0, -1.0, 0.0)),
            ("unit orbit 5", 2.0 * PI, DVec3::new(1.0, 0.0, 0.0)),
        ],
    );
}

#[test]
fn unit_orbit_angle_2d() {
    let orbit = unit_orbit();

    assert_orbit_positions_2d(
        &orbit,
        &[
            ("unit orbit 1", 0.0 * PI, DVec2::new(1.0, 0.0)),
            ("unit orbit 2", 0.5 * PI, DVec2::new(0.0, 1.0)),
            ("unit orbit 3", 1.0 * PI, DVec2::new(-1.0, 0.0)),
            ("unit orbit 4", 1.5 * PI, DVec2::new(0.0, -1.0)),
            ("unit orbit 5", 2.0 * PI, DVec2::new(1.0, 0.0)),
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
        let transformed = orbit.transform_pqw_vector(DVec2::new(point.0, point.1));

        assert_eq!(transformed.x, point.0);
        assert_eq!(transformed.y, point.1);
        assert_eq!(transformed.z, 0.0);
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
        let distance = point.length();

        assert_almost_eq(distance, 1.0, "Distance");
    }
}

#[test]
fn tilted_90deg() {
    let orbit = Orbit::new(0.0, 1.0, PI / 2.0, 0.0, 0.0, 0.0, 1.0);

    // Transform test
    let tests = [
        // Before and after transformation
        (("Vector 1"), (1.0, 0.0), DVec3::new(1.0, 0.0, 0.0)),
        (("Vector 2"), (0.0, 1.0), DVec3::new(0.0, 0.0, 1.0)),
        (("Vector 3"), (-1.0, 0.0), DVec3::new(-1.0, 0.0, 0.0)),
        (("Vector 4"), (0.0, -1.0), DVec3::new(0.0, 0.0, -1.0)),
    ];

    for (what, point, expected) in tests.iter() {
        let transformed = orbit.transform_pqw_vector(DVec2::new(point.0, point.1));

        assert_almost_eq_vec3(transformed, *expected, what);
    }
}

#[test]
fn apoapsis_of_two() {
    let orbit = Orbit::with_apoapsis(2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let point_at_apoapsis = orbit.get_position_at_true_anomaly(PI);
    let point_at_periapsis = orbit.get_position_at_true_anomaly(0.0);

    assert_almost_eq_vec3(point_at_apoapsis, DVec3::new(-2.0, 0.0, 0.0), "Ap");
    assert_almost_eq_vec3(point_at_periapsis, DVec3::new(1.0, 0.0, 0.0), "Pe");
}

#[test]
fn huge_apoapsis() {
    let orbit = Orbit::with_apoapsis(10000.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let point_at_apoapsis = orbit.get_position_at_true_anomaly(PI);
    let point_at_periapsis = orbit.get_position_at_true_anomaly(0.0);

    assert_almost_eq_vec3(point_at_apoapsis, DVec3::new(-10000.0, 0.0, 0.0), "Ap");
    assert_almost_eq_vec3(point_at_periapsis, DVec3::new(1.0, 0.0, 0.0), "Pe");
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
            pos.x.is_finite() && pos.y.is_finite(),
            "2D position algorithm instability at near-parabolic edge case"
        );
    }

    let positions = poll_orbit(&orbit);

    for pos in positions {
        assert!(
            pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
            "3D position algorithm instability at near-parabolic edge case"
        );
    }

    let position_at_periapsis = orbit.get_position_at_true_anomaly(TAU);

    assert_almost_eq_vec3(
        position_at_periapsis,
        DVec3::new(1.0, 0.0, 0.0),
        "Periapsis",
    );

    let position_at_apoapsis = orbit.get_position_at_true_anomaly(PI);

    assert!(
        position_at_apoapsis.x.abs() > 1e12,
        "Apoapsis is not far enough"
    );
}

#[test]
fn parabolic() {
    let orbit = Orbit::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let point_near_infinity = orbit.get_position_at_true_anomaly(PI - 1e-7);
    let point_at_periapsis = orbit.get_position_at_true_anomaly(0.0);

    assert!(
        point_near_infinity.length() > 1e9,
        "Point near infinity is not far enough"
    );
    assert!(
        point_near_infinity.y.abs() > 0.0,
        "Y coord near infinity should move a little"
    );
    assert_almost_eq(
        point_near_infinity.z,
        0.0,
        "Point near infinity should be flat",
    );
    assert_almost_eq_vec3(point_at_periapsis, DVec3::new(1.0, 0.0, 0.0), "Pe");

    let point_at_asymptote = orbit.get_position_at_true_anomaly(PI);

    assert!(
        point_at_asymptote.x.is_nan(),
        "X at asymptote should be undefined"
    );
    assert!(
        point_at_asymptote.y.is_nan(),
        "Y at asymptote should be undefined"
    );
    assert!(
        point_at_asymptote.z.is_nan(),
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
                original_ecc[i].to_bits(),
                compact_ecc[i].to_bits(),
                "{compact_message} (eccentric anomaly)"
            );
            assert_eq!(
                original_ecc[i].to_bits(),
                reexpanded_ecc[i].to_bits(),
                "{reexpanded_message} (eccentric anomaly)"
            );
        }
    }
    {
        let original_true = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let true_anomaly = orbit.get_true_anomaly_at_mean_anomaly(angle);
                vec.push(true_anomaly);
            }

            vec
        };
        let compact_true = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let true_anomaly = compact_orbit.get_true_anomaly_at_mean_anomaly(angle);
                vec.push(true_anomaly);
            }

            vec
        };
        let reexpanded_true = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                vec.push(reexpanded_orbit.get_true_anomaly_at_mean_anomaly(angle));
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
                let altitude = orbit.get_altitude_at_true_anomaly(angle);
                vec.push(altitude);
            }

            vec
        };
        let compact_altitudes = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let altitude = compact_orbit.get_altitude_at_true_anomaly(angle);
                vec.push(altitude);
            }

            vec
        };
        let reexpanded_altitudes = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let altitude = reexpanded_orbit.get_altitude_at_true_anomaly(angle);
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
                .map(|DVec2 { x, y }| (x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>(),
            compact_fvels
                .iter()
                .map(|DVec2 { x, y }| (x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>(),
            "{compact_message}",
        );
        assert_eq!(
            original_fvels
                .iter()
                .map(|DVec2 { x, y }| (x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>(),
            reexpanded_fvels
                .iter()
                .map(|DVec2 { x, y }| (x.to_bits(), y.to_bits()))
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
                .map(|DVec3 { x, y, z }| (x.to_bits(), y.to_bits(), z.to_bits()))
                .collect::<Vec<_>>(),
            compact_vels
                .iter()
                .map(|DVec3 { x, y, z }| (x.to_bits(), y.to_bits(), z.to_bits()))
                .collect::<Vec<_>>(),
            "{compact_message}",
        );
        assert_eq!(
            original_vels
                .iter()
                .map(|DVec3 { x, y, z }| (x.to_bits(), y.to_bits(), z.to_bits()))
                .collect::<Vec<_>>(),
            reexpanded_vels
                .iter()
                .map(|DVec3 { x, y, z }| (x.to_bits(), y.to_bits(), z.to_bits()))
                .collect::<Vec<_>>(),
            "{reexpanded_message}",
        );
    }
    {
        let original_svs = poll_sv(&orbit);
        let compact_svs = poll_sv(&compact_orbit);
        let reexpanded_svs = poll_sv(&reexpanded_orbit);

        let compact_message = format!("{compact_message} (velocity)");
        let reexpanded_message = format!("{reexpanded_message} (velocity)");

        assert_eq!(
            original_svs
                .iter()
                .map(|s| (
                    s.position.x.to_bits(),
                    s.position.y.to_bits(),
                    s.position.z.to_bits(),
                    s.velocity.x.to_bits(),
                    s.velocity.y.to_bits(),
                    s.velocity.z.to_bits(),
                ))
                .collect::<Vec<_>>(),
            compact_svs
                .iter()
                .map(|s| (
                    s.position.x.to_bits(),
                    s.position.y.to_bits(),
                    s.position.z.to_bits(),
                    s.velocity.x.to_bits(),
                    s.velocity.y.to_bits(),
                    s.velocity.z.to_bits(),
                ))
                .collect::<Vec<_>>(),
            "{compact_message}",
        );
        assert_eq!(
            original_svs
                .iter()
                .map(|s| (
                    s.position.x.to_bits(),
                    s.position.y.to_bits(),
                    s.position.z.to_bits(),
                    s.velocity.x.to_bits(),
                    s.velocity.y.to_bits(),
                    s.velocity.z.to_bits(),
                ))
                .collect::<Vec<_>>(),
            reexpanded_svs
                .iter()
                .map(|s| (
                    s.position.x.to_bits(),
                    s.position.y.to_bits(),
                    s.position.z.to_bits(),
                    s.velocity.x.to_bits(),
                    s.velocity.y.to_bits(),
                    s.velocity.z.to_bits(),
                ))
                .collect::<Vec<_>>(),
            "{reexpanded_message}",
        );
    }
}

fn speed_velocity_base_test(orbit: &impl OrbitTrait, what: &str) {
    for i in 0..ORBIT_POLL_ANGLES {
        let t = (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64);

        let speed = orbit.get_speed_at_time(t);
        let flat_vel = orbit.get_pqw_velocity_at_time(t);
        let vel = orbit.get_velocity_at_time(t);

        let flat_vel_mag = flat_vel.length();
        let vel_mag = vel.length();

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

    fn cosine_similarity(v1: DVec3, v2: DVec3) -> f64 {
        let dot_product = v1.dot(v2);
        let v1_mag = v1.length();
        let v2_mag = v2.length();

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
        let diff = pos2 - pos1;

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

        let diff_mag = diff.length();
        let vel_mag = vel.length();
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

mod mu_setter {
    use super::*;

    const NEAR_PARABOLIC_RANGE: f64 = 5e-3;

    fn keep_elements_base_test(orbit: &(impl OrbitTrait + Clone)) {
        for _ in 0..1024 {
            let old_mu = orbit.get_gravitational_parameter();
            let new_mu = orbit.get_gravitational_parameter() * random_mult();
            let mut new_orbit = orbit.clone();
            new_orbit.set_gravitational_parameter(new_mu, crate::MuSetterMode::KeepElements);

            assert_eq!(
                orbit.get_eccentricity(),
                new_orbit.get_eccentricity(),
                "Eccentricity between mu={old_mu} to mu={new_mu}"
            );
            assert_eq!(
                orbit.get_periapsis(),
                new_orbit.get_periapsis(),
                "Periapsis between mu={old_mu} to mu={new_mu}"
            );
            assert_eq!(
                orbit.get_apoapsis(),
                new_orbit.get_apoapsis(),
                "Apoapsis between mu={old_mu} to mu={new_mu}"
            );
            assert_eq!(
                orbit.get_semi_major_axis(),
                new_orbit.get_semi_major_axis(),
                "Semi-major axis between mu={old_mu} to mu={new_mu}"
            );
            assert_eq!(
                orbit.get_inclination(),
                new_orbit.get_inclination(),
                "Inclination between mu={old_mu} to mu={new_mu}"
            );
            assert_eq!(
                orbit.get_arg_pe(),
                new_orbit.get_arg_pe(),
                "Argument of Periapsis between mu={old_mu} to mu={new_mu}"
            );
            assert_eq!(
                orbit.get_long_asc_node(),
                new_orbit.get_long_asc_node(),
                "Longitude of Ascending Node between mu={old_mu} to mu={new_mu}"
            );
            assert_eq!(
                orbit.get_linear_eccentricity(),
                new_orbit.get_linear_eccentricity(),
                "Linear Eccentricity between mu={old_mu} to mu={new_mu}"
            );
            assert_eq!(
                orbit.get_semi_minor_axis(),
                new_orbit.get_semi_minor_axis(),
                "Semi-minor axis between mu={old_mu} to mu={new_mu}"
            );
            assert_eq!(
                orbit.get_semi_latus_rectum(),
                new_orbit.get_semi_latus_rectum(),
                "Semi-latus rectum between mu={old_mu} to mu={new_mu}"
            );
            assert_eq!(
                orbit.get_mean_anomaly_at_epoch(),
                new_orbit.get_mean_anomaly_at_epoch(),
                "Mean anomaly at epoch between mu={old_mu} to mu={new_mu}"
            );
        }
    }

    use std::fmt::Debug;

    fn keep_position_time_base_test(orbit: &(impl OrbitTrait + Clone + Debug)) {
        for i in 0..1024 {
            let time = i as f64 * 0.15f64;
            let mut new_orbit = orbit.clone();
            let pos_before = orbit.get_position_at_time(time);
            new_orbit.set_gravitational_parameter(
                orbit.get_gravitational_parameter() * random_mult(),
                crate::MuSetterMode::KeepPositionAtTime(time),
            );
            let pos_after = new_orbit.get_position_at_time(time);

            let ext_info = format!(
                "with orbits {orbit:?} vs {new_orbit:?}, on iteration {i}, at time {time:?} \
                (KeepPositionAtTime)"
            );

            assert_eq!(
                orbit.get_eccentricity(),
                new_orbit.get_eccentricity(),
                "Eccentricity before and after mu setter, {ext_info}"
            );
            assert_eq!(
                orbit.get_periapsis(),
                new_orbit.get_periapsis(),
                "Periapsis before and after mu setter, {ext_info}"
            );
            assert_eq!(
                orbit.get_apoapsis(),
                new_orbit.get_apoapsis(),
                "Apoapsis before and after mu setter, {ext_info}"
            );
            assert_eq!(
                orbit.get_semi_major_axis(),
                new_orbit.get_semi_major_axis(),
                "Semi-major axis before and after mu setter, {ext_info}"
            );
            assert_eq!(
                orbit.get_inclination(),
                new_orbit.get_inclination(),
                "Inclination before and after mu setter, {ext_info}"
            );
            assert_eq!(
                orbit.get_arg_pe(),
                new_orbit.get_arg_pe(),
                "Argument of Periapsis before and after mu setter, {ext_info}"
            );
            assert_eq!(
                orbit.get_long_asc_node(),
                new_orbit.get_long_asc_node(),
                "Longitude of Ascending Node before and after mu setter, {ext_info}"
            );
            assert_eq!(
                orbit.get_linear_eccentricity(),
                new_orbit.get_linear_eccentricity(),
                "Linear Eccentricity before and after mu setter, {ext_info}"
            );
            assert_eq!(
                orbit.get_semi_minor_axis(),
                new_orbit.get_semi_minor_axis(),
                "Semi-minor axis before and after mu setter, {ext_info}"
            );
            assert_eq!(
                orbit.get_semi_latus_rectum(),
                new_orbit.get_semi_latus_rectum(),
                "Semi-latus rectum before and after mu setter, {ext_info}"
            );
            assert_almost_eq_vec3_rescale(
                pos_before,
                pos_after,
                &format!("Positions before and after mu setter, {ext_info}"),
            );
        }
    }

    fn keep_sv_time_base_test(orbit: &(impl OrbitTrait + Clone + Debug)) {
        for i in 0..1024 {
            let time = i as f64 * 0.15f64;
            let mut new_orbit = orbit.clone();
            let sv_before = orbit.get_state_vectors_at_time(time);
            new_orbit.set_gravitational_parameter(
                orbit.get_gravitational_parameter() * random_mult(),
                crate::MuSetterMode::KeepStateVectorsAtTime(time),
            );
            let sv_after = new_orbit.get_state_vectors_at_time(time);
            let ext_info = format!(
                "with orbits {orbit:?} vs {new_orbit:?}, on iteration {i}, at time={time:?} \
                (KeepStateVectorsAtTime)"
            );

            // TODO: PARABOLIC SUPPORT: This does not test for
            // parabolic orbits.
            if (new_orbit.get_eccentricity() - 1.0).abs() < NEAR_PARABOLIC_RANGE {
                // Currently, numerical instabilities arise near e = 1.
                // Although the tested function does work "fine" near it,
                // it deviates enough from the 1e-6 allowed delta-log to
                // fail the test. We skip checking it in that case.
                continue;
            }

            assert_almost_eq_vec3_rescale(
                sv_before.position,
                sv_after.position,
                &format!("Positions before and after mu setter, {ext_info}"),
            );
            assert_almost_eq_vec3_rescale(
                sv_before.velocity,
                sv_after.velocity,
                &format!("Velocities before and after mu setter, {ext_info}"),
            )
        }
    }

    fn keep_sv_known_base_test(orbit: &(impl OrbitTrait + Clone + Debug)) {
        for i in 0..1024 {
            let time = i as f64 * 0.15f64;
            let sv_before = orbit.get_state_vectors_at_time(time);
            let mut new_orbit = orbit.clone();
            new_orbit.set_gravitational_parameter(
                orbit.get_gravitational_parameter() * random_mult(),
                crate::MuSetterMode::KeepKnownStateVectors {
                    state_vectors: sv_before,
                    time,
                },
            );
            let sv_after = new_orbit.get_state_vectors_at_time(time);
            let ext_info = format!(
                "with orbits {orbit:?} vs {new_orbit:?}, on iteration {i}, at time {time:?} \
                (KeepKnownStateVectors)"
            );

            // TODO: PARABOLIC SUPPORT: This does not test for
            // parabolic orbits.
            if (new_orbit.get_eccentricity() - 1.0).abs() < NEAR_PARABOLIC_RANGE {
                // Currently, numerical instabilities arise near e = 1.
                // Although the tested function does work "fine" near it,
                // it deviates enough from the 1e-6 allowed delta-log to
                // fail the test. We skip checking it in that case.
                continue;
            }

            assert_almost_eq_vec3_rescale(
                sv_before.position,
                sv_after.position,
                &format!("Positions before and after mu setter, {ext_info}"),
            );
            assert_almost_eq_vec3_rescale(
                sv_before.velocity,
                sv_after.velocity,
                &format!("Velocities before and after mu setter, {ext_info}"),
            )
        }
    }

    fn base_test(orbit: &(impl OrbitTrait + Clone + Debug)) {
        keep_elements_base_test(orbit);
        keep_position_time_base_test(orbit);
        keep_sv_time_base_test(orbit);
        keep_sv_known_base_test(orbit);
    }

    #[test]
    fn test_mu_setter() {
        let known_problematic = [
            Orbit::new(
                0.0,
                220730.48307172305,
                0.0,
                -3.2972623354894797,
                -5.8373490691702985,
                -1.4594332879892935,
                818221.6447669249,
            ),
            Orbit::new(
                1.1132696061583278,
                418853.5613979898,
                5.7082319347995245,
                2.0258945240884216,
                -5.080428160893991,
                2.3404620071659235,
                902259.1940612693,
            ),
            Orbit::new(
                0.0,
                423238.6080417206,
                2.0181601877455844,
                5.275244425614675,
                -0.005493130779138156,
                1.3188040399571817,
                942689.3385315664,
            ),
        ];

        for orbit in known_problematic {
            base_test(&orbit);
        }

        // TODO: POST-PARABOLIC SUPPORT: Change to all-random instead of just nonparabolic
        let orbits = random_nonparabolic_iter(1024);

        for orbit in orbits {
            base_test(&orbit);
        }
    }
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

            let ecc_anom = orbit.get_eccentric_anomaly_at_mean_anomaly(angle);

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

            let pos = orbit.get_position_at_true_anomaly(angle);
            let flat_pos = orbit.get_pqw_position_at_true_anomaly(angle);
            let altitude = orbit.get_altitude_at_true_anomaly(angle).abs();

            let pos_alt = (pos.x.powi(2) + pos.y.powi(2) + pos.z.powi(2)).sqrt();

            let pos_flat = (flat_pos.x.powi(2) + flat_pos.y.powi(2)).sqrt();

            if !altitude.is_finite() {
                continue;
            }
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

    // TODO: POST-PARABOLIC SUPPORT: Change to all-random instead of just nonparabolic
    for mut orbit in random_nonparabolic_iter(128) {
        orbit.set_gravitational_parameter(
            100.0 / orbit.get_semi_major_axis().abs(),
            crate::MuSetterMode::KeepElements,
        );
        let what = &format!("random ({orbit:?})");
        speed_velocity_base_test(&orbit, what);
        // We purposely leave out naive speed correlation because some fuzzed orbits
        // are just too extreme for the naive method to be accurate
    }
}

fn state_vectors_getters_base_test(orbit: Orbit) {
    let p = poll_orbit(&orbit);
    let v = poll_vel(&orbit);

    let sv = poll_sv(&orbit);

    for i in 0..ORBIT_POLL_ANGLES {
        let p1 = p[i];
        let v1 = v[i];
        let StateVectors {
            position: p2,
            velocity: v2,
        } = sv[i];

        assert_eq!(
            dvec3_to_bits(p1),
            dvec3_to_bits(p2),
            "Positions of {orbit:?} at i={i}"
        );
        assert_eq!(
            dvec3_to_bits(v1),
            dvec3_to_bits(v2),
            "Velocities of {orbit:?} at i={i}"
        );
    }
}

#[test]
fn test_state_vectors_getters() {
    // TODO: POST-PARABOLIC SUPPORT: Change to all-random instead of just nonparabolic
    for mut orbit in random_nonparabolic_iter(128) {
        orbit.set_gravitational_parameter(
            rand::random_range(0.1..10.0),
            crate::MuSetterMode::KeepElements,
        );
        state_vectors_getters_base_test(orbit);
    }
}

#[test]
fn test_sv_to_orbit() {
    // TODO: POST-PARABOLIC SUPPORT: Change to all-random instead of just nonparabolic
    let known_problematic = [
        CompactOrbit::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        CompactOrbit::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.26, 1.0),
        CompactOrbit::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.42, 1.0),
        CompactOrbit::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.87, 1.0),
        CompactOrbit::new(0.1, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        CompactOrbit::new(0.1, 1.0, 0.0, 0.0, 0.0, 0.26, 1.0),
        CompactOrbit::new(0.1, 1.0, 0.0, 1.0, 0.0, 0.42, 1.0),
        CompactOrbit::new(0.1, 1.0, 0.0, 1.0, 0.0, 0.87, 1.0),
        CompactOrbit::new(
            0.0,
            248352.36201764457,
            -3.7693637740429713,
            -0.706898672541695,
            -4.8477018671546155,
            -1.280427245535563,
            2.3191422190564097,
        ),
        CompactOrbit::new(
            139.37169486186113,
            563133.6452925412,
            0.0,
            4.995613136230864,
            -0.6106457587079852,
            1.336834808877482,
            3.5820444996821963,
        ),
    ];

    for orbit in known_problematic {
        let iter = (0..ORBIT_POLL_ANGLES).into_iter().map(|i| {
            if orbit.get_eccentricity() < 1.0 {
                (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
            } else {
                (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
            }
        });

        for t in iter {
            let mean_anom = orbit.get_mean_anomaly_at_time(t);
            let ecc_anom = orbit.get_eccentric_anomaly_at_time(t);
            let true_anom = orbit.get_true_anomaly_at_time(t);
            let sv = orbit.get_state_vectors_at_time(t);
            let new_orbit = sv.to_compact_orbit(orbit.get_gravitational_parameter(), t);

            assert_almost_eq_orbit(
                &orbit,
                &new_orbit,
                &format!("[known problematic] (pre and post) {orbit:?} and {new_orbit:?} (derived at t={t:?}/M={mean_anom:?}/E={ecc_anom:?}/f={true_anom:?} from {sv:?})"),
            );
        }
    }

    for mut orbit in random_nonparabolic_iter(128) {
        orbit.set_gravitational_parameter(
            rand::random_range(0.1..10.0),
            crate::MuSetterMode::KeepElements,
        );
        let iter = (0..ORBIT_POLL_ANGLES).into_iter().map(|i| {
            if orbit.get_eccentricity() < 1.0 {
                (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
            } else {
                (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
            }
        });

        for t in iter {
            let mean_anom = orbit.get_mean_anomaly_at_time(t);
            let ecc_anom = orbit.get_eccentric_anomaly_at_time(t);
            let true_anom = orbit.get_true_anomaly_at_time(t);
            let sv = orbit.get_state_vectors_at_time(t);
            let new_orbit = sv.to_compact_orbit(orbit.get_gravitational_parameter(), t);

            assert_almost_eq_orbit(
                &orbit,
                &new_orbit,
                &format!(
                    "(pre and post) {orbit:?} and {new_orbit:?} (derived at t={t:?}/M={mean_anom:?}/E={ecc_anom:?}/f={true_anom:?} from {sv:?})"
                ),
            );
        }
    }
}

fn test_alt_to_true_anom_base(orbit: &(impl OrbitTrait + std::fmt::Debug)) {
    if orbit.get_eccentricity() == 0.0 {
        assert!(
            orbit
                .get_true_anomaly_at_altitude(orbit.get_periapsis() - 0.1)
                .is_nan(),
            "Circular orbits should result in NaN (-)"
        );
        assert!(
            orbit
                .get_true_anomaly_at_altitude(orbit.get_periapsis())
                .is_nan(),
            "Circular orbits should result in NaN (n)"
        );
        assert!(
            orbit
                .get_true_anomaly_at_altitude(orbit.get_periapsis() + 0.1)
                .is_nan(),
            "Circular orbits should result in NaN (+)"
        );
        return;
    }

    // CHECK_ITERATIONS may not be divisible by 4
    // as it will make f -> r -> f check
    // fail because it tries to get the apoapsis
    const CHECK_ITERATIONS: usize = 99;

    // Check r -> f -> r
    for i in 0..CHECK_ITERATIONS {
        let altitude = if orbit.get_eccentricity() < 1.0 {
            // We want CHECK_ITERATIONS midpoints, never any of the ends.
            let apoapsis = orbit.get_apoapsis();
            let diff = apoapsis - orbit.get_periapsis();

            let divisor = CHECK_ITERATIONS + 2;
            let index = i + 1;

            (diff * index as f64) / divisor as f64 + orbit.get_periapsis()
        } else {
            orbit.get_periapsis() * (2 + i) as f64
        };

        let true_anom = orbit.get_true_anomaly_at_altitude(altitude);

        assert!(
            true_anom.is_finite(),
            "r -> f -> r: True anomaly is {true_anom} (not finite). Orbit: {orbit:?}"
        );

        let new_altitude = orbit.get_altitude_at_true_anomaly(true_anom);

        assert_almost_eq_rescale(
            altitude,
            new_altitude,
            "Altitudes before vs after true anomaly conversion",
        );
    }

    // Check f -> r -> f
    let max_f = if orbit.get_eccentricity() < 1.0 {
        TAU
    } else {
        orbit.get_hyperbolic_true_anomaly_asymptote()
    };

    let min_f = -max_f;

    (1..=CHECK_ITERATIONS)
        .map(|x| x as f64 / (CHECK_ITERATIONS + 2) as f64)
        .map(|frac| frac * max_f + min_f)
        .for_each(|f| {
            let r = orbit.get_altitude_at_true_anomaly(f);
            let new_f = orbit.get_true_anomaly_at_altitude(r);

            // We compare their cosines because it's the simplest way to do
            // angle wrapping equality (e.g., 2π = 0 for true anomaly)
            assert_almost_eq(
                f.cos(),
                new_f.cos(),
                &format!("f -> r -> f ({f} -> {r} -> {new_f}) conversion for {orbit:?}"),
            );
        });

    // NaN checks
    if orbit.get_eccentricity() <= 1.0 {
        // Check if NaN appears when r < r_p, e ≤ 1
        assert!(orbit
            .get_true_anomaly_at_altitude(orbit.get_periapsis() * 0.5)
            .is_nan());
    } else {
        // Check if NaN appears when r_a < r < r_p, e > 1
        // Check if NaN doesn't appear when r < r_a, e > 1
        let periapsis = orbit.get_periapsis();
        let apoapsis = orbit.get_apoapsis();

        (1..=CHECK_ITERATIONS)
            .map(|x| x as f64 / (CHECK_ITERATIONS + 2) as f64)
            .map(|frac| frac * periapsis + apoapsis)
            .for_each(|altitude| {
                assert!(
                    orbit.get_true_anomaly_at_altitude(altitude).is_nan(),
                    "r_a < r < r_p, e > 1\n\
                    altitude should be out of range and result in NaN\n\
                    Orbit: {orbit:?}"
                )
            });
        (2..CHECK_ITERATIONS + 2)
            .map(|x| x as f64 * apoapsis)
            .for_each(|altitude| {
                assert!(
                    orbit.get_true_anomaly_at_altitude(altitude).is_finite(),
                    "r < r_a, e > 1\n\
                    result should be mathematically valid although outside valid M_h or H\n\
                    Orbit: {orbit:?}"
                )
            })
    }
}

#[test]
fn test_alt_to_true_anom() {
    let known_orbits = [
        Orbit::new(
            108.12478516555166,
            739446.3739243945,
            4.4379866572367686,
            -3.837745491000056,
            -2.723528963332692,
            -1.235055257543035,
            88693.72512103277,
        ),
        Orbit::new(
            0.9966834605870825,
            233530.8545930106,
            0.0,
            6.265961424477499,
            4.903832903204938,
            3.3032020837017892,
            594980.7007391909,
        ),
        Orbit::new(
            2.786789256572612,
            169320.70777622596,
            0.0,
            -1.7310518382457545,
            -4.49795466065101,
            2.1129080263390723,
            543463.4795382554,
        ),
        Orbit::new(
            148.19488171395574,
            884847.2512842646,
            0.0,
            -3.9145535884386926,
            3.3494081091913586,
            6.266067874410721,
            659705.008372745,
        ),
        Orbit::new(
            1.7105101322365424,
            329370.2648665676,
            -5.775473180092787,
            -1.7208954811340194,
            -1.6626463267393303,
            -5.874038755244536,
            802179.9169292466,
        ),
        Orbit::new(
            145.94844015040508,
            203908.025622657,
            -2.40655171307567,
            3.9435683505524715,
            -2.5384292539624265,
            -4.604755233153464,
            169792.97276552342,
        ),
        Orbit::new(
            1.1855842085394555,
            816102.3199830793,
            0.0,
            3.4437691346789023,
            -3.3613404930844695,
            -4.542697384938566,
            509523.27973998577,
        ),
        Orbit::new(
            2.961364164842913,
            621917.9015228765,
            0.0,
            4.285330127811376,
            2.276790533032832,
            1.9511505828892588,
            425581.3212116419,
        ),
        Orbit::new(
            1.0,
            638736.116675038,
            0.0,
            -6.2285141477259005,
            5.762865231402442,
            -3.484031308878359,
            878536.8370220523,
        ),
        Orbit::new(
            0.9929105972509886,
            128589.50974964743,
            0.0,
            -3.314959795311654,
            1.4208937183840566,
            -0.578931201049965,
            600281.5404856752,
        ),
        Orbit::new(
            0.0,
            654207.4308727328,
            0.0,
            5.927773945968786,
            0.9737752128280057,
            4.345813362095528,
            839697.9513802704,
        ),
        Orbit::new(
            13.58212992975093,
            815604.474807989,
            0.0,
            1.3140099764135718,
            2.65264265032366,
            -0.1227234741863299,
            940369.9821905283,
        ),
        Orbit::new(
            0.0,
            918529.8000047116,
            0.0,
            -4.029356132377893,
            4.12743273252093,
            -5.680671790639481,
            464424.01626417803,
        ),
        Orbit::new(
            2.5885349497922467,
            441863.5169381175,
            -1.3020833891335872,
            0.22730327862308286,
            2.1924698142340446,
            -1.3124860767349373,
            16513.878288684613,
        ),
        Orbit::new(
            8.422998841188784,
            943468.2020044628,
            0.0,
            5.864647753975204,
            -5.457490050870641,
            6.21356244211619,
            698090.9211077694,
        ),
    ];

    for orbit in known_orbits {
        if orbit.get_eccentricity() != 0.0 {
            assert_almost_eq(
                orbit.get_true_anomaly_at_altitude(orbit.get_periapsis() * 1.000000000000005),
                0.0,
                &format!(
                    "Orbit periapsis true anomaly should be zero\n\
                    Orbit: {orbit:?}"
                ),
            );
        }
        test_alt_to_true_anom_base(&orbit);
    }

    for orbit in random_any_iter(4096) {
        test_alt_to_true_anom_base(&orbit);
    }
}

fn z_an_dn_base_test(orbit: &(impl OrbitTrait + std::fmt::Debug)) {
    let f_an = orbit.get_true_anomaly_at_asc_node();
    let f_dn = orbit.get_true_anomaly_at_desc_node();

    assert!(
        ((f_an + PI).rem_euclid(TAU) - f_dn).abs() < 1e-15,
        "AN->DN equation should hold for {orbit:?}"
    );
    assert!(
        ((f_dn - PI).rem_euclid(TAU) - f_an).abs() < 1e-15,
        "DN->AN equation should hold for {orbit:?}"
    );

    // For open trajectories, f_an and f_dn may be out of range,
    // which results in NaN velocities. We check this before
    // checking if the vel directions make sense

    let f_range = if orbit.get_eccentricity() < 1.0 {
        -TAU..=TAU
    } else {
        let f_max = orbit.get_hyperbolic_true_anomaly_asymptote();
        (-f_max + 1e-14)..=(f_max - 1e-14)
    };

    if f_range.contains(&f_an) {
        let v_an = orbit.get_velocity_at_true_anomaly(f_an);

        if !v_an.is_nan() {
            assert!(
                v_an.z >= 0.0,
                "Z-vel {v_an} at AN (f = {f_an}) should be positive for {orbit:?}"
            );
        }
    }

    if f_range.contains(&f_dn) {
        let v_dn = orbit.get_velocity_at_true_anomaly(f_dn);

        if !v_dn.is_nan() {
            assert!(
                v_dn.z <= 0.0,
                "Z-vel {v_dn} at DN (f = {f_dn}) should be negative for {orbit:?}"
            );
        }
    }
}

#[test]
fn test_z_an_dn() {
    for orbit in random_any_iter(262144) {
        z_an_dn_base_test(&orbit);
    }
}

fn orbital_plane_normal_base_test(orbit: &impl OrbitTrait) {
    let p = orbit.transform_pqw_vector(DVec2::new(1.0, 0.0));
    let q = orbit.transform_pqw_vector(DVec2::new(0.0, 1.0));
    let w = p.cross(q);

    assert_eq!(orbit.get_orbital_plane_normal(), w);
}

#[test]
fn test_orbital_plane_normal_getter() {
    for orbit in random_any_iter(262144) {
        orbital_plane_normal_base_test(&orbit);
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

#[cfg(feature = "mint")]
mod mint_test {
    use crate::Matrix3x2;
    use mint::RowMatrix3x2;
    use rand::{rngs::ThreadRng, Rng};

    fn get_random_matrix(rng: &mut ThreadRng) -> Matrix3x2 {
        Matrix3x2 {
            e11: rng.random_range(-1e100..1e100),
            e12: rng.random_range(-1e100..1e100),
            e21: rng.random_range(-1e100..1e100),
            e22: rng.random_range(-1e100..1e100),
            e31: rng.random_range(-1e100..1e100),
            e32: rng.random_range(-1e100..1e100),
        }
    }

    #[test]
    fn test_mint_conversions() {
        let mut rng = rand::rng();
        for _ in 0..10000 {
            let my_mat = get_random_matrix(&mut rng);
            let mint_mat: RowMatrix3x2<f64> = my_mat.into();
            let also_mine: Matrix3x2 = mint_mat.into();

            assert_eq!(my_mat, also_mine);
        }
    }
}
