use core::f64::consts::{PI, TAU};

use glam::{DVec2, DVec3};
use std::string::ToString;

use crate::{CompactOrbit, OrbitTrait};

const ALMOST_EQ_TOLERANCE: f64 = 1e-6;
pub(super) fn assert_almost_eq(a: f64, b: f64, what: &str) {
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

pub(super) fn assert_almost_eq_orbit(a: &impl OrbitTrait, b: &impl OrbitTrait, what: &str) {
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

pub(super) fn assert_eq_vec3(a: DVec3, b: DVec3, what: &str) {
    let desc = format!("{a:?} vs {b:?}; {what}");
    assert_eq!(a.x.to_bits(), b.x.to_bits(), "X coord of {desc}");
    assert_eq!(a.y.to_bits(), b.y.to_bits(), "Y coord of {desc}");
    assert_eq!(a.z.to_bits(), b.z.to_bits(), "Z coord of {desc}");
}

pub(super) fn assert_eq_vec2(a: DVec2, b: DVec2, what: &str) {
    let desc = format!("{a:?} vs {b:?}; {what}");
    assert_eq!(a.x.to_bits(), b.x.to_bits(), "X coord of {desc}");
    assert_eq!(a.y.to_bits(), b.y.to_bits(), "Y coord of {desc}");
}

pub(super) fn assert_almost_eq_vec3(a: DVec3, b: DVec3, what: &str) {
    let desc = format!("{a:?} vs {b:?}; {what}");
    assert_almost_eq(a.x, b.x, &format!("X coord of {desc}"));
    assert_almost_eq(a.y, b.y, &format!("Y coord of {desc}"));
    assert_almost_eq(a.z, b.z, &format!("Z coord of {desc}"));
}

pub(super) fn assert_almost_eq_rescale(a: f64, b: f64, what: &str) {
    assert_eq!(
        a.signum(),
        b.signum(),
        "sign of given params not the same: {what}"
    );

    let a_scale = a.abs().log2();
    let b_scale = b.abs().log2();

    assert_almost_eq(a_scale, b_scale, &format!("logarithmic scale of {what}"));
}

pub(super) fn assert_almost_eq_vec3_rescale(a: DVec3, b: DVec3, what: &str) {
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

pub(super) fn assert_almost_eq_vec2(a: DVec2, b: DVec2, what: &str) {
    let desc = format!("{a:?} vs {b:?}; {what}");
    assert_almost_eq(a.x, b.x, &("X coord of ".to_string() + &desc));
    assert_almost_eq(a.y, b.y, &("Y coord of ".to_string() + &desc));
}

pub(super) fn assert_orbit_positions_3d(orbit: &impl OrbitTrait, tests: &[(&str, f64, DVec3)]) {
    for (what, angle, expected) in tests.iter() {
        let pos = orbit.get_position_at_true_anomaly(*angle);
        assert_almost_eq_vec3(pos, *expected, what);
    }
}

pub(super) fn assert_orbit_positions_2d(orbit: &impl OrbitTrait, tests: &[(&str, f64, DVec2)]) {
    for (what, angle, expected) in tests.iter() {
        let pos = orbit.get_pqw_position_at_true_anomaly(*angle);
        assert_almost_eq_vec2(pos, *expected, what);
    }
}

pub(super) fn assert_eq_orbit(a: &impl OrbitTrait, b: &impl OrbitTrait, what: &str) {
    fn to_compact(orbit: &impl OrbitTrait) -> CompactOrbit {
        let eccentricity = orbit.get_eccentricity();
        let periapsis = orbit.get_periapsis();
        let inclination = orbit.get_inclination();
        let arg_pe = orbit.get_arg_pe();
        let long_asc_node = orbit.get_long_asc_node();
        let mean_anomaly = orbit.get_mean_anomaly_at_epoch();
        let mu = orbit.get_gravitational_parameter();

        CompactOrbit {
            eccentricity,
            periapsis,
            inclination,
            arg_pe,
            long_asc_node,
            mean_anomaly,
            mu,
        }
    }

    let a = to_compact(a);
    let b = to_compact(b);

    assert_eq!(a, b, "assertion failed: orbits are not equal: {what}");
}
