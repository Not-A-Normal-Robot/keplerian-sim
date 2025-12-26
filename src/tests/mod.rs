#![cfg(test)]

extern crate std;
use std::{eprintln, format, println, string::String, vec::Vec};

use glam::{DVec2, DVec3};

use crate::{
    CompactOrbit, CompactOrbit2D, Matrix3x2, MuSetterMode, Orbit, Orbit2D, OrbitTrait,
    OrbitTrait2D, StateVectors, StateVectors2D,
};
use std::{
    f64::consts::{PI, TAU},
    fmt::Debug,
};

const ORBIT_POLL_ANGLES: usize = 4096;

mod assertions;
mod polling;
mod seeders;

use assertions::*;
use polling::*;
use seeders::*;

fn dvec3_to_bits(v: DVec3) -> (u64, u64, u64) {
    (v.x.to_bits(), v.y.to_bits(), v.z.to_bits())
}
fn unit_orbit() -> Orbit {
    return Orbit::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
}

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

fn orbit2d_conversion_base_test(orbit: Orbit2D, what: &str) {
    let compact_orbit = CompactOrbit2D::from(orbit.clone());
    let reexpanded_orbit = Orbit2D::from(compact_orbit.clone());

    let compact_message = format!("Original / Compact ({what})");
    let reexpanded_message = format!("Compact / Reexpanded ({what})");

    {
        let original_vectors = orbit.get_pqw_basis_vectors();
        let compact_vectors = compact_orbit.get_pqw_basis_vectors();
        let reexpanded_vectors = reexpanded_orbit.get_pqw_basis_vectors();

        let original_separate = (
            orbit.get_pqw_basis_vector_p(),
            orbit.get_pqw_basis_vector_q(),
        );
        let compact_separate = (
            compact_orbit.get_pqw_basis_vector_p(),
            compact_orbit.get_pqw_basis_vector_q(),
        );
        let reexpanded_separate = (
            reexpanded_orbit.get_pqw_basis_vector_p(),
            reexpanded_orbit.get_pqw_basis_vector_q(),
        );

        assert_eq!(original_vectors, compact_vectors);
        assert_eq!(compact_vectors, reexpanded_vectors);
        assert_eq!(original_vectors, original_separate);
        assert_eq!(compact_vectors, compact_separate);
        assert_eq!(reexpanded_vectors, reexpanded_separate);
    }
    {
        let original_transforms = poll_transform2d(&orbit);
        let compact_transforms = poll_transform2d(&compact_orbit);
        let reexpanded_transforms = poll_transform2d(&reexpanded_orbit);

        let compact_message = format!("{compact_message} (transform)");
        let reexpanded_message = format!("{reexpanded_message} (transform)");

        for i in 0..original_transforms.len() {
            assert_eq_vec2(
                original_transforms[i],
                compact_transforms[i],
                &compact_message,
            );
            assert_eq_vec2(
                original_transforms[i],
                reexpanded_transforms[i],
                &reexpanded_message,
            );
        }
    }
    {
        let original_ecc = poll_eccentric_anomaly2d(&orbit);
        let compact_ecc = poll_eccentric_anomaly2d(&compact_orbit);
        let reexpanded_ecc = poll_eccentric_anomaly2d(&reexpanded_orbit);

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
        let original_positions = poll_orbit2d(&orbit);
        let compact_positions = poll_orbit2d(&compact_orbit);
        let reexpanded_positions = poll_orbit2d(&reexpanded_orbit);

        let compact_message = format!("{compact_message} (position)");
        let reexpanded_message = format!("{reexpanded_message} (position)");

        for i in 0..original_positions.len() {
            assert_eq_vec2(
                original_positions[i],
                compact_positions[i],
                &compact_message,
            );
            assert_eq_vec2(
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
        let original_speeds = poll_speed2d(&orbit);
        let compact_speeds = poll_speed2d(&compact_orbit);
        let reexpanded_speeds = poll_speed2d(&reexpanded_orbit);

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
        let original_vels = poll_vel2d(&orbit);
        let compact_vels = poll_vel2d(&compact_orbit);
        let reexpanded_vels = poll_vel2d(&reexpanded_orbit);

        let compact_message = format!("{compact_message} (velocity)");
        let reexpanded_message = format!("{reexpanded_message} (velocity)");

        assert_eq!(
            original_vels
                .iter()
                .map(|DVec2 { x, y }| (x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>(),
            compact_vels
                .iter()
                .map(|DVec2 { x, y }| (x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>(),
            "{compact_message}",
        );
        assert_eq!(
            original_vels
                .iter()
                .map(|DVec2 { x, y }| (x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>(),
            reexpanded_vels
                .iter()
                .map(|DVec2 { x, y }| (x.to_bits(), y.to_bits()))
                .collect::<Vec<_>>(),
            "{reexpanded_message}",
        );
    }
    {
        let original_svs = poll_sv2d(&orbit);
        let compact_svs = poll_sv2d(&compact_orbit);
        let reexpanded_svs = poll_sv2d(&reexpanded_orbit);

        let compact_message = format!("{compact_message} (velocity)");
        let reexpanded_message = format!("{reexpanded_message} (velocity)");

        assert_eq!(
            original_svs
                .iter()
                .map(|s| (
                    s.position.x.to_bits(),
                    s.position.y.to_bits(),
                    s.velocity.x.to_bits(),
                    s.velocity.y.to_bits(),
                ))
                .collect::<Vec<_>>(),
            compact_svs
                .iter()
                .map(|s| (
                    s.position.x.to_bits(),
                    s.position.y.to_bits(),
                    s.velocity.x.to_bits(),
                    s.velocity.y.to_bits(),
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
                    s.velocity.x.to_bits(),
                    s.velocity.y.to_bits(),
                ))
                .collect::<Vec<_>>(),
            reexpanded_svs
                .iter()
                .map(|s| (
                    s.position.x.to_bits(),
                    s.position.y.to_bits(),
                    s.velocity.x.to_bits(),
                    s.velocity.y.to_bits(),
                ))
                .collect::<Vec<_>>(),
            "{reexpanded_message}",
        );
    }
    {
        let original_pqw = orbit.get_pqw_basis_vectors();
        let compact_pqw = compact_orbit.get_pqw_basis_vectors();
        let reexpanded_pqw = reexpanded_orbit.get_pqw_basis_vectors();

        let compact_message = format!("{compact_message} (PQW basis vectors)");
        let reexpanded_message = format!("{reexpanded_message} (PQW basis vectors)");

        assert_eq!(original_pqw, compact_pqw, "{compact_message}");
        assert_eq!(compact_pqw, reexpanded_pqw, "{reexpanded_message}");
    }
}

fn orbit_conversion_base_test(orbit: Orbit, what: &str) {
    let compact_orbit = CompactOrbit::from(orbit.clone());
    let reexpanded_orbit = Orbit::from(compact_orbit.clone());

    let compact_message = format!("Original / Compact ({what})");
    let reexpanded_message = format!("Compact /  Reexpanded ({what})");

    {
        let original_vectors = orbit.get_pqw_basis_vectors();
        let compact_vectors = compact_orbit.get_pqw_basis_vectors();
        let reexpanded_vectors = reexpanded_orbit.get_pqw_basis_vectors();

        assert_eq!(original_vectors, compact_vectors, "{compact_message}");
        assert_eq!(compact_vectors, reexpanded_vectors, "{reexpanded_message}");

        assert_eq!(
            original_vectors,
            (
                orbit.get_pqw_basis_vector_p(),
                orbit.get_pqw_basis_vector_q(),
                orbit.get_pqw_basis_vector_w()
            ),
        );
        assert_eq!(
            compact_vectors,
            (
                compact_orbit.get_pqw_basis_vector_p(),
                compact_orbit.get_pqw_basis_vector_q(),
                compact_orbit.get_pqw_basis_vector_w()
            ),
        );
        assert_eq!(
            reexpanded_vectors,
            (
                reexpanded_orbit.get_pqw_basis_vector_p(),
                reexpanded_orbit.get_pqw_basis_vector_q(),
                reexpanded_orbit.get_pqw_basis_vector_w()
            ),
        );
    }
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
    {
        let original_pqw = orbit.get_pqw_basis_vectors();
        let compact_pqw = compact_orbit.get_pqw_basis_vectors();
        let reexpanded_pqw = reexpanded_orbit.get_pqw_basis_vectors();

        let compact_message = format!("{compact_message} (PQW basis vectors)");
        let reexpanded_message = format!("{reexpanded_message} (PQW basis vectors)");

        assert_eq!(original_pqw, compact_pqw, "{compact_message}");
        assert_eq!(compact_pqw, reexpanded_pqw, "{reexpanded_message}");
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

fn orbit_dim_parity_base_test(orbit2: &Orbit2D) {
    let orbit3 = Orbit::new(
        orbit2.get_eccentricity(),
        orbit2.get_periapsis(),
        0.0,
        orbit2.get_arg_pe(),
        0.0,
        orbit2.get_mean_anomaly_at_epoch(),
        orbit2.get_gravitational_parameter(),
    );

    {
        let mat2 = orbit2.get_transformation_matrix();
        let mat3 = orbit3.get_transformation_matrix();

        assert_eq!(mat2.x_axis.x, mat3.e11);
        assert_eq!(mat2.x_axis.y, mat3.e21);
        assert_eq!(mat2.y_axis.x, mat3.e12);
        assert_eq!(mat2.y_axis.y, mat3.e22);
        assert_eq!(0.0, mat3.e31);
        assert_eq!(0.0, mat3.e32);
    }
    {
        let (p2, q2) = orbit2.get_pqw_basis_vectors();
        let (p3, q3, w3) = orbit3.get_pqw_basis_vectors();

        assert_eq!(p2, p3.truncate());
        assert_eq!(q2, q3.truncate());
        assert_eq!(w3.truncate(), DVec2::ZERO);
    }
    {
        let tf2 = poll_transform2d(orbit2);
        let tf3 = poll_transform(&orbit3);

        for i in 0..tf2.len() {
            let dim2 = tf2[i];
            let dim3 = tf3[i];

            assert_eq!(dim2.extend(0.0), dim3);
        }
    }
    {
        let ecc2 = poll_eccentric_anomaly2d(orbit2);
        let ecc3 = poll_eccentric_anomaly(&orbit3);
        assert_eq!(ecc2, ecc3);
    }
    {
        let true2 = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let true_anomaly = orbit2.get_true_anomaly_at_mean_anomaly(angle);
                vec.push(true_anomaly);
            }

            vec
        };
        let true3 = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let true_anomaly = orbit3.get_true_anomaly_at_mean_anomaly(angle);
                vec.push(true_anomaly);
            }

            vec
        };

        for i in 0..true2.len() {
            assert_eq!(true2[i].to_bits(), true3[i].to_bits());
        }
    }
    {
        let alt2 = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let altitude = orbit2.get_altitude_at_true_anomaly(angle);
                vec.push(altitude);
            }

            vec
        };
        let alt3 = {
            let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

            for i in 0..ORBIT_POLL_ANGLES {
                let angle = (i as f64) * 2.0 * PI / (ORBIT_POLL_ANGLES as f64);
                let altitude = orbit3.get_altitude_at_true_anomaly(angle);
                vec.push(altitude);
            }

            vec
        };

        assert_eq!(alt2, alt3);
    }
    {
        let slr2 = orbit2.get_semi_latus_rectum();
        let slr3 = orbit3.get_semi_latus_rectum();

        assert_eq!(slr2.to_bits(), slr3.to_bits());
    }
    {
        let apoapsis2 = orbit2.get_apoapsis();
        let apoapsis3 = orbit3.get_apoapsis();

        assert_eq!(apoapsis2.to_bits(), apoapsis3.to_bits());
    }
    {
        for i in 0..31 {
            let true_anomaly = i as f64 * 0.1;

            let ecc2 = orbit2.get_eccentric_anomaly_at_true_anomaly(true_anomaly);
            let ecc3 = orbit3.get_eccentric_anomaly_at_true_anomaly(true_anomaly);

            assert_eq!(ecc2.to_bits(), ecc3.to_bits());
        }
    }
    {
        let speed2 = poll_speed2d(orbit2);
        let speed3 = poll_speed(&orbit3);

        assert_eq!(speed2, speed3);
    }
    {
        let vel2 = poll_vel2d(orbit2);
        let vel3 = poll_vel(&orbit3);

        for i in 0..vel2.len() {
            assert_eq!(
                vel2[i].to_array().map(f64::to_bits),
                vel3[i].truncate().to_array().map(f64::to_bits)
            );
        }
    }
    {
        let pqw2 = orbit2.get_pqw_basis_vectors();
        let pqw3 = orbit3.get_pqw_basis_vectors();

        assert_eq!(pqw2.0.extend(0.0), pqw3.0);
        assert_eq!(pqw2.1.extend(0.0), pqw3.1);
    }
    {
        let pos2 = poll_orbit2d(orbit2);
        let pos3 = poll_orbit(&orbit3);

        for i in 0..pos2.len() {
            assert_eq!(
                pos2[i].to_array().map(f64::to_bits),
                pos3[i].truncate().to_array().map(f64::to_bits),
                "Position at i={i} for {orbit2:?}"
            )
        }
    }
    {
        let sv2 = poll_sv2d(orbit2);
        let sv3 = poll_sv(&orbit3);

        for i in 0..sv2.len() {
            let sv3_element = sv3[i];

            let sv3_truncated = StateVectors2D {
                position: sv3_element.position.truncate(),
                velocity: sv3_element.velocity.truncate(),
            };

            let sv2_element = sv2[i];
            let sv2_arr = [
                sv2_element.position.to_array().map(f64::to_bits),
                sv2_element.velocity.to_array().map(f64::to_bits),
            ];
            let sv3_arr = [
                sv3_truncated.position.to_array().map(f64::to_bits),
                sv3_truncated.velocity.to_array().map(f64::to_bits),
            ];

            assert_eq!(sv2_arr, sv3_arr);
        }
    }
}

#[test]
fn orbit_dim_parity_test() {
    for orbit in random_any_2d_iter(1000) {
        orbit_dim_parity_base_test(&orbit.into());
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
fn orbit2d_conversions() {
    for orbit in random_any_2d_iter(1000) {
        let message = format!("Random orbit {orbit:?}");
        orbit2d_conversion_base_test(orbit.into(), &message);
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
        orbit_conversion_base_test(orbit.into(), message);
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

fn state_vectors_getters_base_test(orbit: &(impl OrbitTrait + Debug)) {
    let p = poll_orbit(orbit);
    let v = poll_vel(orbit);

    let sv = poll_sv(orbit);

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
        state_vectors_getters_base_test(&orbit);
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
        orbit.get_true_anomaly_at_asymptote()
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
        let f_max = orbit.get_true_anomaly_at_asymptote();
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

    assert_eq!(orbit.get_pqw_basis_vectors(), (p, q, w));
}

#[test]
fn test_orbital_plane_normal_getter() {
    for orbit in random_any_iter(262144) {
        orbital_plane_normal_base_test(&orbit);
    }
}

fn orbit_plane_an_dn_base_test(orbit: &(impl OrbitTrait + std::fmt::Debug)) {
    let other_random = random_any();
    let other_flat = {
        let mut orbit = other_random.clone();
        orbit.set_inclination(0.0);
        orbit.set_arg_pe(0.0);
        orbit.set_long_asc_node(0.0);
        assert_eq!(orbit.get_transformation_matrix(), Matrix3x2::IDENTITY);
        orbit
    };

    let flat_an = orbit.get_true_anomaly_at_asc_node_with_plane(DVec3::Z);
    let flat_dn = (flat_an + PI).rem_euclid(TAU);

    if !flat_an.is_nan() {
        let flat_an_2 = orbit.get_true_anomaly_at_asc_node();

        assert_almost_eq(
            flat_an,
            flat_an_2,
            &format!("XY AN for orbits {orbit:?} and {other_flat:?}"),
        );
    }
    if !flat_dn.is_nan() {
        let flat_dn_2 = orbit.get_true_anomaly_at_desc_node();

        assert_almost_eq(
            flat_dn,
            flat_dn_2,
            &format!("XY DN for orbits {orbit:?} and {other_flat:?}"),
        );
    }

    if !flat_an.is_nan() && !flat_dn.is_nan() {
        let flat_an_conv = (flat_dn + PI).rem_euclid(TAU);
        assert_almost_eq(
            flat_an,
            flat_an_conv,
            &format!("XY DN->AN conv ({flat_dn}->{flat_an_conv} != {flat_an})"),
        );

        let flat_dn_conv = (flat_an + PI).rem_euclid(TAU);
        assert_almost_eq(
            flat_dn,
            flat_dn_conv,
            &format!("XY AN->DN conv ({flat_an}->{flat_dn_conv} != {flat_dn})"),
        );
    }

    let other_plane = other_random.get_pqw_basis_vectors().2;
    let plane_an = orbit.get_true_anomaly_at_asc_node_with_plane(other_plane);
    let plane_dn = orbit.get_true_anomaly_at_desc_node_with_plane(other_plane);

    if !plane_an.is_nan() && !plane_dn.is_nan() {
        let plane_an_conv = (plane_dn + PI).rem_euclid(TAU);
        assert_almost_eq(
            plane_an,
            plane_an_conv,
            &format!(
                "General DN->AN conv ({plane_dn}->{plane_an_conv} != {plane_an}).\n\
                Other plane: {other_plane}\n\
                {orbit:?}"
            ),
        );

        let plane_dn_conv = (plane_an + PI).rem_euclid(TAU);
        assert_almost_eq(
            plane_dn,
            plane_dn_conv,
            &format!(
                "General AN->DN conv ({plane_an}->{plane_dn_conv} != {plane_dn}).\n\
                Other plane: {other_plane}\n\
                {orbit:?}"
            ),
        );
    }

    let v_an = orbit.get_velocity_at_true_anomaly(plane_an);
    if !v_an.is_nan() {
        let similarity = v_an.dot(other_plane);

        assert!(
            similarity > 0.0,
            "Using f = {plane_an}, expected v_an {v_an} to point roughly {other_plane}.\n\
            Got similarity of {similarity} < 0.\n\
            Orbit: {orbit:?}"
        );
    }

    let v_dn = orbit.get_velocity_at_true_anomaly(plane_dn);
    if !v_dn.is_nan() {
        let similarity = v_dn.dot(other_plane);

        assert!(
            similarity < 0.0,
            "Using f = {plane_dn}, expected v_dn {v_dn} to point roughly opposite of {other_plane}.\n\
            Got similarity of {similarity} > 0.\n\
            Orbit: {orbit:?}"
        );
    }

    let orbit_plane = orbit.get_pqw_basis_vectors().2;
    let self_an = orbit.get_true_anomaly_at_asc_node_with_plane(orbit_plane);
    assert!(self_an.is_nan());

    let self_dn = orbit.get_true_anomaly_at_desc_node_with_plane(orbit_plane);
    assert!(self_dn.is_nan());
}

#[test]
fn orbit_plane_an_dn() {
    for orbit in random_any_iter(262144) {
        orbit_plane_an_dn_base_test(&orbit);
    }
}

fn time_of_periapsis_base_test(orbit: &(impl OrbitTrait + Debug)) {
    if orbit.get_eccentricity() == 1.0 {
        assert!(!orbit.get_time_of_periapsis().is_finite());
        return;
    }

    let time = orbit.get_time_of_periapsis();

    let mean = orbit.get_mean_anomaly_at_time(time);

    assert_almost_eq(
        mean,
        0.0,
        &format!("Mean anomaly at t={time} expected 0, found {mean}\n{orbit:?}"),
    );
}

#[test]
fn time_of_periapsis() {
    let known_problematic = [Orbit::new(
        86.94922983553334,
        40745.65131311674,
        0.0,
        5.819968798514683,
        -5.7223447207239175,
        2.4398968458921217,
        596078.3249426814,
    )];

    for orbit in known_problematic {
        time_of_periapsis_base_test(&orbit);
    }

    for orbit in random_any_iter(262144) {
        time_of_periapsis_base_test(&orbit);
    }
}

fn focal_parameter_base_test(orbit: &impl OrbitTrait) {
    // Use Wikipedia's equations:
    // https://en.wikipedia.org/wiki/Conic_section#Conic_parameters
    let expected_focal_param = match orbit.get_eccentricity() {
        e if e == 0.0 => f64::INFINITY,
        e if e < 1.0 => {
            let b = orbit.get_semi_minor_axis();
            let a = orbit.get_semi_major_axis();
            b.powi(2) / (a.powi(2) - b.powi(2)).sqrt()
        }
        e if e == 1.0 => orbit.get_periapsis() * 2.0,
        e if e > 1.0 => {
            let b = orbit.get_semi_minor_axis();
            let a = orbit.get_semi_major_axis();
            b.powi(2) / (a.powi(2) + b.powi(2)).sqrt()
        }
        _ => f64::NAN,
    };

    let result = orbit.get_focal_parameter();

    if result.to_bits() == expected_focal_param.to_bits() {
        return; // Success
    }

    if result == expected_focal_param {
        return; // Success
    }

    assert_almost_eq_rescale(result, expected_focal_param, "Focal param");
}

#[test]
fn focal_parameter() {
    for orbit in random_any_iter(262144) {
        focal_parameter_base_test(&orbit);
    }
}

#[test]
fn individual_vs_combined_pqw_getters() {
    fn all_individual_pqw(orbit: &impl OrbitTrait) -> (DVec3, DVec3, DVec3) {
        let p = orbit.get_pqw_basis_vector_p();
        let q = orbit.get_pqw_basis_vector_q();
        let w = orbit.get_pqw_basis_vector_w();

        (p, q, w)
    }

    for orbit in random_any_iter(262144) {
        let combined_pqw = orbit.get_pqw_basis_vectors();
        let individual_pqw = all_individual_pqw(&orbit);

        assert_eq!(combined_pqw, individual_pqw);
    }
}

fn get_periapsis_position_base_test(orbit: &impl OrbitTrait) {
    let from_f = orbit.get_position_at_true_anomaly(0.0);
    let from_getter = orbit.get_position_at_periapsis();
    assert_almost_eq_vec3(from_f, from_getter, "Periapsis positions");
}

fn get_apoapsis_position_base_test(orbit: &(impl OrbitTrait + Debug)) {
    let from_f = orbit.get_position_at_true_anomaly(PI);

    if orbit.get_eccentricity() > 1.0 {
        return;
    } else if orbit.get_eccentricity() == 1.0 {
        assert!(!from_f.is_finite() || from_f.length() > orbit.get_periapsis() * 1000.0);
        return;
    }

    let from_getter = orbit.get_position_at_apoapsis();

    assert_almost_eq_vec3_rescale(from_f, from_getter, "Apoapsis positions");
}

#[test]
fn get_extrema_positions() {
    for orbit in random_any_iter(262144) {
        get_periapsis_position_base_test(&orbit);
        get_apoapsis_position_base_test(&orbit);
    }
}

fn periapsis_speed_base_test(orbit: &impl OrbitTrait) {
    let naive = orbit.get_speed_at_altitude(orbit.get_periapsis());
    let dedicated = orbit.get_speed_at_periapsis();

    assert_almost_eq(naive, dedicated, "Periapsis speed");
}

fn apoapsis_speed_base_test(orbit: &impl OrbitTrait) {
    let naive = orbit.get_speed_at_altitude(orbit.get_apoapsis());
    let dedicated = orbit.get_speed_at_apoapsis();

    assert_almost_eq(naive, dedicated, "Apoapsis speed");
}

fn limit_speed_base_test(orbit: &impl OrbitTrait) {
    let speed_at_inf = orbit.get_speed_at_infinity();

    if orbit.is_closed() {
        assert!(speed_at_inf.is_nan());
        return;
    }

    let apoapsis_speed = orbit.get_speed_at_apoapsis();
    let derived_speed_at_inf = apoapsis_speed * {
        let e = orbit.get_eccentricity();
        ((e + 1.0) / (e - 1.0)).sqrt()
    };

    if derived_speed_at_inf.is_nan() {
        return;
    }

    assert_almost_eq(speed_at_inf, derived_speed_at_inf, "Speed at infinity");
}

#[test]
fn get_extrema_speeds() {
    for orbit in random_any_iter(262144) {
        periapsis_speed_base_test(&orbit);
        apoapsis_speed_base_test(&orbit);
        limit_speed_base_test(&orbit);
    }
}

fn periapsis_vel_base_test(orbit: &impl OrbitTrait) {
    let naive = orbit.get_velocity_at_eccentric_anomaly(0.0);
    let specific = orbit.get_velocity_at_periapsis();
    let speed = orbit.get_speed_at_periapsis();

    if naive.is_finite() {
        assert_almost_eq_vec3_rescale(naive, specific, "Periapsis vel");
    }
    assert_almost_eq_rescale(specific.length(), speed, "Periapsis vel vs spd");
}

fn apoapsis_vel_base_test(orbit: &impl OrbitTrait) {
    let specific = orbit.get_velocity_at_apoapsis();
    let speed = orbit.get_speed_at_apoapsis();

    if specific.length() < f64::EPSILON || speed.abs() < f64::EPSILON {
        assert_almost_eq(specific.length(), speed, "Apoapsis vel vs spd");
    } else {
        assert_almost_eq_rescale(specific.length(), speed, "Apoapsis vel vs spd");
    }

    if orbit.is_open() {
        return;
    }

    let naive = orbit.get_velocity_at_eccentric_anomaly(PI);

    if naive.length() < f64::EPSILON || specific.length() < f64::EPSILON {
        assert_almost_eq_vec3(naive, specific, "Apoapsis vel");
    } else {
        assert_almost_eq_vec3_rescale(naive, specific, "Apoapsis vel");
    }
}

fn asymp_vel_base_test(orbit: &impl OrbitTrait) {
    let asymp_speed = orbit.get_speed_at_infinity();
    let asymp_in_vel = orbit.get_velocity_at_incoming_asymptote();
    let asymp_out_vel = orbit.get_velocity_at_outgoing_asymptote();

    if orbit.is_closed() {
        assert!(asymp_speed.is_nan());
        assert!(asymp_in_vel.is_nan());
        assert!(asymp_out_vel.is_nan());
        return;
    }

    assert_almost_eq(asymp_speed, asymp_in_vel.length(), "Asymp spd vs in vel");
    assert_almost_eq(asymp_speed, asymp_out_vel.length(), "Asymp spd vs out vel");

    const ECC_ANOM_STEP: f64 = 100.0;
    for i in -100..100 {
        let ecc_anom = i as f64 * ECC_ANOM_STEP;
        let speed = orbit.get_speed_at_eccentric_anomaly(ecc_anom);

        if speed.is_nan() {
            continue;
        }

        // Slight algorithmic differences cause there to be slight differences
        // meaning we have to tolerate being just a little less than
        // the asymptote speed
        const TOLERANCE_FACTOR: f64 = 1.0000001;

        assert!(
            speed * TOLERANCE_FACTOR >= asymp_speed,
            "{speed} > {asymp_speed} should be true"
        );
    }

    const HUGE_ECC_ANOM: f64 = 10.0;

    let past_vel = orbit.get_velocity_at_eccentric_anomaly(-HUGE_ECC_ANOM);
    let fut_vel = orbit.get_velocity_at_eccentric_anomaly(HUGE_ECC_ANOM);

    if past_vel.is_finite() {
        assert!(
            asymp_in_vel.dot(past_vel) >= 0.0,
            "{asymp_in_vel} should point the same way as {past_vel}"
        );
    }
    if fut_vel.is_finite() {
        assert!(
            asymp_out_vel.dot(fut_vel) >= 0.0,
            "{asymp_out_vel} should point the same way as {fut_vel}"
        );
    }
}

#[test]
fn get_extrema_velocities() {
    for orbit in random_any_iter(65536) {
        periapsis_vel_base_test(&orbit);
        apoapsis_vel_base_test(&orbit);
        asymp_vel_base_test(&orbit);
    }
}

#[test]
fn specific_energy() {
    for orbit in random_any_iter(262144) {
        if orbit.is_closed() {
            assert!(orbit.get_specific_orbital_energy() < 0.0);
        } else if orbit.is_parabolic() {
            assert_eq!(orbit.get_specific_orbital_energy(), 0.0);
        } else if orbit.is_hyperbolic() {
            assert!(orbit.get_specific_orbital_energy() > 0.0);
        }
    }
}

#[derive(Clone, Copy)]
enum OrbitMutation {
    Eccentricity(f64),
    Periapsis(f64),
    Inclination(f64),
    ArgPe(f64),
    LongAscNode(f64),
    MeanAnomaly(f64),
    Mu(f64),
}

impl OrbitMutation {
    fn new_random() -> Self {
        let val = random_mult();
        match rand::random_range(0..7) {
            0 => Self::Eccentricity(val),
            1 => Self::Periapsis(val),
            2 => Self::Inclination(val),
            3 => Self::ArgPe(val),
            4 => Self::LongAscNode(val),
            5 => Self::MeanAnomaly(val),
            _ => Self::Mu(val),
        }
    }

    fn mutate(self, orbit: &mut impl OrbitTrait) {
        match self {
            OrbitMutation::Eccentricity(e) => orbit.set_eccentricity(e),
            OrbitMutation::Periapsis(rp) => orbit.set_periapsis(rp),
            OrbitMutation::Inclination(i) => orbit.set_inclination(i),
            OrbitMutation::ArgPe(arg_pe) => orbit.set_arg_pe(arg_pe),
            OrbitMutation::LongAscNode(long_asc_node) => orbit.set_long_asc_node(long_asc_node),
            OrbitMutation::MeanAnomaly(m0) => orbit.set_mean_anomaly_at_epoch(m0),
            OrbitMutation::Mu(mu) => {
                orbit.set_gravitational_parameter(mu, MuSetterMode::KeepElements)
            }
        }
    }
}

fn cache_coherency_base_test(compact_orbit: &mut CompactOrbit) {
    const CACHE_COHERENCY_ITERS: usize = 128;
    // let mut compact_orbit: CompactOrbit = cached_orbit.clone().into();
    let mut cached_orbit: Orbit = compact_orbit.clone().into();

    for _ in 0..CACHE_COHERENCY_ITERS {
        let mutation = OrbitMutation::new_random();

        mutation.mutate(&mut cached_orbit);
        mutation.mutate(compact_orbit);

        assert_eq_orbit(
            &cached_orbit,
            compact_orbit,
            "cached vs compact post-mutation",
        );
    }
}

#[test]
fn cache_coherency() {
    for mut orbit in random_any_iter(16384) {
        cache_coherency_base_test(&mut orbit);
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
    use super::*;
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
