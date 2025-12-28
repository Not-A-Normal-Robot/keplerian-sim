use core::f64::consts::TAU;

use glam::{DVec2, DVec3};

use crate::{tests::ORBIT_POLL_ANGLES, OrbitTrait, OrbitTrait2D, StateVectors, StateVectors2D};
extern crate std;
use std::vec::Vec;

pub fn poll_orbit(orbit: &impl OrbitTrait) -> Vec<DVec3> {
    let mut vec: Vec<DVec3> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let time = if orbit.get_eccentricity() < 1.0 {
            (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
        } else {
            (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
        };
        vec.push(orbit.get_position_at_time(time));
    }

    vec
}

pub fn poll_orbit2d(orbit: &impl OrbitTrait2D) -> Vec<DVec2> {
    (0..ORBIT_POLL_ANGLES)
        .map(|i| {
            let time = if orbit.get_eccentricity() < 1.0 {
                (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
            } else {
                (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
            };
            orbit.get_position_at_time(time)
        })
        .collect()
}

pub fn poll_flat(orbit: &impl OrbitTrait) -> Vec<DVec2> {
    let mut vec: Vec<DVec2> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let angle = (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64);
        vec.push(orbit.get_pqw_position_at_true_anomaly(angle));
    }

    vec
}

pub fn poll_transform(orbit: &impl OrbitTrait) -> Vec<DVec3> {
    let mut vec: Vec<DVec3> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let angle = (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64);
        vec.push(orbit.transform_pqw_vector(DVec2::new(1.0 * angle.cos(), 1.0 * angle.sin())));
    }

    vec
}

pub fn poll_transform2d(orbit: &impl OrbitTrait2D) -> Vec<DVec2> {
    (0..ORBIT_POLL_ANGLES)
        .map(|i| {
            let angle = (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64);
            orbit.transform_pqw_vector(DVec2::new(1.0 * angle.cos(), 1.0 * angle.sin()))
        })
        .collect()
}

pub fn poll_eccentric_anomaly(orbit: &impl OrbitTrait) -> Vec<f64> {
    let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let time = if orbit.get_eccentricity() < 1.0 {
            (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
        } else {
            (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
        };

        vec.push(orbit.get_eccentric_anomaly_at_time(time));
    }

    vec
}

pub fn poll_eccentric_anomaly2d(orbit: &impl OrbitTrait2D) -> Vec<f64> {
    let mut vec: Vec<f64> = Vec::with_capacity(ORBIT_POLL_ANGLES);

    for i in 0..ORBIT_POLL_ANGLES {
        let time = if orbit.get_eccentricity() < 1.0 {
            (i as f64) * orbit.get_orbital_period() / (ORBIT_POLL_ANGLES as f64)
        } else {
            (i as f64) * TAU / (ORBIT_POLL_ANGLES as f64)
        };

        vec.push(orbit.get_eccentric_anomaly_at_time(time));
    }

    vec
}

pub fn poll_speed(orbit: &impl OrbitTrait) -> Vec<f64> {
    (0..ORBIT_POLL_ANGLES)
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

pub fn poll_speed2d(orbit: &impl OrbitTrait2D) -> Vec<f64> {
    (0..ORBIT_POLL_ANGLES)
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

pub fn poll_flat_vel(orbit: &impl OrbitTrait) -> Vec<DVec2> {
    (0..ORBIT_POLL_ANGLES)
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

pub fn poll_vel(orbit: &impl OrbitTrait) -> Vec<DVec3> {
    (0..ORBIT_POLL_ANGLES)
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

pub fn poll_vel2d(orbit: &impl OrbitTrait2D) -> Vec<DVec2> {
    (0..ORBIT_POLL_ANGLES)
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

pub fn poll_sv(orbit: &impl OrbitTrait) -> Vec<StateVectors> {
    (0..ORBIT_POLL_ANGLES)
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

pub fn poll_sv2d(orbit: &impl OrbitTrait2D) -> Vec<StateVectors2D> {
    (0..ORBIT_POLL_ANGLES)
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
