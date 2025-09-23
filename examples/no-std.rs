#![no_std]

use glam::DVec3;
use keplerian_sim::{Orbit, OrbitTrait, StateVectors};

fn main() {
    let orbit = Orbit::new(1.2, 4.9, 1.3, 2.2, 3.0, 0.8, 1.0);
    let sv = orbit.get_state_vectors_at_time(10.0);

    assert_eq!(
        sv,
        StateVectors {
            position: DVec3::new(12.591001801881454, 8.188842573472511, -35.602225964159544),
            velocity: DVec3::new(
                0.015701488836984412,
                0.07996611716514077,
                -0.29314504143026743
            ),
        }
    );
}
