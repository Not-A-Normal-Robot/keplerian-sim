use std::ops::Add;

use glam::DVec3;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Orbital state vectors, relative to a parent object.
#[derive(Default, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StateVectors {
    /// The position of the object.
    pub position: DVec3,
    /// The velocity of the object.
    pub velocity: DVec3,
}

impl Add for StateVectors {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            position: self.position + rhs.position,
            velocity: self.velocity + rhs.velocity,
        }
    }
}
