use glam::DVec2;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CachedStateVectorValues {
    /// `(mu * semi_major_axis).sqrt()`.
    /// This is used when calculating velocity.
    pub sqrt_mu_sma: f64,

    /// A unit vector that can be used to calculate the velocity.
    pub velocity_unit_vector: DVec2,
}
