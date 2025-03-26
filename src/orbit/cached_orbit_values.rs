#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::Matrix3x2;

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub(super) struct CachedOrbitValues {
    /// The semi-major axis of the orbit, in meters.
    pub semi_major_axis: f64,

    /// The semi-minor axis of the orbit, in meters.
    pub semi_minor_axis: f64,

    /// The linear eccentricity of the orbit, in meters.
    pub linear_eccentricity: f64,

    /// The transformation matrix to tilt the 2D planar orbit into 3D space.
    pub transformation_matrix: Matrix3x2,

    /// A value based on the orbit's eccentricity, used to calculate
    /// the true anomaly from the eccentric anomaly.  
    /// https://en.wikipedia.org/wiki/True_anomaly#From_the_eccentric_anomaly
    pub beta: f64,
}
