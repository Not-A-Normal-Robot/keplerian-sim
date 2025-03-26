use glam::DVec2;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{ApoapsisSetterError, Matrix3x2, Orbit, OrbitTrait};

/// A minimal struct representing a Keplerian orbit.
///
/// This struct minimizes memory footprint by not caching variables.  
/// Because of this, calculations can be slower than caching those variables.  
/// For this reason, you might consider using the `Orbit` struct instead.
///
/// # Example
/// ```
/// use keplerian_sim::{CompactOrbit, OrbitTrait};
///
/// let orbit = CompactOrbit::new(
///     // Initialize using eccentricity, periapsis, inclination,
///     // argument of periapsis, longitude of ascending node,
///     // and mean anomaly at epoch
///
///     // Eccentricity
///     0.0,
///
///     // Periapsis
///     1.0,
///
///     // Inclination
///     0.0,
///
///     // Argument of periapsis
///     0.0,
///
///     // Longitude of ascending node
///     0.0,
///
///     // Mean anomaly at epoch
///     0.0,
///
///     // Mass of the parent body
///     1.0
/// );
///
/// let orbit = CompactOrbit::with_apoapsis(
///     // Initialize using apoapsis in place of eccentricity
///     
///     // Apoapsis
///     2.0,
///
///     // Periapsis
///     1.0,
///
///     // Inclination
///     0.0,
///
///     // Argument of periapsis
///     0.0,
///
///     // Longitude of ascending node
///     0.0,
///
///     // Mean anomaly at epoch
///     0.0,
///
///     // Mass of the parent body
///     1.0
/// );
/// ```
/// See [Orbit::new] and [Orbit::with_apoapsis] for more information.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CompactOrbit {
    /// The eccentricity of the orbit.  
    /// e < 1: ellipse  
    /// e = 1: parabola  
    /// e > 1: hyperbola  
    ///
    /// See more: <https://en.wikipedia.org/wiki/Orbital_eccentricity>
    pub eccentricity: f64,

    /// The periapsis of the orbit, in meters.
    ///
    /// The periapsis of an orbit is the distance at the closest point
    /// to the parent body.
    ///
    /// More simply, this is the "minimum altitude" of an orbit.
    pub periapsis: f64,

    /// The inclination of the orbit, in radians.
    /// The inclination of an orbit is the angle between the plane of the
    /// orbit and the reference plane.
    ///
    /// In simple terms, it tells you how "tilted" the orbit is.
    pub inclination: f64,

    /// The argument of periapsis of the orbit, in radians.
    ///
    /// Wikipedia:  
    /// The argument of periapsis is the angle from the body's
    /// ascending node to its periapsis, measured in the direction of
    /// motion.  
    /// <https://en.wikipedia.org/wiki/Argument_of_periapsis>
    ///
    /// In simple terms, it tells you how, and in which direction,
    /// the orbit "tilts".
    pub arg_pe: f64,

    /// The longitude of ascending node of the orbit, in radians.
    ///
    /// Wikipedia:  
    /// The longitude of ascending node is the angle from a specified
    /// reference direction, called the origin of longitude, to the direction
    /// of the ascending node, as measured in a specified reference plane.  
    /// <https://en.wikipedia.org/wiki/Longitude_of_the_ascending_node>
    ///
    /// In simple terms, it tells you how, and in which direction,
    /// the orbit "tilts".
    pub long_asc_node: f64,

    /// The mean anomaly at orbit epoch, in radians.
    ///
    /// For elliptic orbits, it's measured in radians and so are bounded
    /// between 0 and tau; anything out of range will get wrapped around.  
    /// For hyperbolic orbits, it's unbounded.
    ///
    /// Wikipedia:  
    /// The mean anomaly at epoch, `M_0`, is defined as the instantaneous mean
    /// anomaly at a given epoch, `t_0`.  
    /// <https://en.wikipedia.org/wiki/Mean_anomaly#Mean_anomaly_at_epoch>
    ///
    /// In simple terms, this modifies the "offset" of the orbit progression.
    pub mean_anomaly: f64,

    /// The mass of the parent body, in kilograms.
    pub parent_mass: f64,
}

impl OrbitTrait for CompactOrbit {
    fn new(
        eccentricity: f64,
        periapsis: f64,
        inclination: f64,
        arg_pe: f64,
        long_asc_node: f64,
        mean_anomaly: f64,
        parent_mass: f64,
    ) -> CompactOrbit {
        CompactOrbit {
            eccentricity,
            periapsis,
            inclination,
            arg_pe,
            long_asc_node,
            mean_anomaly,
            parent_mass,
        }
    }

    fn with_apoapsis(
        apoapsis: f64,
        periapsis: f64,
        inclination: f64,
        arg_pe: f64,
        long_asc_node: f64,
        mean_anomaly: f64,
        parent_mass: f64,
    ) -> CompactOrbit {
        let eccentricity = (apoapsis - periapsis) / (apoapsis + periapsis);
        CompactOrbit::new(
            eccentricity,
            periapsis,
            inclination,
            arg_pe,
            long_asc_node,
            mean_anomaly,
            parent_mass,
        )
    }

    fn get_parent_mass(&self) -> f64 {
        self.parent_mass
    }

    fn get_semi_major_axis(&self) -> f64 {
        self.periapsis / (1.0 - self.eccentricity)
    }

    fn get_semi_minor_axis(&self) -> f64 {
        let semi_major_axis = self.get_semi_major_axis();
        let eccentricity_squared = self.eccentricity * self.eccentricity;
        semi_major_axis * (1.0 - eccentricity_squared).abs().sqrt()
    }

    fn get_linear_eccentricity(&self) -> f64 {
        self.get_semi_major_axis() - self.periapsis
    }

    fn set_apoapsis(&mut self, apoapsis: f64, _: f64) -> Result<(), ApoapsisSetterError> {
        if apoapsis < 0.0 {
            Err(ApoapsisSetterError::ApoapsisNegative)
        } else if apoapsis < self.periapsis {
            Err(ApoapsisSetterError::ApoapsisLessThanPeriapsis)
        } else {
            self.eccentricity = (apoapsis - self.periapsis) / (apoapsis + self.periapsis);

            Ok(())
        }
    }

    fn set_apoapsis_force(&mut self, apoapsis: f64, _: f64) {
        let mut apoapsis = apoapsis;
        if apoapsis < self.periapsis && apoapsis > 0.0 {
            (apoapsis, self.periapsis) = (self.periapsis, apoapsis);
        }

        self.eccentricity = (apoapsis - self.periapsis) / (apoapsis + self.periapsis);
    }

    fn get_transformation_matrix(&self) -> Matrix3x2 {
        let mut matrix = Matrix3x2::default();

        let (sin_inc, cos_inc) = self.inclination.sin_cos();
        let (sin_arg_pe, cos_arg_pe) = self.arg_pe.sin_cos();
        let (sin_lan, cos_lan) = self.long_asc_node.sin_cos();

        // https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
        matrix.e11 = cos_arg_pe * cos_lan - sin_arg_pe * cos_inc * sin_lan;
        matrix.e12 = -(sin_arg_pe * cos_lan + cos_arg_pe * cos_inc * sin_lan);

        matrix.e21 = cos_arg_pe * sin_lan + sin_arg_pe * cos_inc * cos_lan;
        matrix.e22 = cos_arg_pe * cos_inc * cos_lan - sin_arg_pe * sin_lan;

        matrix.e31 = sin_arg_pe * sin_inc;
        matrix.e32 = cos_arg_pe * sin_inc;

        matrix
    }

    #[inline]
    fn get_eccentricity(&self) -> f64 {
        self.eccentricity
    }

    #[inline]
    fn get_periapsis(&self) -> f64 {
        self.periapsis
    }

    #[inline]
    fn get_inclination(&self) -> f64 {
        self.inclination
    }

    #[inline]
    fn get_arg_pe(&self) -> f64 {
        self.arg_pe
    }

    #[inline]
    fn get_long_asc_node(&self) -> f64 {
        self.long_asc_node
    }

    #[inline]
    fn get_mean_anomaly_at_epoch(&self) -> f64 {
        self.mean_anomaly
    }

    fn get_sqrt_mu_sma(&self, g: f64) -> f64 {
        crate::get_sqrt_mu_sma(self, g)
    }

    fn get_velocity_unit_vector(&self) -> DVec2 {
        crate::get_velocity_unit_vector(self)
    }

    fn set_eccentricity(&mut self, value: f64, _: f64) {
        self.eccentricity = value
    }
    fn set_periapsis(&mut self, value: f64, _: f64) {
        self.periapsis = value
    }
    fn set_inclination(&mut self, value: f64, _: f64) {
        self.inclination = value
    }
    fn set_arg_pe(&mut self, value: f64, _: f64) {
        self.arg_pe = value
    }
    fn set_long_asc_node(&mut self, value: f64, _: f64) {
        self.long_asc_node = value
    }
    fn set_mean_anomaly_at_epoch(&mut self, value: f64, _: f64) {
        self.mean_anomaly = value
    }
}

impl Default for CompactOrbit {
    /// Creates a unit orbit.
    ///
    /// The unit orbit is a perfect circle of radius 1 and no "tilt".
    fn default() -> CompactOrbit {
        Self::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    }
}

impl From<Orbit> for CompactOrbit {
    fn from(cached: Orbit) -> Self {
        Self {
            eccentricity: cached.get_eccentricity(),
            periapsis: cached.get_periapsis(),
            inclination: cached.get_inclination(),
            arg_pe: cached.get_arg_pe(),
            long_asc_node: cached.get_long_asc_node(),
            mean_anomaly: cached.get_mean_anomaly_at_epoch(),
            parent_mass: cached.get_parent_mass(),
        }
    }
}
