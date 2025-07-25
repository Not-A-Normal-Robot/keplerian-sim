#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{ApoapsisSetterError, Matrix3x2, Orbit, OrbitTrait};
use std::f64::consts::{PI, TAU};

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
///     // Gravitational parameter of the parent body
///     1.0,
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
///     // Gravitational parameter of the parent body
///     1.0,
/// );
/// ```
/// See [Orbit::new] and [Orbit::with_apoapsis] for more information.
#[derive(Clone, Debug, PartialEq)]
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

    /// The gravitational parameter of the parent body.
    ///
    /// This is a constant value that represents the mass of the parent body
    /// multiplied by the gravitational constant.
    ///
    /// In other words, mu = GM.
    pub mu: f64,
}

// Initialization and cache management
impl CompactOrbit {
    /// Creates a new `CompactOrbit` instance with the given parameters.
    ///
    /// Note: This function uses eccentricity instead of apoapsis.  
    /// If you want to provide an apoapsis instead, consider using the
    /// [`CompactOrbit::with_apoapsis`] function instead.
    ///
    /// ### Parameters
    /// - `eccentricity`: The eccentricity of the orbit.
    /// - `periapsis`: The periapsis of the orbit, in meters.
    /// - `inclination`: The inclination of the orbit, in radians.
    /// - `arg_pe`: The argument of periapsis of the orbit, in radians.
    /// - `long_asc_node`: The longitude of ascending node of the orbit, in radians.
    /// - `mean_anomaly`: The mean anomaly of the orbit, in radians.
    /// - `mu`: The gravitational parameter of the parent body, in m^3 s^-2.
    pub fn new(
        eccentricity: f64,
        periapsis: f64,
        inclination: f64,
        arg_pe: f64,
        long_asc_node: f64,
        mean_anomaly: f64,
        mu: f64,
    ) -> CompactOrbit {
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

    /// Creates a new `CompactOrbit` instance with the given parameters.
    ///
    /// Note: This function uses apoapsis instead of eccentricity.  
    /// Because of this, it's not recommended to initialize
    /// parabolic or hyperbolic 'orbits' with this function.  
    /// If you're looking to initialize a parabolic or hyperbolic
    /// trajectory, consider using the [`CompactOrbit::new`] function instead.
    ///
    /// ### Parameters
    /// - `apoapsis`: The apoapsis of the orbit, in meters.
    /// - `periapsis`: The periapsis of the orbit, in meters.
    /// - `inclination`: The inclination of the orbit, in radians.
    /// - `arg_pe`: The argument of periapsis of the orbit, in radians.
    /// - `long_asc_node`: The longitude of ascending node of the orbit, in radians.
    /// - `mean_anomaly`: The mean anomaly of the orbit, in radians.
    /// - `mu`: The gravitational parameter of the parent body, in m^3 s^-2.
    pub fn with_apoapsis(
        apoapsis: f64,
        periapsis: f64,
        inclination: f64,
        arg_pe: f64,
        long_asc_node: f64,
        mean_anomaly: f64,
        mu: f64,
    ) -> CompactOrbit {
        let eccentricity = (apoapsis - periapsis) / (apoapsis + periapsis);
        CompactOrbit::new(
            eccentricity,
            periapsis,
            inclination,
            arg_pe,
            long_asc_node,
            mean_anomaly,
            mu,
        )
    }
}

impl OrbitTrait for CompactOrbit {
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

    fn set_apoapsis(&mut self, apoapsis: f64) -> Result<(), ApoapsisSetterError> {
        if apoapsis < 0.0 {
            Err(ApoapsisSetterError::ApoapsisNegative)
        } else if apoapsis < self.periapsis {
            Err(ApoapsisSetterError::ApoapsisLessThanPeriapsis)
        } else {
            self.eccentricity = (apoapsis - self.periapsis) / (apoapsis + self.periapsis);

            Ok(())
        }
    }

    fn set_apoapsis_force(&mut self, apoapsis: f64) {
        let mut apoapsis = apoapsis;
        if apoapsis < self.periapsis && apoapsis >= 0.0 {
            (apoapsis, self.periapsis) = (self.periapsis, apoapsis);
            self.arg_pe = (self.arg_pe + PI).rem_euclid(TAU);
            self.mean_anomaly = (self.mean_anomaly + PI).rem_euclid(TAU);
        }

        if apoapsis < 0.0 && apoapsis > -self.periapsis {
            // Even for hyperbolic orbits, apoapsis cannot be between 0 and -periapsis
            // We will interpret this as an infinite apoapsis (parabolic trajectory)
            self.eccentricity = 1.0;
        } else {
            self.eccentricity = (apoapsis - self.periapsis) / (apoapsis + self.periapsis);
        }
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

    #[inline]
    fn get_gravitational_parameter(&self) -> f64 {
        self.mu
    }

    fn set_eccentricity(&mut self, value: f64) {
        self.eccentricity = value
    }
    fn set_periapsis(&mut self, value: f64) {
        self.periapsis = value
    }
    fn set_inclination(&mut self, value: f64) {
        self.inclination = value
    }
    fn set_arg_pe(&mut self, value: f64) {
        self.arg_pe = value
    }
    fn set_long_asc_node(&mut self, value: f64) {
        self.long_asc_node = value
    }
    fn set_mean_anomaly_at_epoch(&mut self, value: f64) {
        self.mean_anomaly = value
    }

    fn set_gravitational_parameter(
        &mut self,
        gravitational_parameter: f64,
        mode: crate::MuSetterMode,
    ) {
        let new_mu = gravitational_parameter;
        match mode {
            crate::MuSetterMode::KeepElements => {
                self.mu = new_mu;
            }
            crate::MuSetterMode::KeepPositionAtTime(t) => {
                // We need to keep the position at time t
                // This means keeping the mean anomaly at that point, since the
                // orbit shape does not change
                // Current mean anomaly:
                // M_1(t) = M_0_1 + t * sqrt(mu_1 / |a^3|)
                //
                // Mean anomaly after mu changes:
                // M_2(t) = M_0_2 + t * sqrt(mu_2 / |a^3|)
                //
                // M_1(t) = M_2(t)
                //
                // We need to find M_0_2
                //
                // M_0_1 + t * sqrt(mu_1 / |a^3|) = M_0_2 + t * sqrt(mu_2 / |a^3|)
                // M_0_2 + t * sqrt(mu_2 / |a^3|) = M_0_1 + t * sqrt(mu_1 / |a^3|)
                // M_0_2 = M_0_1 + t * sqrt(mu_1 / |a^3|) - t * sqrt(mu_2 / |a^3|)
                // M_0_2 = M_0_1 + t * (sqrt(mu_1 / |a^3|) - sqrt(mu_2 / |a^3|))
                let inv_abs_a_cubed = self.get_semi_major_axis().powi(3).abs().recip();

                self.mean_anomaly +=
                    t * ((self.mu * inv_abs_a_cubed).sqrt() - (new_mu * inv_abs_a_cubed).sqrt());

                self.mu = new_mu;
            }
            crate::MuSetterMode::KeepKnownStateVectors {
                state_vectors,
                time,
            } => {
                let new = state_vectors.to_compact_orbit(new_mu, time);
                *self = new;
            }
            crate::MuSetterMode::KeepStateVectorsAtTime(time) => {
                let ecc_anom = self.get_eccentric_anomaly_at_time(time);
                let state_vectors = self.get_state_vectors_at_eccentric_anomaly(ecc_anom);
                let new = state_vectors.to_compact_orbit(new_mu, time);
                *self = new;
            }
        }
    }
}

impl Default for CompactOrbit {
    /// Creates a unit orbit.
    ///
    /// The unit orbit is a perfect circle of radius 1 and no "tilt".
    ///
    /// It also uses a gravitational parameter of 1.
    fn default() -> Self {
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
            mu: cached.get_gravitational_parameter(),
        }
    }
}
