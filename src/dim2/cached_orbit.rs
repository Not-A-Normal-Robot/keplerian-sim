use core::f64::consts::{PI, TAU};
use glam::{DMat2, DVec2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "libm")]
#[allow(unused_imports)]
use crate::math::F64Math;
use crate::{ApoapsisSetterError, CompactOrbit2D, MuSetterMode2D, OrbitTrait2D};

/// A struct representing a 2D Keplerian orbit with some cached values.
///
/// This struct consumes significantly more memory because of the cache.
/// However, this will slightly speed up orbital calculations.
/// If memory efficiency is your goal, you may consider using the [`CompactOrbit2D`]
/// struct instead.
///
/// # Example
/// ```
/// use keplerian_sim::{Orbit2D, OrbitTrait2D};
///
/// let orbit = Orbit2D::new(
///     // Initialize using eccentricity, periapsis,
///     // argument of periapsis, mean anomaly at epoch,
///     // and gravitational parameter
///
///     // Eccentricity
///     0.0,
///
///     // Periapsis
///     1.0,
///
///     // Argument of periapsis
///     0.0,
///
///     // Mean anomaly at epoch
///     0.0,
///
///     // Gravitational parameter of the parent body
///     1.0,
/// );
///
/// let orbit = Orbit2D::with_apoapsis(
///     // Initialize using apoapsis in place of eccentricity
///     
///     // Apoapsis
///     2.0,
///
///     // Periapsis
///     1.0,
///
///     // Argument of periapsis
///     0.0,
///
///     // Mean anomaly at epoch
///     0.0,
///
///     // Gravitational parameter of the parent body
///     1.0,
/// );
/// ```
///
/// See [Orbit2D::new] and [Orbit2D::with_apoapsis] for more information.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Orbit2D {
    /// The eccentricity of the orbit.  
    /// e < 1: ellipse  
    /// e = 1: parabola  
    /// e > 1: hyperbola  
    ///
    /// See more: <https://en.wikipedia.org/wiki/Orbital_eccentricity>
    eccentricity: f64,

    /// The periapsis of the orbit, in meters.
    ///
    /// The periapsis of an orbit is the distance at the closest point
    /// to the parent body.
    ///
    /// More simply, this is the "minimum altitude" of an orbit.
    periapsis: f64,

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
    arg_pe: f64,

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
    mean_anomaly: f64,

    /// The gravitational parameter of the parent body.
    ///
    /// This is a constant value that represents the mass of the parent body
    /// multiplied by the gravitational constant.
    ///
    /// In other words, mu = GM.
    mu: f64,

    cache: OrbitCachedCalculations,
}

// -------- MEMO --------
// When updating this struct, please review the following methods:
// `Orbit2D::get_cached_calculations()`
// `<Orbit2D as OrbitTrait2D>::set_*()`
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
struct OrbitCachedCalculations {
    /// The transformation matrix to transform from the 2D PQW space into 3D space.
    transformation_matrix: DMat2,
}

impl Orbit2D {
    /// Creates a new orbit with the given parameters.
    ///
    /// Note: This function uses eccentricity instead of periapsis.  
    /// If you want to provide an apoapsis instead, consider using the
    /// [`Orbit2D::with_apoapsis`] function instead.
    ///
    /// # Parameters
    /// - `eccentricity`: The eccentricity of the orbit.
    /// - `periapsis`: The periapsis of the orbit, in meters.
    /// - `arg_pe`: The argument of periapsis of the orbit, in radians.
    /// - `mean_anomaly`: The mean anomaly of the orbit at epoch, in radians.
    /// - `mu`: The gravitational parameter of the parent body, in m^3 s^-2.
    ///
    /// # Example
    ///
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let eccentricity = 0.2;
    /// let periapsis = 2.8;
    /// let argument_of_periapsis = 0.7;
    /// let mean_anomaly_at_epoch = 2.9;
    /// let gravitational_parameter = 9.2;
    ///
    /// let orbit = Orbit2D::new(
    ///     eccentricity,
    ///     periapsis,
    ///     argument_of_periapsis,
    ///     mean_anomaly_at_epoch,
    ///     gravitational_parameter
    /// );
    ///
    /// assert_eq!(orbit.get_eccentricity(), eccentricity);
    /// assert_eq!(orbit.get_periapsis(), periapsis);
    /// assert_eq!(orbit.get_arg_pe(), argument_of_periapsis);
    /// assert_eq!(orbit.get_mean_anomaly_at_epoch(), mean_anomaly_at_epoch);
    /// assert_eq!(orbit.get_gravitational_parameter(), gravitational_parameter);
    /// ```
    pub fn new(eccentricity: f64, periapsis: f64, arg_pe: f64, mean_anomaly: f64, mu: f64) -> Self {
        let cache = Self::get_cached_calculations(arg_pe);
        Self {
            eccentricity,
            periapsis,
            arg_pe,
            mean_anomaly,
            mu,
            cache,
        }
    }

    /// Creates a new orbit with the given parameters.
    ///
    /// Note: This function uses apoapsis instead of eccentricity.  
    /// Because of this, it's not recommended to initialize
    /// parabolic or hyperbolic 'orbits' with this function.  
    /// If you're looking to initialize a parabolic or hyperbolic
    /// trajectory, consider using the [`Orbit2D::new`] function instead.
    ///
    /// # Parameters
    /// - `apoapsis`: The apoapsis of the orbit, in meters.
    /// - `periapsis`: The periapsis of the orbit, in meters.
    /// - `arg_pe`: The argument of periapsis of the orbit, in radians.
    /// - `mean_anomaly`: The mean anomaly of the orbit at epoch, in radians.
    /// - `mu`: The gravitational parameter of the parent body, in m^3 s^-2.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let apoapsis = 4.1;
    /// let periapsis = 2.8;
    /// let argument_of_periapsis = 0.7;
    /// let mean_anomaly_at_epoch = 2.9;
    /// let gravitational_parameter = 9.2;
    ///
    /// let orbit = Orbit2D::with_apoapsis(
    ///     apoapsis,
    ///     periapsis,
    ///     argument_of_periapsis,
    ///     mean_anomaly_at_epoch,
    ///     gravitational_parameter
    /// );
    ///
    /// let eccentricity = (apoapsis - periapsis) / (apoapsis + periapsis);
    ///
    /// assert_eq!(orbit.get_eccentricity(), eccentricity);
    /// assert_eq!(orbit.get_periapsis(), periapsis);
    /// assert_eq!(orbit.get_arg_pe(), argument_of_periapsis);
    /// assert_eq!(orbit.get_mean_anomaly_at_epoch(), mean_anomaly_at_epoch);
    /// assert_eq!(orbit.get_gravitational_parameter(), gravitational_parameter);
    /// ```
    pub fn with_apoapsis(
        apoapsis: f64,
        periapsis: f64,
        arg_pe: f64,
        mean_anomaly: f64,
        mu: f64,
    ) -> Self {
        let eccentricity = (apoapsis - periapsis) / (apoapsis + periapsis);
        Self::new(eccentricity, periapsis, arg_pe, mean_anomaly, mu)
    }

    /// Creates a new circular `Orbit2D` instance with the given parameters.
    ///
    /// # Parameters
    /// - `radius`: The radius of the orbit, in meters.
    /// - `arg_pe`: The argument of periapsis of the orbit, in radians.
    /// - `mean_anomaly`: The mean anomaly of the orbit at epoch, in radians.
    /// - `mu`: The gravitational parameter of the parent body, in m^3 s^-2.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let radius = 4.2;
    /// let mean_anomaly_at_epoch = 1.5;
    /// let gravitational_parameter = 5.0;
    ///
    /// let orbit = Orbit2D::new_circular(
    ///     radius,
    ///     mean_anomaly_at_epoch,
    ///     gravitational_parameter,
    /// );
    ///
    /// assert_eq!(orbit.get_eccentricity(), 0.0);
    /// assert_eq!(orbit.get_periapsis(), radius);
    /// assert_eq!(orbit.get_arg_pe(), 0.0);
    /// assert_eq!(orbit.get_mean_anomaly_at_epoch(), mean_anomaly_at_epoch);
    /// assert_eq!(orbit.get_gravitational_parameter(), gravitational_parameter);
    /// ```
    pub fn new_circular(radius: f64, mean_anomaly: f64, mu: f64) -> Self {
        let matrix = DMat2::IDENTITY;

        debug_assert_eq!(matrix, Self::get_transformation_matrix(0.0),);

        Self {
            eccentricity: 0.0,
            periapsis: radius,
            arg_pe: 0.0,
            mean_anomaly,
            mu,
            cache: OrbitCachedCalculations {
                transformation_matrix: matrix,
            },
        }
    }

    /// Updates the cached values in the orbit struct.
    ///
    /// Should only be called when the following things change:
    /// 1. Argument of periapsis
    fn update_cache(&mut self) {
        self.cache = Self::get_cached_calculations(self.arg_pe);
    }

    fn get_cached_calculations(arg_pe: f64) -> OrbitCachedCalculations {
        let transformation_matrix = Self::get_transformation_matrix(arg_pe);

        OrbitCachedCalculations {
            transformation_matrix,
        }
    }

    fn get_transformation_matrix(arg_pe: f64) -> DMat2 {
        let (sin_arg_pe, cos_arg_pe) = arg_pe.sin_cos();

        // From https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
        // matrix.e11 = cos_arg_pe * cos_lan - sin_arg_pe * cos_inc * sin_lan;
        // matrix.e12 = -(sin_arg_pe * cos_lan + cos_arg_pe * cos_inc * sin_lan);
        // matrix.e21 = cos_arg_pe * sin_lan + sin_arg_pe * cos_inc * cos_lan;
        // matrix.e22 = cos_arg_pe * cos_inc * cos_lan - sin_arg_pe * sin_lan;
        // matrix.e31 = sin_arg_pe * sin_inc;
        // matrix.e32 = cos_arg_pe * sin_inc;
        //
        // Simplifying with inc, lan = 0:
        // matrix.e11 = cos_arg_pe;
        // matrix.e12 = -sin_arg_pe;
        // matrix.e21 = sin_arg_pe;
        // matrix.e22 = cos_arg_pe;

        DMat2 {
            x_axis: DVec2::new(cos_arg_pe, -sin_arg_pe),
            y_axis: DVec2::new(sin_arg_pe, cos_arg_pe),
        }
    }
}

impl OrbitTrait2D for Orbit2D {
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
            self.update_cache();
        }

        if apoapsis < 0.0 && apoapsis > -self.periapsis {
            // Even for hyperbolic orbits, apoapsis cannot be between 0 and -periapsis
            // We will interpret this as an infinite apoapsis (parabolic trajectory)
            self.eccentricity = 1.0;
        } else {
            self.eccentricity = (apoapsis - self.periapsis) / (apoapsis + self.periapsis);
        }
    }

    #[inline]
    fn get_transformation_matrix(&self) -> DMat2 {
        self.cache.transformation_matrix
    }

    #[inline]
    fn get_pqw_basis_vector_p(&self) -> DVec2 {
        self.cache.transformation_matrix.x_axis
    }

    #[inline]
    fn get_pqw_basis_vector_q(&self) -> DVec2 {
        self.cache.transformation_matrix.y_axis
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
    fn get_arg_pe(&self) -> f64 {
        self.arg_pe
    }

    #[inline]
    fn get_mean_anomaly_at_epoch(&self) -> f64 {
        self.mean_anomaly
    }

    #[inline]
    fn get_gravitational_parameter(&self) -> f64 {
        self.mu
    }

    #[inline]
    fn set_eccentricity(&mut self, value: f64) {
        self.eccentricity = value;
    }

    #[inline]
    fn set_periapsis(&mut self, value: f64) {
        self.periapsis = value;
    }

    fn set_arg_pe(&mut self, value: f64) {
        self.arg_pe = value;
        self.update_cache();
    }

    #[inline]
    fn set_mean_anomaly_at_epoch(&mut self, value: f64) {
        self.mean_anomaly = value;
    }

    fn set_gravitational_parameter(&mut self, gravitational_parameter: f64, mode: MuSetterMode2D) {
        let new_mu = gravitational_parameter;
        match mode {
            MuSetterMode2D::KeepElements => {
                self.mu = new_mu;
            }
            MuSetterMode2D::KeepPositionAtTime(t) => {
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
            MuSetterMode2D::KeepKnownStateVectors {
                state_vectors,
                time,
            } => {
                let new = state_vectors.to_cached_orbit(new_mu, time);
                *self = new;
            }
            MuSetterMode2D::KeepStateVectorsAtTime(time) => {
                let ecc_anom = self.get_eccentric_anomaly_at_time(time);
                let state_vectors = self.get_state_vectors_at_eccentric_anomaly(ecc_anom);
                let new = state_vectors.to_cached_orbit(new_mu, time);
                *self = new;
            }
        }
    }
}

impl From<CompactOrbit2D> for Orbit2D {
    fn from(compact: CompactOrbit2D) -> Self {
        Self::new(
            compact.eccentricity,
            compact.periapsis,
            compact.arg_pe,
            compact.mean_anomaly,
            compact.mu,
        )
    }
}

impl Default for Orbit2D {
    /// Creates a unit orbit.
    ///
    /// The unit orbit is a perfect circle of radius 1 and
    /// zero mean anomaly at epoch.
    ///
    /// It also uses a gravitational parameter of 1.
    fn default() -> Self {
        Self::new_circular(1.0, 0.0, 1.0)
    }
}
