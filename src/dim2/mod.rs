use core::f64::consts::{PI, TAU};

use glam::{DMat2, DVec2};

pub mod cached_orbit;
pub mod compact_orbit;

#[cfg(feature = "libm")]
#[allow(unused_imports)]
use crate::math::F64Math;
use crate::{sinhcosh, solvers, ApoapsisSetterError};
pub use cached_orbit::Orbit2D;
pub use compact_orbit::CompactOrbit2D;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A trait that defines the methods that a 2D-constrained Keplerian
/// orbit must implement.
///
/// This trait is implemented by both [`Orbit2D`] and [`CompactOrbit2D`].
///
/// # Examples
/// ```
/// use keplerian_sim::{Orbit2D, OrbitTrait2D, CompactOrbit2D};
///
/// fn accepts_orbit(orbit: &impl OrbitTrait2D) {
///     println!("That's an orbit!");
/// }
///
/// fn main() {
///     let orbit = Orbit2D::default();
///     accepts_orbit(&orbit);
///
///     let compact = CompactOrbit2D::default();
///     accepts_orbit(&compact);
/// }
/// ```
///
/// This example will fail to compile:
///
/// ```compile_fail
/// # use keplerian_sim::{Orbit2D, OrbitTrait2D, CompactOrbit2D};
/// #
/// # fn accepts_orbit(orbit: &impl OrbitTrait2D) {
/// #     println!("That's an orbit!");
/// # }
/// #
/// # fn main() {
/// #     let orbit = Orbit2D::default();
/// #     accepts_orbit(&orbit);
/// #  
/// #     let compact = CompactOrbit2D::default();
/// #     accepts_orbit(&compact);
///       let not_orbit = (0.0, 1.0);
///       accepts_orbit(&not_orbit);
/// # }
pub trait OrbitTrait2D {
    /// Gets the semi-major axis of the orbit.
    ///
    /// In an elliptic orbit, the semi-major axis is the
    /// average of the apoapsis and periapsis.  
    /// This function uses a generalization which uses
    /// eccentricity instead.
    ///
    /// This function returns infinity for parabolic orbits,
    /// and negative values for hyperbolic orbits.
    ///
    /// Learn more: <https://en.wikipedia.org/wiki/Semi-major_and_semi-minor_axes>
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(50.0);
    /// orbit.set_apoapsis_force(100.0);
    /// let sma = orbit.get_semi_major_axis();
    /// let expected = 75.0;
    /// assert!((sma - expected).abs() < 1e-6);
    /// ```
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    fn get_semi_major_axis(&self) -> f64 {
        // a = r_p / (1 - e)
        self.get_periapsis() / (1.0 - self.get_eccentricity())
    }

    /// Gets the semi-minor axis of the orbit.
    ///
    /// In an elliptic orbit, the semi-minor axis is half of the maximum "width"
    /// of the orbit.
    ///
    /// Learn more: <https://en.wikipedia.org/wiki/Semi-major_and_semi-minor_axes>
    ///
    /// # Performance
    /// This function is performant and is unlikely to be the cause of any
    /// performance issues.
    fn get_semi_minor_axis(&self) -> f64 {
        self.get_semi_major_axis() * (1.0 - self.get_eccentricity().powi(2)).abs().sqrt()
    }

    /// Gets the semi-latus rectum of the orbit, in meters.
    ///
    /// Learn more: <https://en.wikipedia.org/wiki/Ellipse#Semi-latus_rectum>  
    /// <https://en.wikipedia.org/wiki/Conic_section#Conic_parameters>
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    fn get_semi_latus_rectum(&self) -> f64 {
        // https://control.asu.edu/Classes/MAE462/462Lecture03.pdf
        // Lecture on Spacecraft Dynamics and Control
        // by Matthew M. Peet
        // Slide 20 (or 12?), section titled "Periapse for all Orbits"
        // r_p = p / (1 + e)
        // ...where r_p = periapsis; p = semi-latus rectum; e = eccentricity.
        // => r_p (1 + e) = p
        self.get_periapsis() * (1.0 + self.get_eccentricity())
    }

    /// Gets the linear eccentricity of the orbit, in meters.
    ///
    /// In an elliptic orbit, the linear eccentricity is the distance
    /// between its center and either of its two foci (focuses).
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(50.0);
    /// orbit.set_apoapsis_force(100.0);
    ///
    /// // Let's say the periapsis is at x = -50.
    /// // The apoapsis would be at x = 100.
    /// // The midpoint would be at x = 25.
    /// // The parent body - one of its foci - is always at the origin (x = 0).
    /// // This means the linear eccentricity is 25.
    ///
    /// let linear_eccentricity = orbit.get_linear_eccentricity();
    /// let expected = 25.0;
    ///
    /// assert!((linear_eccentricity - expected).abs() < 1e-6);
    /// ```
    fn get_linear_eccentricity(&self) -> f64 {
        self.get_semi_major_axis() - self.get_periapsis()
    }

    /// Gets the true anomaly asymptote (f_∞) of the hyperbolic trajectory.
    ///
    /// This returns a positive number between π/2 and π for open
    /// trajectories, and NaN for closed orbits.
    ///
    /// This can be used to get the range of possible true anomalies that
    /// a hyperbolic trajectory can be in.  
    /// This function returns the maximum true anomaly, and the minimum
    /// true anomaly can be derived simply by negating the result:
    /// ```text
    /// f_-∞ = -f_∞
    /// ```
    /// The minimum and maximum together represent the range of possible
    /// true anomalies.
    ///
    /// # Performance
    /// This function is moderately performant and contains only one
    /// trigonometry operation.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// // Closed (elliptic) orbit with eccentricity = 0.8
    /// let closed = Orbit2D::new(0.8, 1.0, 0.0, 0.0, 1.0);
    ///
    /// // True anomaly asymptote is only defined for open orbits,
    /// // i.e., eccentricity ≥ 1
    /// assert!(closed.get_true_anomaly_at_asymptote().is_nan());
    ///
    /// let parabolic = Orbit2D::new(1.0, 1.0, 0.0, 0.0, 1.0);
    /// assert_eq!(
    ///     parabolic.get_true_anomaly_at_asymptote(),
    ///     std::f64::consts::PI
    /// );
    ///
    /// let hyperbolic = Orbit2D::new(2.0, 1.0, 0.0, 0.0, 1.0);
    /// let asymptote = 2.0943951023931957;
    /// assert_eq!(
    ///     hyperbolic.get_true_anomaly_at_asymptote(),
    ///     asymptote
    /// );
    ///
    /// // At the asymptote, the altitude is infinite.
    /// // Note: We can't use the regular `get_altitude_at_true_anomaly` here
    /// // because it is less accurate (since it uses cos() while the asymptote uses
    /// // acos(), and the roundtrip causes precision loss).
    /// // We use the unchecked version with the exact cosine value
    /// // of the true anomaly (-1/e) to avoid float inaccuracies.
    /// let asymptote_cos = -1.0 / hyperbolic.get_eccentricity();
    ///
    /// // We first check that asymptote_cos is close to cos(asymptote):
    /// assert!(
    ///     (asymptote_cos - asymptote.cos()).abs() < 1e-15
    /// );
    ///
    /// // Then we can be fairly confident this will be exactly infinite:
    /// assert!(
    ///     hyperbolic
    ///         .get_altitude_at_true_anomaly_unchecked(
    ///             hyperbolic.get_semi_latus_rectum(),
    ///             asymptote_cos
    ///         )
    ///         .is_infinite()
    /// )
    /// ```
    #[doc(alias = "get_theta_infinity")]
    #[doc(alias = "get_hyperbolic_true_anomaly_range")]
    fn get_true_anomaly_at_asymptote(&self) -> f64 {
        // https://en.wikipedia.org/wiki/Hyperbolic_trajectory#Parameters_describing_a_hyperbolic_trajectory
        // 2f_∞ = 2cos^-1(-1/e)
        // ⇒ f_∞ = acos(-1/e)
        use core::ops::Neg;
        self.get_eccentricity().recip().neg().acos()
    }

    /// Gets the position of the orbit at periapsis.
    ///
    /// # Performance
    /// In the cached orbit struct ([`Orbit2D`]), this function is
    /// very performant and only involves three multiplications.
    ///
    /// However, in the compact orbit struct ([`CompactOrbit2D`]), this
    /// is a lot slower and involves some trigonometric calculations.
    /// If you already know the P basis vector of the PQW coordinate system,
    /// you may use the unchecked version instead
    /// ([`get_position_at_periapsis_unchecked`][OrbitTrait2D::get_position_at_periapsis_unchecked]),
    /// which is a lot faster and would skip repeated calculations.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// let orbit = Orbit2D::new(
    ///     0.25, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Argument of periapsis
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter
    /// );
    ///
    /// assert_eq!(
    ///     orbit.get_position_at_periapsis(),
    ///     DVec2::new(1.0, 0.0)
    /// );
    /// ```
    #[inline]
    fn get_position_at_periapsis(&self) -> DVec2 {
        self.get_position_at_periapsis_unchecked(self.get_pqw_basis_vector_p())
    }

    /// Gets the position of the orbit at periapsis
    /// based on a known P basis vector.
    ///
    /// The P basis vector is one of the basis vector from the PQW
    /// coordinate system.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// # Unchecked Operation
    /// This function does not check that the given `p_vector`
    /// is a unit vector nor whether it actually is the basis vector in the
    /// PQW coordinate system.
    ///
    /// It is expected that callers get this vector from either
    /// the transformation matrix
    /// ([`get_transformation_matrix`][OrbitTrait2D::get_transformation_matrix]),
    /// the basis vector collective getter
    /// ([`get_pqw_basis_vectors`][OrbitTrait2D::get_pqw_basis_vectors]),
    /// or the individual basis vector getter
    /// ([`get_pqw_basis_vector_p`][OrbitTrait2D::get_pqw_basis_vector_p]).
    ///
    /// A safe wrapper is available, but that may be slower; see the
    /// Performance section for details.
    ///
    /// # Performance
    /// There is no reason to use this if you are using the cached
    /// orbit struct ([`Orbit2D`]) as the performance is identical to the
    /// wrapper function.
    ///
    /// However, in the compact orbit struct ([`CompactOrbit2D`]), this
    /// skips some expensive trigonometry operations and therefore is
    /// a lot faster than the wrapper function.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{CompactOrbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// let orbit = CompactOrbit2D::new(
    ///     0.25, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Argument of periapsis
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter
    /// );
    ///
    /// let p_vector = orbit.get_pqw_basis_vector_p();
    ///
    /// // Use p_vector here...
    /// // Use it again for periapsis position!
    ///
    /// assert_eq!(
    ///     orbit.get_position_at_periapsis_unchecked(p_vector),
    ///     DVec2::new(1.0, 0.0)
    /// );
    /// ```
    #[inline]
    fn get_position_at_periapsis_unchecked(&self, p_vector: DVec2) -> DVec2 {
        self.get_periapsis() * p_vector
    }

    /// Gets the position of the orbit at apoapsis.
    ///
    /// # Performance
    /// In the cached orbit struct ([`Orbit2D`]), this function is
    /// very performant and only involves three multiplications.
    ///
    /// However, in the compact orbit struct ([`CompactOrbit2D`]), this
    /// is a lot slower and involves some trigonometric calculations.
    /// If you already know the P basis vector of the PQW coordinate system,
    /// you may use the unchecked version instead
    /// ([`get_position_at_apoapsis_unchecked`][OrbitTrait2D::get_position_at_apoapsis_unchecked]),
    /// which is a lot faster and would skip repeated calculations.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// let orbit = Orbit2D::new(
    ///     0.25, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Argument of periapsis
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter
    /// );
    ///
    /// assert_eq!(
    ///     orbit.get_position_at_apoapsis(),
    ///     DVec2::new(-1.6666666666666665, 0.0)
    /// );
    /// ```
    #[inline]
    fn get_position_at_apoapsis(&self) -> DVec2 {
        self.get_position_at_apoapsis_unchecked(self.get_pqw_basis_vector_p())
    }

    /// Gets the position of the orbit at apoapsis
    /// based on a known P basis vector.
    ///
    /// The P basis vector is one of the basis vector from the PQW
    /// coordinate system.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// # Unchecked Operation
    /// This function does not check that the given `p_vector`
    /// is a unit vector nor whether it actually is the basis vector in the
    /// PQW coordinate system.
    ///
    /// It is expected that callers get this vector from either
    /// the transformation matrix
    /// ([`get_transformation_matrix`][OrbitTrait2D::get_transformation_matrix]),
    /// the basis vector collective getter
    /// ([`get_pqw_basis_vectors`][OrbitTrait2D::get_pqw_basis_vectors]),
    /// or the individual basis vector getter
    /// ([`get_pqw_basis_vector_p`][OrbitTrait2D::get_pqw_basis_vector_p]).
    ///
    /// A safe wrapper is available, but that may be slower; see the
    /// Performance section for details.
    ///
    /// # Performance
    /// There is no reason to use this if you are using the cached
    /// orbit struct ([`Orbit2D`]) as the performance is identical to the
    /// wrapper function.
    ///
    /// However, in the compact orbit struct ([`CompactOrbit2D`]), this
    /// skips some expensive trigonometry operations and therefore is
    /// a lot faster than the wrapper function.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{CompactOrbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// let orbit = CompactOrbit2D::new(
    ///     0.25, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Argument of periapsis
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter
    /// );
    ///
    /// let p_vector = orbit.get_pqw_basis_vector_p();
    ///
    /// // Use p_vector here...
    /// // Use it again for apoapsis position!
    ///
    /// assert_eq!(
    ///     orbit.get_position_at_apoapsis_unchecked(p_vector),
    ///     DVec2::new(-1.6666666666666665, 0.0)
    /// );
    /// ```
    #[inline]
    fn get_position_at_apoapsis_unchecked(&self, p_vector: DVec2) -> DVec2 {
        -self.get_apoapsis() * p_vector
    }

    /// Gets the apoapsis of the orbit.  
    /// Returns infinity for parabolic orbits.  
    /// Returns negative values for hyperbolic orbits.  
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    ///
    /// # Examples
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_eccentricity(0.5); // Elliptic
    /// assert!(orbit.get_apoapsis() > 0.0);
    ///
    /// orbit.set_eccentricity(1.0); // Parabolic
    /// assert!(orbit.get_apoapsis().is_infinite());
    ///
    /// orbit.set_eccentricity(2.0); // Hyperbolic
    /// assert!(orbit.get_apoapsis() < 0.0);
    /// ```
    fn get_apoapsis(&self) -> f64 {
        let eccentricity = self.get_eccentricity();
        if eccentricity == 1.0 {
            f64::INFINITY
        } else {
            self.get_semi_major_axis() * (1.0 + eccentricity)
        }
    }

    /// Sets the apoapsis of the orbit.  
    /// Errors when the apoapsis is less than the periapsis, or less than zero.  
    /// If you want a setter that does not error, use `set_apoapsis_force`, which will
    /// try its best to interpret what you might have meant, but may have
    /// undesirable behavior.
    ///
    /// # Examples
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(50.0);
    ///
    /// assert!(
    ///     orbit.set_apoapsis(100.0)
    ///         .is_ok()
    /// );
    ///
    /// let result = orbit.set_apoapsis(25.0);
    /// assert!(result.is_err());
    /// assert_eq!(
    ///     result.unwrap_err(),
    ///     keplerian_sim::ApoapsisSetterError::ApoapsisLessThanPeriapsis
    /// );
    ///
    /// let result = orbit.set_apoapsis(-25.0);
    /// assert!(result.is_err());
    /// assert_eq!(
    ///     result.unwrap_err(),
    ///     keplerian_sim::ApoapsisSetterError::ApoapsisNegative
    /// );
    /// ```
    fn set_apoapsis(&mut self, apoapsis: f64) -> Result<(), ApoapsisSetterError>;

    /// Sets the apoapsis of the orbit, with a best-effort attempt at interpreting
    /// possibly-invalid values.  
    /// This function will not error, but may have undesirable behavior:
    /// - If the given apoapsis is less than the periapsis but more than zero,
    ///   the orbit will be flipped and the periapsis will be set to the given apoapsis.
    /// - If the given apoapsis is negative but between zero and negative periapsis,
    ///   the apoapsis will get treated as infinity and the orbit will be parabolic.
    ///   (This is because even in hyperbolic orbits, apoapsis cannot be between 0 and -periapsis)  
    /// - If the given apoapsis is negative AND less than negative periapsis,
    ///   the orbit will be hyperbolic.
    ///
    /// If these behaviors are undesirable, consider creating a custom wrapper around
    /// `set_eccentricity` instead.
    ///
    /// # Examples
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut base = Orbit2D::default();
    /// base.set_periapsis(50.0);
    ///
    /// let mut normal = base.clone();
    /// // Set the apoapsis to 100
    /// normal.set_apoapsis_force(100.0);
    /// assert_eq!(normal.get_apoapsis(), 99.99999999999997);
    /// assert_eq!(normal.get_periapsis(), 50.0);
    /// assert_eq!(normal.get_arg_pe(), 0.0);
    /// assert_eq!(normal.get_mean_anomaly_at_epoch(), 0.0);
    ///
    /// let mut flipped = base.clone();
    /// // Set the "apoapsis" to 25
    /// // This will flip the orbit, setting the altitude
    /// // where the current apoapsis is, to 25, and
    /// // flipping the orbit.
    /// // This sets the periapsis to 25, and the apoapsis to the
    /// // previous periapsis.
    /// flipped.set_apoapsis_force(25.0);
    /// assert_eq!(flipped.get_apoapsis(), 49.999999999999986);
    /// assert_eq!(flipped.get_periapsis(), 25.0);
    /// assert_eq!(flipped.get_arg_pe(), std::f64::consts::PI);
    /// assert_eq!(flipped.get_mean_anomaly_at_epoch(), std::f64::consts::PI);
    ///
    /// let mut hyperbolic = base.clone();
    /// // Set the "apoapsis" to -250
    /// hyperbolic.set_apoapsis_force(-250.0);
    /// assert_eq!(hyperbolic.get_apoapsis(), -250.0);
    /// assert_eq!(hyperbolic.get_periapsis(), 50.0);
    /// assert_eq!(hyperbolic.get_arg_pe(), 0.0);
    /// assert!(hyperbolic.get_eccentricity() > 1.0);
    /// assert_eq!(hyperbolic.get_mean_anomaly_at_epoch(), 0.0);
    ///
    /// let mut parabolic = base.clone();
    /// // Set the "apoapsis" to between 0 and -50
    /// // This will set the apoapsis to infinity, and the orbit will be parabolic.
    /// parabolic.set_apoapsis_force(-25.0);
    /// assert!(parabolic.get_apoapsis().is_infinite());
    /// assert_eq!(parabolic.get_periapsis(), 50.0);
    /// assert_eq!(parabolic.get_arg_pe(), 0.0);
    /// assert_eq!(parabolic.get_eccentricity(), 1.0);
    /// assert_eq!(parabolic.get_mean_anomaly_at_epoch(), 0.0);
    /// ```
    fn set_apoapsis_force(&mut self, apoapsis: f64);

    /// Gets the mean motion of the orbit, in radians per second.
    ///
    /// Mean motion (represented by `n`) is the angular speed required for
    /// a body to complete one orbit, assuming constant speed in a circular
    /// orbit which completes in the same time as the variable speed,
    /// elliptical orbit of the actual body.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Mean_motion)
    ///
    /// # Performance
    /// This function is performant and is unlikely to be the cause
    /// of any performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let orbit = Orbit2D::new(
    ///     0.8, // Eccentricity
    ///     9.4, // Periapsis
    ///     2.0, // Argument of periapsis
    ///     1.5, // Mean anomaly at epoch
    ///     0.8, // Gravitational parameter
    /// );
    ///
    /// let orbital_period = orbit.get_orbital_period();
    /// let mean_motion = std::f64::consts::TAU / orbital_period;
    ///
    /// assert!((orbit.get_mean_motion() - mean_motion).abs() < f64::EPSILON);
    /// ```
    fn get_mean_motion(&self) -> f64 {
        // n = TAU / T, radians
        //
        // note: T = 2pi * sqrt(a^3 / GM)
        //
        // => n = TAU / (TAU * sqrt(a^3 / GM))
        // => n = 1 / sqrt(a^3 / GM)
        // => n = sqrt(GM / a^3)
        (self.get_gravitational_parameter() / self.get_semi_major_axis().powi(3)).sqrt()
    }

    /// Gets the focal parameter of the orbit, in meters.
    ///
    /// This returns infinity in circular orbits (e = 0).
    ///
    /// The focal parameter (p) is the distance from a focus
    /// to the corresponding directrix.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Conic_section#Conic_parameters)
    ///
    /// # Performance
    /// This function is very performant and should not be
    /// the cause of any performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let orbit = Orbit2D::new(
    ///     1.0, // Eccentricity
    ///     3.0, // Periapsis
    ///     2.9, // Argument of periapsis
    ///     0.8, // Mean anomaly at epoch
    ///     1.9, // Gravitational parameter
    /// );
    ///
    /// // From Wikipedia's focal parameter equation for parabolas (e = 1)
    /// // (a in this case is periapsis distance, not semi-major axis)
    /// let expected_focal_parameter = 2.0 * orbit.get_periapsis();
    ///
    /// assert!(
    ///     (orbit.get_focal_parameter() - expected_focal_parameter).abs() < f64::EPSILON
    /// );
    /// ```
    fn get_focal_parameter(&self) -> f64 {
        // https://web.ma.utexas.edu/users/m408s/m408d/CurrentWeb/LM10-6-3.php
        //
        // > Polar equations of conic sections: If the directrix is a distance d away,
        // > then the polar form of a conic section with eccentricity e is
        // >
        // > r(θ) = ed / (1 - e cos (θ - θ_0))
        // >
        // > where the constant θ_0 depends on the direction of the directrix.
        // >
        // > If the directrix is the line x = d, then we have
        // > r = ed / (1 + e cos θ)
        //
        // Using periapsis (r = r_p, θ = ν = 0):
        // r_p = ed / (1 + e cos 0)
        // r_p = ed / (1 + e), since cos 0 = 1
        // 1 = ed / (r_p * (1 + e))
        // 1/d = e / (r_p * (1 + e))
        // d = r_p * (1 + e) / e
        self.get_periapsis() * (1.0 + self.get_eccentricity()) / self.get_eccentricity()
    }

    /// Gets the specific angular momentum of the orbit, in
    /// square meters per second (m^2/s).
    ///
    /// The specific relative angular momentum of a body is the
    /// angular momentum of that body divided by its mass.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Specific_angular_momentum)
    ///
    /// # Performance
    /// This function is performant and is unlikely to be the cause
    /// of any performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let orbit = Orbit2D::new(
    ///     1.2, // Eccentricity
    ///     2.0, // Periapsis
    ///     3.0, // Argument of periapsis
    ///     4.8, // Mean anomaly at epoch
    ///     5.0, // Gravitational parameter
    /// );
    ///
    /// const EXPECTED_VALUE: f64 = 4.69041575982343;
    ///
    /// let momentum =
    ///     orbit.get_specific_angular_momentum();
    ///
    /// assert!((momentum - EXPECTED_VALUE).abs() < f64::EPSILON);
    /// ```
    fn get_specific_angular_momentum(&self) -> f64 {
        // https://faculty.fiu.edu/~vanhamme/ast3213/orbits.pdf
        // Page 3, eq. 17: "h = sqrt(μa(1 - e^2))"
        //
        // to simplify:
        // inner := a(1 - e^2)
        // => h = sqrt(μ * inner)
        //
        // recall a = r_p / (1 - e)
        //
        // => inner = (r_p / (1 - e))(1 - e^2)
        // => inner = (r_p / (1 - e))(1 - e)(1 + e)
        // => inner = r_p (1 + e)
        // => inner = semi-latus rectum `p`
        // (see [OrbitTrait2D::get_semi_latus_rectum])
        // => h = sqrt(μp)
        (self.get_gravitational_parameter() * self.get_semi_latus_rectum()).sqrt()
    }

    /// Gets the specific orbital energy `ε` of the orbit,
    /// in joules per kilogram (J/kg, equiv. to m^2 ⋅ s^-2).
    ///
    /// For closed orbits (eccentricity < 0), ε < 0.  
    /// When eccentricity equals 1 (parabolic), ε equals 0,
    /// and when eccentricity exceeds 1 (hyperbolic), ε is positive.
    ///
    /// The specific orbital energy ε of two orbiting bodies is
    /// the constant quotient of their mechanical energy
    /// (the sum of their mutual potential energy, ε_p, and their
    /// kinetic energy, ε_k) to their reduced mass.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Specific_orbital_energy)
    ///
    /// # Performance
    /// This function is very performant and should not be the
    /// cause of any performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let elliptic = Orbit2D::new(0.3, 1.0, 0.0, 0.0, 1.0);
    /// let parabolic = Orbit2D::new(1.0, 1.0, 0.0, 0.0, 1.0);
    /// let hyperbolic = Orbit2D::new(2.6, 1.0, 0.0, 0.0, 1.0);
    ///
    /// assert!(elliptic.get_specific_orbital_energy() < 0.0);
    /// assert!(parabolic.get_specific_orbital_energy() == 0.0);
    /// assert!(hyperbolic.get_specific_orbital_energy() > 0.0);
    /// ```
    fn get_specific_orbital_energy(&self) -> f64 {
        // https://en.wikipedia.org/wiki/Specific_orbital_energy
        // ε = -μ / (2a)
        //
        // note: a = r_p / (1 - e)
        // => 1 / a = (1 - e) / r_p
        //
        // => ε = -μ (1 - e) / r_p
        -self.get_gravitational_parameter() * (1.0 - self.get_eccentricity()) / self.get_periapsis()
    }

    /// Gets the area swept out by the orbit in square meters
    /// per second (m^2/s).
    ///
    /// # Performance
    /// This function is very performant and should not
    /// be the cause of any performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let orbit = Orbit2D::new(
    ///     2.9, // Eccentricity
    ///     4.9, // Periapsis
    ///     0.2, // Argument of periapsis
    ///     0.9, // Mean anomaly at epoch
    ///     9.8, // Gravitational parameter
    /// );
    ///
    /// const EXPECTED_RATE: f64 = 6.842477621446782;
    ///
    /// assert!(
    ///     (orbit.get_area_sweep_rate() - EXPECTED_RATE).abs() < f64::EPSILON
    /// );
    /// ```
    fn get_area_sweep_rate(&self) -> f64 {
        // We can measure the instantaneous area sweep rate
        // at periapsis.
        // As the delta-time gets smaller and smaller,
        // a triangle approximation gets more and more accurate.
        // (This is basically a derivative in calculus)
        // That approximation triangle has a length equal to
        // the speed at periapsis, and a height equal to the
        // radius/altitude at periapsis.
        // This means the triangle has an area of (1/2 * base * height)
        // = 1/2 * speed_at_periapsis * periapsis
        //
        // This would be the area sweep rate at the periapsis
        // of the orbit. We can then utilize Kepler's 2nd law:
        //
        //     A line segment joining a planet and the Sun
        //     sweeps out equal areas during equal intervals of time.
        //
        // (from https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion)
        //
        // This means this actually is a constant throughout
        // the orbit, so the equation below is valid.

        0.5 * self.get_speed_at_periapsis() * self.get_periapsis()
    }

    // TODO: PARABOLIC SUPPORT: This function returns NaN on parabolic
    /// Gets the time when the orbit is in periapsis, in seconds since epoch.
    ///
    /// This returns the time when mean anomaly equals zero.  
    /// This means although it will represent a time of periapsis,
    /// it doesn't mean "next periapsis" nor "previous periapsis",
    /// it just means "a periapsis", at least for closed orbits
    /// (e < 1).
    ///
    /// # Parabolic Support
    /// This function does not support parabolic trajectories yet.  
    /// Calling this function on a parabolic trajectory results in a
    /// non-finite number.
    ///
    /// # Performance
    /// This function is performant and is unlikely to be the cause
    /// of any performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// const PERIAPSIS: f64 = 1.0;
    ///
    /// let orbit = Orbit2D::new(
    ///     0.3, // Eccentricity
    ///     PERIAPSIS,
    ///     2.9, // Argument of periapsis
    ///     1.5, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter
    /// );
    ///
    /// let time_of_pe = orbit.get_time_of_periapsis();
    ///
    /// let alt_of_pe = orbit.get_altitude_at_time(time_of_pe);
    ///
    /// assert!(
    ///     (alt_of_pe - PERIAPSIS).abs() < 1e-15
    /// );
    /// ```
    fn get_time_of_periapsis(&self) -> f64 {
        // We want to find M = 0
        // Per `get_mean_anomaly_at_time`:
        // M = t * sqrt(mu / |a^3|) + M_0
        // => 0 = t * sqrt(mu / |a^3|) + M_0
        // => -M_0 = t * sqrt(mu / |a^3|)
        // => t * sqrt(mu / |a^3|) = -M_0
        // => t = -M_0 / sqrt(mu / |a^3|)
        //
        // note: 1 / sqrt(mu / |a^3|) = sqrt(|a^3| / mu)
        //
        // => t = -M_0 * sqrt(|a^3| / mu)

        -self.get_mean_anomaly_at_epoch()
            * (self.get_semi_major_axis().powi(3).abs() / self.get_gravitational_parameter()).sqrt()
    }

    /// Gets the time when the orbit is in apoapsis, in seconds since epoch.
    ///
    /// This returns the time when mean anomaly equals pi.  
    /// This means although it will represent a time of apoapsis,
    /// it doesn't mean "next apoapsis" nor "previous apoapsis",
    /// it just means "an apoapsis", at least for closed orbits
    /// (e < 1).
    ///
    /// # Performance
    /// This function is performant and is unlikely to be the cause
    /// of any performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// const APOAPSIS: f64 = 2.0;
    /// const PERIAPSIS: f64 = 1.0;
    ///
    /// let orbit = Orbit2D::with_apoapsis(
    ///     APOAPSIS,
    ///     PERIAPSIS,
    ///     2.9, // Argument of periapsis
    ///     1.5, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter
    /// );
    ///
    /// let time_of_ap = orbit.get_time_of_apoapsis();
    ///
    /// let alt_of_ap = orbit.get_altitude_at_time(time_of_ap);
    ///
    /// assert!(
    ///     (alt_of_ap - APOAPSIS).abs() < 1e-15
    /// );
    /// ```
    fn get_time_of_apoapsis(&self) -> f64 {
        // We want to find M = pi
        // Per `get_mean_anomaly_at_time`:
        // M = t * sqrt(mu / |a^3|) + M_0
        // => pi = t * sqrt(mu / |a^3|) + M_0
        // => pi - M_0 = t * sqrt(mu / |a^3|)
        // => t * sqrt(mu / |a^3|) = pi - M_0
        // => t = (pi - M_0) / sqrt(mu / |a^3|)
        //
        // note: 1 / sqrt(mu / |a^3|) = sqrt(|a^3| / mu)
        //
        // => t = (pi - M_0) * sqrt(|a^3| / mu)
        (PI - self.get_mean_anomaly_at_epoch())
            * (self.get_semi_major_axis().powi(3) / self.get_gravitational_parameter()).sqrt()
    }

    /// Gets the transformation matrix needed to tilt a 2D vector into the
    /// tilted orbital plane.
    ///
    /// # Performance
    /// For [`CompactOrbit2D`], this will perform a few trigonometric operations.  
    /// If you need this value often, consider using [the cached orbit struct][crate::Orbit2D] instead.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    /// use glam::{DVec2, DMat2};
    ///
    /// let orbit = Orbit2D::default();
    /// let matrix = orbit.get_transformation_matrix();
    ///
    /// assert_eq!(matrix, DMat2 {
    ///     x_axis: DVec2::new(1.0, 0.0),
    ///     y_axis: DVec2::new(0.0, 1.0),
    /// });
    /// ```
    fn get_transformation_matrix(&self) -> DMat2;

    /// Gets the basis vectors for the perifocal coordinate (PQW)
    /// system.
    ///
    /// # Output
    /// This function returns a tuple of two vectors. The vectors
    /// are the p and q basis vectors, respectively.
    ///
    /// The p basis vector is a unit vector that points to the periapsis.  
    /// The q basis vector is orthogonal to that and points 90° counterclockwise
    /// from the periapsis on the orbital plane.  
    /// The w basis vector is orthogonal to the XY plane and is
    /// excluded from this function's output.
    ///
    /// For more information about the PQW system, visit the
    /// [Wikipedia article](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Performance
    /// For [`CompactOrbit2D`], this will perform a few trigonometric operations
    /// and therefore is not too performant.  
    ///
    /// For [`Orbit2D`], this will only need to access the cache, and
    /// therefore is much more performant.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// let orbit = Orbit2D::default();
    /// let (p, q) = orbit.get_pqw_basis_vectors();
    ///
    /// assert_eq!(p, DVec2::X);
    /// assert_eq!(q, DVec2::Y);
    /// ```
    fn get_pqw_basis_vectors(&self) -> (DVec2, DVec2) {
        let mat = self.get_transformation_matrix();

        (mat.x_axis, mat.y_axis)
    }

    /// Gets the p basis vector for the perifocal coordinate (PQW)
    /// system.
    ///
    /// The p basis vector is a unit vector that points to the periapsis.  
    ///
    /// For more information about the PQW system, visit the
    /// [Wikipedia article](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Performance
    /// For [`CompactOrbit2D`], this will perform a few trigonometric operations
    /// and multiplications, and therefore is not too performant.  
    ///
    /// For [`Orbit2D`], this will only need to access the cache, and
    /// therefore is much more performant.
    ///
    /// If you want to get multiple basis vectors, use
    /// [`get_pqw_basis_vectors`][OrbitTrait2D::get_pqw_basis_vectors]
    /// instead, as that skips some duplicated work.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, CompactOrbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// let orbit = Orbit2D::default();
    /// let p = orbit.get_pqw_basis_vector_p();
    /// let q = orbit.get_pqw_basis_vector_q();
    ///
    /// assert_eq!(p, DVec2::X);
    /// assert_eq!(q, DVec2::Y);
    ///
    /// let compact_orbit = CompactOrbit2D::default();
    /// let p = compact_orbit.get_pqw_basis_vector_p();
    /// let q = compact_orbit.get_pqw_basis_vector_q();
    ///
    /// assert_eq!(p, DVec2::X);
    /// assert_eq!(q, DVec2::Y);
    /// ```
    fn get_pqw_basis_vector_p(&self) -> DVec2;

    /// Gets the q basis vector for the perifocal coordinate (PQW)
    /// system.
    ///
    /// The q basis vector is orthogonal to the p basis vector
    /// and points 90° counterclockwise from the periapsis on the
    /// orbital plane.  
    ///
    /// For more information about the PQW system, visit the
    /// [Wikipedia article](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Performance
    /// For [`CompactOrbit2D`], this will perform a few trigonometric operations
    /// and multiplications, and therefore is not too performant.  
    ///
    /// For [`Orbit2D`], this will only need to access the cache, and
    /// therefore is much more performant.
    ///
    /// If you want to get multiple basis vectors, use
    /// [`get_pqw_basis_vectors`][OrbitTrait2D::get_pqw_basis_vectors]
    /// instead, as that skips some duplicated work.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, CompactOrbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// let orbit = Orbit2D::default();
    /// let p = orbit.get_pqw_basis_vector_p();
    /// let q = orbit.get_pqw_basis_vector_q();
    ///
    /// assert_eq!(p, DVec2::X);
    /// assert_eq!(q, DVec2::Y);
    ///
    /// let compact_orbit = CompactOrbit2D::default();
    /// let p = compact_orbit.get_pqw_basis_vector_p();
    /// let q = compact_orbit.get_pqw_basis_vector_q();
    ///
    /// assert_eq!(p, DVec2::X);
    /// assert_eq!(q, DVec2::Y);
    /// ```
    fn get_pqw_basis_vector_q(&self) -> DVec2;

    /// Gets the eccentricity vector of this orbit.
    ///
    /// The eccentricity vector of a Kepler orbit is the dimensionless vector
    /// with direction pointing from apoapsis to periapsis and with magnitude
    /// equal to the orbit's scalar eccentricity.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Eccentricity_vector)
    ///
    /// # Performance
    /// This function is significantly faster in the cached version of the
    /// orbit struct ([`Orbit2D`]) than the compact version ([`CompactOrbit2D`]).  
    /// Consider using the cached version if this function will be called often.
    ///
    /// Alternatively, if you want to keep using the compact version and know
    /// the periapsis unit vector, use the unchecked version:
    /// [`get_eccentricity_vector_unchecked`][OrbitTrait2D::get_eccentricity_vector_unchecked]
    ///
    /// The cached version only needs to do a multiplication, and therefore is
    /// very performant.
    ///
    /// The compact version additionally has to compute many multiplications,
    /// additions, and several trig operations.
    /// // TODO: Check if this is also true for 2D orbits
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// // Parabolic orbit (e = 1)
    /// let orbit = Orbit2D::new(1.0, 1.0, 0.0, 0.0, 1.0);
    /// let eccentricity_vector = orbit.get_eccentricity_vector();
    ///
    /// assert_eq!(
    ///     eccentricity_vector,
    ///     DVec2::X
    /// );
    /// ```
    fn get_eccentricity_vector(&self) -> DVec2 {
        self.get_eccentricity_vector_unchecked(self.get_pqw_basis_vector_p())
    }

    /// Gets the eccentricity vector of this orbit.
    ///
    /// The eccentricity vector of a Kepler orbit is the dimensionless vector
    /// with direction pointing from apoapsis to periapsis and with magnitude
    /// equal to the orbit's scalar eccentricity.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Eccentricity_vector)
    ///
    /// # Unchecked Operation
    /// This function does not check for the validity of the given
    /// P basis vector. The given P vector should be of length 1.
    ///
    /// It is expected that callers get this basis vector
    /// from either the transformation matrix or the
    /// [`get_pqw_basis_vectors`][OrbitTrait2D::get_pqw_basis_vectors]
    /// function.
    ///
    /// If the given P vector is not of length 1, you may get
    /// nonsensical outputs.
    ///
    /// # Performance
    /// This function, by itself, is very performant, and should not be
    /// the cause of any performance problems.
    ///
    /// However, for the cached orbit struct ([`Orbit2D`]), this function
    /// has the same performance as the safer
    /// [`get_eccentricity_vector`][OrbitTrait2D::get_eccentricity_vector]
    /// function. There should be no need to use this function if you are
    /// using the cached orbit struct.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{CompactOrbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// // Parabolic orbit (e = 1)
    /// let orbit = CompactOrbit2D::new(1.0, 1.0, 0.0, 0.0, 1.0);
    ///
    /// // Expensive op for compact orbit: get basis vectors
    /// let basis_vectors = orbit.get_pqw_basis_vectors();
    ///
    /// // Use basis vectors for something...
    /// assert_eq!(
    ///     basis_vectors,
    ///     (DVec2::X, DVec2::Y)
    /// );
    ///
    /// // You can reuse it here! No need to recompute (as long as
    /// // orbit hasn't changed)
    /// let eccentricity_vector = orbit.get_eccentricity_vector_unchecked(
    ///     // basis vectors: P, Q, and W; we get the first one (0th index)
    ///     basis_vectors.0
    /// );
    ///
    /// assert_eq!(
    ///     eccentricity_vector,
    ///     DVec2::X
    /// );
    /// ```
    fn get_eccentricity_vector_unchecked(&self, p_vector: DVec2) -> DVec2 {
        self.get_eccentricity() * p_vector
    }

    // TODO: POST-PARABOLIC SUPPORT: Add note about parabolic eccentric anomaly (?), remove parabolic support sections
    /// Gets the eccentric anomaly at a given mean anomaly in the orbit.
    ///
    /// When the orbit is open (has an eccentricity of at least 1),
    /// the [hyperbolic eccentric anomaly](https://space.stackexchange.com/questions/27602/what-is-hyperbolic-eccentric-anomaly-f)
    /// would be returned instead.
    ///
    /// # Parabolic Support
    /// This function doesn't yet support parabolic trajectories. It may return `NaN`s
    /// or nonsensical values in this case.
    ///
    /// # Performance
    /// The method to get the eccentric anomaly from the mean anomaly
    /// uses numerical approach methods, and so it is not performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// The eccentric anomaly is an angular parameter that defines the position
    /// of a body that is moving along an elliptic Kepler orbit.
    ///
    /// — [Wikipedia](https://en.wikipedia.org/wiki/Eccentric_anomaly)
    fn get_eccentric_anomaly_at_mean_anomaly(&self, mean_anomaly: f64) -> f64 {
        // TODO: PARABOLIC SUPPORT: This function doesn't consider parabolic support yet.
        if self.is_closed() {
            self.get_elliptic_eccentric_anomaly(mean_anomaly)
        } else {
            self.get_hyperbolic_eccentric_anomaly(mean_anomaly)
        }
    }

    /// Get an initial guess for the hyperbolic eccentric anomaly of an orbit.
    ///
    /// # Performance
    /// This function uses plenty of floating-point operations, including
    /// divisions, natural logarithms, squareroots, and cuberoots, and thus
    /// it is not very performant.
    ///
    /// # Unchecked Operation
    /// This function does not check whether or not the orbit is hyperbolic. If
    /// this function is called on a non-hyperbolic orbit (i.e., elliptic or parabolic),
    /// invalid values may be returned.
    ///
    /// # Approximate Guess
    /// This function returns a "good" initial guess for the hyperbolic eccentric anomaly.  
    /// There are no constraints on the accuracy of the guess, and users may not
    /// rely on this value being very accurate, especially in some edge cases.
    ///
    /// # Source
    /// From the paper:  
    /// "A new method for solving the hyperbolic Kepler equation"  
    /// by Baisheng Wu et al.  
    /// Quote:
    /// "we divide the hyperbolic eccentric anomaly interval into two parts:
    /// a finite interval and an infinite interval. For the finite interval,
    /// we apply a piecewise Pade approximation to establish an initial
    /// approximate solution of HKE. For the infinite interval, an analytical
    /// initial approximate solution is constructed."
    #[doc(alias = "get_approx_hyp_ecc_anomaly")]
    fn get_approx_hyperbolic_eccentric_anomaly(&self, mean_anomaly: f64) -> f64 {
        solvers::get_approx_hyperbolic_eccentric_anomaly(self.get_eccentricity(), mean_anomaly)
    }

    /// Gets the hyperbolic eccentric anomaly of the orbit.
    ///
    /// # Unchecked Operation
    /// This function does not check whether or not the orbit is actually hyperbolic.  
    /// Nonsensical output may be produced if the orbit is not hyperbolic, but rather
    /// elliptic or parabolic.
    ///
    /// # Performance
    /// This function uses numerical methods to approach the value and therefore
    /// is not performant. It is recommended to cache this value if you can.
    ///
    /// # Source
    /// From the paper:  
    /// "A new method for solving the hyperbolic Kepler equation"  
    /// by Baisheng Wu et al.  
    fn get_hyperbolic_eccentric_anomaly(&self, mean_anomaly: f64) -> f64 {
        solvers::get_hyperbolic_eccentric_anomaly(self.get_eccentricity(), mean_anomaly)
    }

    /// Gets the elliptic eccentric anomaly of the orbit.
    ///
    /// # Unchecked Operation
    /// This function does not check whether or not the orbit is actually elliptic (e < 1).  
    /// Nonsensical output may be produced if the orbit is not elliptic, but rather
    /// hyperbolic or parabolic.
    ///
    /// # Performance
    /// This function uses numerical methods to approach the value and therefore
    /// is not performant. It is recommended to cache this value if you can.
    ///
    /// # Source
    /// From the paper  
    /// "An improved algorithm due to laguerre for the solution of Kepler's equation."  
    /// by Bruce A. Conway  
    /// <https://doi.org/10.1007/bf01230852>
    fn get_elliptic_eccentric_anomaly(&self, mean_anomaly: f64) -> f64 {
        solvers::get_elliptic_eccentric_anomaly(self.get_eccentricity(), mean_anomaly)
    }

    /// Gets the eccentric anomaly at a given true anomaly in the orbit.
    ///
    /// When the orbit is open (has an eccentricity of at least 1),
    /// the [hyperbolic eccentric anomaly](https://space.stackexchange.com/questions/27602/what-is-hyperbolic-eccentric-anomaly-f)
    /// would be returned instead.
    ///
    /// The eccentric anomaly is an angular parameter that defines the position
    /// of a body that is moving along an elliptic Kepler orbit.
    ///
    /// — [Wikipedia](https://en.wikipedia.org/wiki/Eccentric_anomaly)
    ///
    /// # Parabolic Support
    /// This function doesn't support parabolic trajectories yet.  
    /// `NaN`s or nonsensical values may be returned.
    ///
    /// # Performance
    /// The method to get the eccentric anomaly from the true anomaly
    /// uses a few trigonometry operations, and so it is not too performant.  
    /// It is, however, faster than the numerical approach methods used by
    /// the mean anomaly to eccentric anomaly conversion.  
    /// It is still recommended to cache this value if you can.
    #[doc(alias = "get_eccentric_anomaly_at_angle")]
    fn get_eccentric_anomaly_at_true_anomaly(&self, true_anomaly: f64) -> f64 {
        // TODO: PARABOLIC SUPPORT: This does not play well with parabolic trajectories.
        // Implement inverse of Barker's Equation for parabolas.
        let e = self.get_eccentricity();
        let true_anomaly = true_anomaly.rem_euclid(TAU);

        if e < 1.0 {
            // let v = true_anomaly,
            //   e = eccentricity,
            //   E = eccentric anomaly
            //
            // https://en.wikipedia.org/wiki/True_anomaly#From_the_eccentric_anomaly:
            // tan(v / 2) = sqrt((1 + e)/(1 - e)) * tan(E / 2)
            // 1 = sqrt((1 + e)/(1 - e)) * tan(E / 2) / tan(v / 2)
            // 1 / tan(E / 2) = sqrt((1 + e)/(1 - e)) / tan(v / 2)
            // tan(E / 2) = tan(v / 2) / sqrt((1 + e)/(1 - e))
            // E / 2 = atan(tan(v / 2) / sqrt((1 + e)/(1 - e)))
            // E = 2 * atan(tan(v / 2) / sqrt((1 + e)/(1 - e)))
            // E = 2 * atan(tan(v / 2) * sqrt((1 - e)/(1 + e)))

            2.0 * ((true_anomaly * 0.5).tan() * ((1.0 - e) / (1.0 + e)).sqrt()).atan()
        } else {
            // From the presentation "Spacecraft Dynamics and Control"
            // by Matthew M. Peet
            // https://control.asu.edu/Classes/MAE462/462Lecture05.pdf
            // Slide 25 of 27
            // Section "The Method for Hyperbolic Orbits"
            //
            // tan(f/2) = sqrt((e+1)/(e-1))*tanh(H/2)
            // 1 / tanh(H/2) = sqrt((e+1)/(e-1)) / tan(f/2)
            // tanh(H/2) = tan(f/2) / sqrt((e+1)/(e-1))
            // tanh(H/2) = tan(f/2) * sqrt((e-1)/(e+1))
            // H/2 = atanh(tan(f/2) * sqrt((e-1)/(e+1)))
            // H = 2 atanh(tan(f/2) * sqrt((e-1)/(e+1)))
            2.0 * ((true_anomaly * 0.5).tan() * ((e - 1.0) / (e + 1.0)).sqrt()).atanh()
        }
    }

    /// Gets the eccentric anomaly at a given time in the orbit.
    ///
    /// When the orbit is open (has an eccentricity of at least 1),
    /// the [hyperbolic eccentric anomaly](https://space.stackexchange.com/questions/27602/what-is-hyperbolic-eccentric-anomaly-f)
    /// would be returned instead.
    ///
    /// The eccentric anomaly is an angular parameter that defines the position
    /// of a body that is moving along an elliptic Kepler orbit.
    ///
    /// — [Wikipedia](https://en.wikipedia.org/wiki/Eccentric_anomaly)
    ///
    /// # Time
    /// The time is expressed in seconds.
    ///
    /// # Performance
    /// The method to get the eccentric anomaly from the time
    /// uses numerical approach methods, and so it is not performant.  
    /// It is recommended to cache this value if you can.
    fn get_eccentric_anomaly_at_time(&self, time: f64) -> f64 {
        self.get_eccentric_anomaly_at_mean_anomaly(self.get_mean_anomaly_at_time(time))
    }

    /// Gets the true anomaly at a given eccentric anomaly in the orbit.
    ///
    /// This function is faster than the function which takes mean anomaly as input,
    /// as the eccentric anomaly is hard to calculate.
    ///
    /// This function returns +/- pi for parabolic orbits due to how the equation works,
    /// and so **may result in infinities when combined with other functions**.
    ///
    /// The true anomaly is the angle between the direction of periapsis
    /// and the current position of the body, as seen from the main focus
    /// of the ellipse.
    ///
    /// — [Wikipedia](https://en.wikipedia.org/wiki/True_anomaly)
    ///
    /// # Performance  
    /// This function is faster than the function which takes mean anomaly as input,
    /// as the eccentric anomaly is hard to calculate.  
    /// However, this function still uses a few trigonometric functions, so it is
    /// not too performant.  
    /// It is recommended to cache this value if you can.
    fn get_true_anomaly_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> f64 {
        let eccentricity = self.get_eccentricity();
        if eccentricity < 1.0 {
            // https://en.wikipedia.org/wiki/True_anomaly#From_the_eccentric_anomaly
            let (s, c) = eccentric_anomaly.sin_cos();
            let beta = eccentricity / (1.0 + (1.0 - eccentricity * eccentricity).sqrt());

            eccentric_anomaly + 2.0 * (beta * s / (1.0 - beta * c)).atan()
        } else {
            // From the presentation "Spacecraft Dynamics and Control"
            // by Matthew M. Peet
            // https://control.asu.edu/Classes/MAE462/462Lecture05.pdf
            // Slide 25 of 27
            // Section "The Method for Hyperbolic Orbits"
            //
            // tan(f/2) = sqrt((e+1)/(e-1))*tanh(H/2)
            // f/2 = atan(sqrt((e+1)/(e-1))*tanh(H/2))
            // f = 2atan(sqrt((e+1)/(e-1))*tanh(H/2))

            2.0 * (((eccentricity + 1.0) / (eccentricity - 1.0)).sqrt()
                * (eccentric_anomaly * 0.5).tanh())
            .atan()
        }
    }

    /// Gets the true anomaly at a given mean anomaly in the orbit.
    ///
    /// This function returns +/- pi for parabolic orbits due to how the equation works,
    /// and so **may result in infinities when combined with other functions**.
    ///
    /// The true anomaly is the angle between the direction of periapsis
    /// and the current position of the body, as seen from the main focus
    /// of the ellipse.
    ///
    /// — [Wikipedia](https://en.wikipedia.org/wiki/True_anomaly)
    ///
    /// # Performance
    /// The true anomaly is derived from the eccentric anomaly, which
    /// uses numerical approach methods and so is not performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you already know the eccentric anomaly, you should use
    /// [`get_true_anomaly_at_eccentric_anomaly`][OrbitTrait2D::get_true_anomaly_at_eccentric_anomaly]
    /// instead.
    fn get_true_anomaly_at_mean_anomaly(&self, mean_anomaly: f64) -> f64 {
        self.get_true_anomaly_at_eccentric_anomaly(
            self.get_eccentric_anomaly_at_mean_anomaly(mean_anomaly),
        )
    }

    /// Gets the true anomaly at a given time in the orbit.
    ///
    /// The true anomaly is the angle between the direction of periapsis
    /// and the current position of the body, as seen from the main focus
    /// of the ellipse.
    ///
    /// — [Wikipedia](https://en.wikipedia.org/wiki/True_anomaly)
    ///
    /// This function returns +/- pi for parabolic orbits due to how the equation works,
    /// and so **may result in infinities when combined with other functions**.
    ///
    /// # Time
    /// The time is expressed in seconds.
    ///
    /// # Performance
    /// The true anomaly is derived from the eccentric anomaly, which
    /// uses numerical approach methods and so is not performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you already know the eccentric anomaly, you should use
    /// [`get_true_anomaly_at_eccentric_anomaly`][OrbitTrait2D::get_true_anomaly_at_eccentric_anomaly]
    /// instead.
    ///
    /// If you already know the mean anomaly, consider using
    /// [`get_true_anomaly_at_mean_anomaly`][OrbitTrait2D::get_true_anomaly_at_mean_anomaly]
    /// instead.  
    /// It won't help performance much, but it's not zero.
    fn get_true_anomaly_at_time(&self, time: f64) -> f64 {
        self.get_true_anomaly_at_mean_anomaly(self.get_mean_anomaly_at_time(time))
    }

    /// Gets the true anomaly where a certain altitude is reached.
    ///
    /// Returns NaN if the orbit is circular or there are no solutions.
    ///
    /// # Altitude
    /// The altitude is measured in meters, and measured from the
    /// center of the parent body (origin).
    ///
    /// The altitude given to this function should be between
    /// the periapsis (minimum) and the apoapsis (maximum).  
    /// Anything out of range will return NaN.
    ///
    /// In the case of hyperbolic orbits, there is no maximum,
    /// but the altitude should be positive and more than the periapsis.  
    /// Although there technically is a mathematical solution for "negative altitudes"
    /// between negative infinity and the apoapsis (which in this case is negative),
    /// they may not be very useful in most scenarios.
    ///
    /// # Domain
    /// This function returns a float between 0 and π, unless if
    /// it returns NaN.  
    /// Do note that, although this is the principal solution,
    /// other solutions exist, and may be desired. There exists an
    /// alternate solution when you negate the principal solution,
    /// and the solutions repeat every 2π.
    ///
    /// ## Example
    /// If there is a principal solution at `1`, that means there
    /// is an alternate solution at `-1`, and there are also
    /// solutions `2π + 1`, `2π - 1`, `4π + 1`, `4π - 1`, etc.
    ///
    /// # Performance
    /// This function is moderately performant and is unlikely to be
    /// the culprit of any performance issues.
    ///
    /// However, if you already computed the semi-latus rectum or the
    /// reciprocal of the eccentricity, you may use the unchecked version
    /// of this function for a small performance boost:  
    /// [`get_true_anomaly_at_altitude_unchecked`][OrbitTrait2D::get_true_anomaly_at_altitude_unchecked]
    #[doc(alias = "get_angle_at_altitude")]
    fn get_true_anomaly_at_altitude(&self, altitude: f64) -> f64 {
        self.get_true_anomaly_at_altitude_unchecked(
            self.get_semi_latus_rectum(),
            altitude,
            self.get_eccentricity().recip(),
        )
    }

    /// Gets the true anomaly where a certain altitude is reached.
    ///
    /// Returns NaN if the orbit is circular or there are no solutions.
    ///
    /// # Altitude
    /// The altitude is measured in meters, and measured from the
    /// center of the parent body (origin).
    ///
    /// The altitude given to this function should be between
    /// the periapsis (minimum) and the apoapsis (maximum).  
    /// Anything out of range will return NaN.
    ///
    /// In the case of hyperbolic orbits, there is no maximum,
    /// but the altitude should be positive and more than the periapsis.  
    /// Although there technically is a mathematical solution for "negative altitudes"
    /// between negative infinity and the apoapsis (which in this case is negative),
    /// they may not be very useful in most scenarios.
    ///
    /// # Unchecked Operation
    /// This function does not check the validity of the inputted
    /// values. Nonsensical/invalid inputs may result in
    /// nonsensical/invalid outputs.
    ///
    /// # Domain
    /// This function returns a float between 0 and π, unless if
    /// it returns NaN.  
    /// Do note that, although this is the principal solution,
    /// other solutions exist, and may be desired. There exists an
    /// alternate solution when you negate the principal solution,
    /// and the solutions repeat every 2π.
    ///
    /// ## Example
    /// If there is a principal solution at `1`, that means there
    /// is an alternate solution at `-1`, and there are also
    /// solutions `2π + 1`, `2π - 1`, `4π + 1`, `4π - 1`, etc.
    ///
    /// # Performance
    /// This function is moderately performant and is unlikely to be
    /// the culprit of any performance issues.
    #[doc(alias = "get_angle_at_altitude_unchecked")]
    fn get_true_anomaly_at_altitude_unchecked(
        &self,
        semi_latus_rectum: f64,
        altitude: f64,
        eccentricity_recip: f64,
    ) -> f64 {
        // r = p / (1 + e cos ν), r > 0, p > 0.
        // 1 / r = (1 + e cos ν) / p
        // 1 / r = 1 / p + (e cos ν / p)
        // 1 / r - 1 / p = e cos ν / p
        // e cos ν / p = 1 / r - 1 / p
        // e cos ν = p (1 / r - 1 / p)
        // e cos ν = p / r - 1
        // cos ν = (p / r - 1) / e
        // ν = acos((p / r - 1) / e)
        ((semi_latus_rectum / altitude - 1.0) * eccentricity_recip).acos()
    }

    /// Gets the mean anomaly at a given time in the orbit.
    ///
    /// The mean anomaly is the fraction of an elliptical orbit's period
    /// that has elapsed since the orbiting body passed periapsis,
    /// expressed as an angle which can be used in calculating the position
    /// of that body in the classical two-body problem.
    ///
    /// — [Wikipedia](https://en.wikipedia.org/wiki/Mean_anomaly)
    ///
    /// # Time
    /// The time is expressed in seconds.
    ///
    /// # Performance
    /// This function is performant and is unlikely to be the culprit of
    /// any performance issues.
    fn get_mean_anomaly_at_time(&self, time: f64) -> f64 {
        // M = t * sqrt(mu / |a^3|) + M_0
        time * (self.get_gravitational_parameter() / self.get_semi_major_axis().powi(3).abs())
            .sqrt()
            + self.get_mean_anomaly_at_epoch()
    }

    /// Gets the mean anomaly at a given eccentric anomaly in the orbit.
    ///
    /// The mean anomaly is the fraction of an elliptical orbit's period
    /// that has elapsed since the orbiting body passed periapsis,
    /// expressed as an angle which can be used in calculating the position
    /// of that body in the classical two-body problem.
    ///
    /// — [Wikipedia](https://en.wikipedia.org/wiki/Mean_anomaly)
    ///
    /// # Parabolic Support
    /// This function doesn't consider parabolic trajectories yet.  
    /// `NaN`s or nonsensical values may be returned.
    ///
    /// # Performance
    /// This function is a wrapper around
    /// [`get_mean_anomaly_at_elliptic_eccentric_anomaly`][OrbitTrait2D::get_mean_anomaly_at_elliptic_eccentric_anomaly]
    /// and
    /// [`get_mean_anomaly_at_hyperbolic_eccentric_anomaly`][OrbitTrait2D::get_mean_anomaly_at_hyperbolic_eccentric_anomaly].  
    /// It does some trigonometry, but if you know `sin(eccentric_anomaly)` or `sinh(eccentric_anomaly)`
    /// beforehand, this can be skipped by directly using those inner functions.
    fn get_mean_anomaly_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> f64 {
        // TODO: PARABOLIC SUPPORT: This function doesn't consider parabolas yet.
        if self.get_eccentricity() < 1.0 {
            self.get_mean_anomaly_at_elliptic_eccentric_anomaly(
                eccentric_anomaly,
                eccentric_anomaly.sin(),
            )
        } else {
            self.get_mean_anomaly_at_hyperbolic_eccentric_anomaly(
                eccentric_anomaly,
                eccentric_anomaly.sinh(),
            )
        }
    }

    /// Gets the mean anomaly at a given eccentric anomaly in the orbit and
    /// its precomputed sine.  
    ///
    /// The mean anomaly is the fraction of an elliptical orbit's period
    /// that has elapsed since the orbiting body passed periapsis,
    /// expressed as an angle which can be used in calculating the position
    /// of that body in the classical two-body problem.
    ///
    /// — [Wikipedia](https://en.wikipedia.org/wiki/Mean_anomaly)
    ///
    /// # Unchecked Operation
    /// This function does no checks on the validity of the value given
    /// as `sin_eccentric_anomaly`. It also doesn't check if the orbit is elliptic.  
    /// If invalid values are passed in, you will receive a possibly-nonsensical value as output.  
    ///
    /// # Performance
    /// This function is performant and is unlikely to be the culprit of
    /// any performance issues.
    fn get_mean_anomaly_at_elliptic_eccentric_anomaly(
        &self,
        eccentric_anomaly: f64,
        sin_eccentric_anomaly: f64,
    ) -> f64 {
        // https://en.wikipedia.org/wiki/Kepler%27s_equation#Equation
        //
        //      M = E - e sin E
        //
        // where:
        //   M = mean anomaly
        //   E = eccentric anomaly
        //   e = eccentricity
        eccentric_anomaly - self.get_eccentricity() * sin_eccentric_anomaly
    }

    /// Gets the mean anomaly at a given eccentric anomaly in the orbit and
    /// its precomputed sine.  
    ///
    /// The mean anomaly is the fraction of an elliptical orbit's period
    /// that has elapsed since the orbiting body passed periapsis,
    /// expressed as an angle which can be used in calculating the position
    /// of that body in the classical two-body problem.
    ///
    /// — [Wikipedia](https://en.wikipedia.org/wiki/Mean_anomaly)
    ///
    /// # Unchecked Operation
    /// This function does no checks on the validity of the value given
    /// as `sinh_eccentric_anomaly`. It also doesn't check if the orbit is hyperbolic.  
    /// If invalid values are passed in, you will receive a possibly-nonsensical value as output.  
    ///
    /// # Performance
    /// This function is performant and is unlikely to be the culprit of
    /// any performance issues.
    fn get_mean_anomaly_at_hyperbolic_eccentric_anomaly(
        &self,
        eccentric_anomaly: f64,
        sinh_eccentric_anomaly: f64,
    ) -> f64 {
        // https://en.wikipedia.org/wiki/Kepler%27s_equation#Hyperbolic_Kepler_equation
        //
        //      M = e sinh(H) - H
        //
        // where:
        //   M = mean anomaly
        //   e = eccentricity
        //   H = hyperbolic eccentric anomaly
        self.get_eccentricity() * sinh_eccentric_anomaly - eccentric_anomaly
    }

    /// Gets the mean anomaly at a given true anomaly in the orbit.
    ///
    /// The mean anomaly is the fraction of an elliptical orbit's period
    /// that has elapsed since the orbiting body passed periapsis,
    /// expressed as an angle which can be used in calculating the position
    /// of that body in the classical two-body problem.
    ///
    /// — [Wikipedia](https://en.wikipedia.org/wiki/Mean_anomaly)
    ///
    /// # Performance
    /// The method to get the eccentric anomaly from the true anomaly
    /// uses a few trigonometry operations, and so it is not too performant.  
    /// It is, however, faster than the numerical approach methods used by
    /// the mean anomaly to eccentric anomaly conversion.  
    /// It is still recommended to cache this value if you can.
    ///
    /// Alternatively, if you already know the eccentric anomaly, use
    /// [`get_mean_anomaly_at_eccentric_anomaly`][OrbitTrait2D::get_mean_anomaly_at_eccentric_anomaly]
    /// instead.
    #[doc(alias = "get_mean_anomaly_at_angle")]
    fn get_mean_anomaly_at_true_anomaly(&self, true_anomaly: f64) -> f64 {
        let ecc_anom = self.get_eccentric_anomaly_at_true_anomaly(true_anomaly);
        self.get_mean_anomaly_at_eccentric_anomaly(ecc_anom)
    }

    /// Gets the 3D position at a given angle (true anomaly) in the orbit.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Performance
    /// This function is somewhat performant.
    ///
    /// This function benefits significantly from being in the
    /// [cached version of the orbit struct][crate::Orbit2D].  
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_true_anomaly`][OrbitTrait2D::get_state_vectors_at_true_anomaly]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    ///
    /// If you only want to get the altitude of the orbit, you can use the
    /// [`get_altitude_at_true_anomaly`][OrbitTrait2D::get_altitude_at_true_anomaly]
    /// function instead.
    ///
    /// If you already know the altitude at the angle, you can
    /// rotate the altitude using the true anomaly, then tilt
    /// it using the [`transform_pqw_vector`][OrbitTrait2D::transform_pqw_vector]
    /// function instead.
    ///
    /// # Example
    /// ```
    /// use glam::DVec2;
    ///
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    ///
    /// let pos = orbit.get_position_at_true_anomaly(0.0);
    ///
    /// assert_eq!(pos, DVec2::new(100.0, 0.0));
    /// ```
    #[doc(alias = "get_position_at_angle")]
    fn get_position_at_true_anomaly(&self, angle: f64) -> DVec2 {
        self.transform_pqw_vector(self.get_pqw_position_at_true_anomaly(angle))
    }

    /// Gets the 3D position at a given eccentric anomaly in the orbit.
    ///
    /// # Performance
    /// This function benefits significantly from being in the
    /// [cached version of the orbit struct][crate::Orbit2D].  
    /// This function is not too performant as it uses a few trigonometric
    /// operations. It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you already know the true anomaly, you can use the
    /// [`get_position_at_true_anomaly`][OrbitTrait2D::get_position_at_true_anomaly]
    /// function instead.  
    /// Or, if you only need the altitude, use the
    /// [`get_altitude_at_eccentric_anomaly`][OrbitTrait2D::get_altitude_at_eccentric_anomaly]
    /// function instead.
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_eccentric_anomaly`][OrbitTrait2D::get_state_vectors_at_eccentric_anomaly]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    fn get_position_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> DVec2 {
        self.transform_pqw_vector(self.get_pqw_position_at_eccentric_anomaly(eccentric_anomaly))
    }

    /// Gets the speed at a given angle (true anomaly) in the orbit.
    ///
    /// The speed is derived from the vis-viva equation, and so is
    /// a lot faster than the velocity calculation.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.
    /// Anything out of range will get wrapped around.
    ///
    /// # Performance
    /// This function is performant, however, if you already know the altitude at the angle,
    /// you can use the [`get_speed_at_altitude`][OrbitTrait2D::get_speed_at_altitude]
    /// function instead to skip some calculations.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let speed_periapsis = orbit.get_speed_at_true_anomaly(0.0);
    /// let speed_apoapsis = orbit.get_speed_at_true_anomaly(std::f64::consts::PI);
    ///
    /// assert!(speed_periapsis > speed_apoapsis);
    /// ```
    #[doc(alias = "get_speed_at_angle")]
    fn get_speed_at_true_anomaly(&self, angle: f64) -> f64 {
        self.get_speed_at_altitude(self.get_altitude_at_true_anomaly(angle))
    }

    /// Gets the speed at a given altitude in the orbit.
    ///
    /// The speed is derived from the vis-viva equation, and so is
    /// a lot faster than the velocity calculation.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    ///
    /// # Unchecked Operation
    /// This function does no checks on the validity of the value given
    /// in the `altitude` parameter, namely whether or not this altitude
    /// is possible in the given orbit.  
    /// If invalid values are passed in, you will receive a possibly-nonsensical
    /// value as output.
    ///
    /// # Altitude
    /// The altitude is expressed in meters, and is the distance to the
    /// center of the orbit.
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// const PERIAPSIS: f64 = 100.0;
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(PERIAPSIS);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let apoapsis = orbit.get_apoapsis();
    ///
    /// let speed_periapsis = orbit.get_speed_at_altitude(PERIAPSIS);
    /// let speed_apoapsis = orbit.get_speed_at_altitude(apoapsis);
    ///
    /// assert!(speed_periapsis > speed_apoapsis);
    /// ```
    fn get_speed_at_altitude(&self, altitude: f64) -> f64 {
        // https://en.wikipedia.org/wiki/Vis-viva_equation
        // v^2 = GM (2/r - 1/a)
        // => v = sqrt(GM * (2/r - 1/a))
        //
        // note:
        // a = r_p / (1 - e)
        // => 1 / a = (1 - e) / r_p

        let r = altitude;
        let a_recip = (1.0 - self.get_eccentricity()) / self.get_periapsis();

        ((2.0 / r - a_recip) * self.get_gravitational_parameter()).sqrt()
    }

    /// Gets the speed at the periapsis of the orbit.
    ///
    /// The speed is derived from the vis-viva equation, and so is
    /// a lot faster than the velocity calculation.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// const PERIAPSIS: f64 = 100.0;
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(PERIAPSIS);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let naive_getter = orbit.get_speed_at_altitude(PERIAPSIS);
    /// let dedicated_getter = orbit.get_speed_at_periapsis();
    ///
    /// assert!((naive_getter - dedicated_getter).abs() < 1e-14);
    /// ```
    fn get_speed_at_periapsis(&self) -> f64 {
        // https://en.wikipedia.org/wiki/Vis-viva_equation
        // v^2 = GM (2/r_p - 1/a)
        // => v = sqrt(GM * (2/r_p - 1/a))
        //
        // note:
        // a = r_p / (1 - e)
        // => 1 / a = (1 - e) / r_p
        //
        // => v = sqrt(
        //      GM
        //      * (2/r_p - (1 - e) / r_p)
        //    )
        // => v = sqrt(
        //      GM
        //      * ((2 - (1 - e)) / r_p)
        //    )
        //
        // note:
        // 2 - (1 - e) = e + 1
        //
        // => v = sqrt(
        //      GM
        //      * ((e + 1) / r_p)
        //    )
        (self.get_gravitational_parameter()
            * ((self.get_eccentricity() + 1.0) / self.get_periapsis()))
        .sqrt()
    }

    /// Gets the speed at the apoapsis of the orbit.
    ///
    /// The speed is derived from the vis-viva equation, and so is
    /// a lot faster than the velocity calculation.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    ///
    /// # Open orbits (eccentricity >= 1)
    /// This function does not handle open orbits specially, and will return
    /// a non-physical value. You might want to use the getter for the speed
    /// at infinity:
    /// [`get_speed_at_infinity`][OrbitTrait2D::get_speed_at_infinity]
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// const APOAPSIS: f64 = 200.0;
    /// const PERIAPSIS: f64 = 100.0;
    ///
    /// let orbit = Orbit2D::with_apoapsis(
    ///     APOAPSIS,
    ///     PERIAPSIS,
    ///     0.0, // Argument of periapsis
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter
    /// );
    ///
    /// let naive_getter = orbit.get_speed_at_altitude(APOAPSIS);
    /// let dedicated_getter = orbit.get_speed_at_apoapsis();
    ///
    /// assert!((naive_getter - dedicated_getter).abs() < 1e-14);
    /// ```
    fn get_speed_at_apoapsis(&self) -> f64 {
        // https://en.wikipedia.org/wiki/Vis-viva_equation
        // v^2 = GM (2/r_a - 1/a)
        // => v = sqrt(GM * (2/r_a - 1/a))
        //
        // note: to simplify,
        // we define `left` and `right`:
        // left := 2 / r_a; right := 1 / a
        //
        // and also define `inner`:
        // inner := left - right
        //
        // which means we can simplify v:
        // => v = sqrt(GM * inner)
        //
        // note:
        // a = r_p / (1 - e)
        // => 1 / a = (1 - e) / r_p
        //
        // => right = (1 - e) / r_p
        //
        // note:
        // r_a = a * (1 + e)
        // = r_p / (1 - e) * (1 + e)
        // = r_p * (1 + e) / (1 - e)
        //
        // => left = 2 / (r_p * (1 + e) / (1 - e))
        // = (2 * (1 - e)) / (r_p * (1 + e))
        //
        // note: use the same denominator `(r_p * (1 + e))`
        //
        // recall right = (1 - e) / r_p.
        // right = right * (1 + e)/(1 + e)
        // = ((1 - e) * (1 + e)) / (r_p * (1 + e))
        //
        // recall inner := left - right.
        // => inner = (2 * (1 - e)) / (r_p * (1 + e))
        //  - ((1 - e) * (1 + e)) / (r_p * (1 + e))
        //
        // factor out 1 / (r_p * (1 + e)):
        // => inner = (2 * (1 - e) - (1 - e) * (1 + e)) / (r_p * (1 + e))
        //
        // factor out (1 - e) in numerator:
        // => inner = ((2 - (1 + e)) * (1 - e)) / (r_p * (1 + e))
        // = ((2 - 1 - e) * (1 - e)) / (r_p * (1 + e))
        // = ((1 - e) * (1 - e)) / (r_p * (1 + e))
        // = (1 - e)^2 / (r_p * (1 + e))
        //
        // recall v = sqrt(GM * inner).
        // => v = sqrt(GM * (1 - e)^2 / (r_p * (1 + e)))
        //
        // recall that for all x >= 0, sqrt(x^2) = x,
        //   and that for all x < 0, sqrt(x^2) = -x.
        //
        // => we can factor out (1 - e)^2 outside the sqrt.
        // => v = |1 - e| sqrt(GM / (r_p * (1 + e)))

        (1.0 - self.get_eccentricity()).abs()
            * (self.get_gravitational_parameter()
                / (self.get_periapsis() * (1.0 + self.get_eccentricity())))
            .sqrt()
    }

    /// Gets the hyperbolic excess speed (v_∞) of the trajectory.
    ///
    /// Under simplistic assumptions a body traveling along
    /// [a hyperbolic] trajectory will coast towards infinity,
    /// settling to __a final excess velocity__ relative to
    /// the central body.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Hyperbolic_trajectory)
    ///
    /// In other words, as the time of a hyperbolic trajectory
    /// approaches infinity, the speed approaches a certain
    /// speed, called the hyperbolic excess speed.
    ///
    /// # Unchecked Operation
    /// This function does not check that the orbit is open.  
    /// This function will return NaN for closed orbits (e < 1).
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let orbit = Orbit2D::new(
    ///     1.0, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Argument of periapsis
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter
    /// );
    ///
    /// assert_eq!(orbit.get_speed_at_infinity(), 0.0);
    /// ```
    #[doc(alias = "get_hyperbolic_excess_speed")]
    #[doc(alias = "get_speed_at_asymptote")]
    fn get_speed_at_infinity(&self) -> f64 {
        // https://en.wikipedia.org/wiki/Hyperbolic_trajectory#Parameters_describing_a_hyperbolic_trajectory
        // v_∞ = sqrt(-μ / a)
        //
        // recall a = r_p / (1 - e)
        //
        // => v_∞ = sqrt(-μ / (r_p / (1 - e)))
        // => v_∞ = sqrt(-μ * (1 - e) / r_p)
        // => v_∞ = sqrt(μ * (e - 1) / r_p)
        (self.get_gravitational_parameter() * (self.get_eccentricity() - 1.0)
            / self.get_periapsis())
        .sqrt()
    }

    /// Gets the speed at a given time in the orbit.
    ///
    /// # Time
    /// The time is expressed in seconds.
    ///
    /// # Performance
    /// The time will be converted into an eccentric anomaly, which uses
    /// numerical methods and so is not very performant.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you know the eccentric anomaly or the true anomaly,
    /// then you should use the
    /// [`get_speed_at_eccentric_anomaly`][OrbitTrait2D::get_speed_at_eccentric_anomaly]
    /// and
    /// [`get_speed_at_true_anomaly`][OrbitTrait2D::get_speed_at_true_anomaly]
    /// functions instead.  
    /// Those do not use numerical methods and therefore are a lot faster.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_speed_at_time(&self, time: f64) -> f64 {
        self.get_speed_at_true_anomaly(self.get_true_anomaly_at_time(time))
    }

    /// Gets the speed at a given eccentric anomaly in the orbit.
    ///
    /// The speed is derived from the vis-viva equation, and so is
    /// a lot faster than the velocity calculation.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    ///
    /// # Performance
    /// This function is not too performant as it uses a few trigonometric
    /// operations.  
    /// It is recommended to cache this value if you can.  
    ///
    /// Alternatively, if you already know the true anomaly,
    /// then you should use the
    /// [`get_speed_at_true_anomaly`][OrbitTrait2D::get_speed_at_true_anomaly]
    /// function instead.
    fn get_speed_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> f64 {
        self.get_speed_at_true_anomaly(
            self.get_true_anomaly_at_eccentric_anomaly(eccentric_anomaly),
        )
    }

    /// Gets the velocity at a given angle (true anomaly) in the orbit
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// If you want to get the vector in the regular coordinate system instead, use
    /// [`get_velocity_at_true_anomaly`][OrbitTrait2D::get_velocity_at_true_anomaly] instead.
    ///
    /// # Performance
    /// This function is not too performant as it uses some trigonometric operations.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_true_anomaly`][OrbitTrait2D::get_speed_at_true_anomaly] instead.  
    /// And if you already know the eccentric anomaly, use
    /// [`get_pqw_velocity_at_eccentric_anomaly`][OrbitTrait2D::get_pqw_velocity_at_eccentric_anomaly]
    /// instead.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let vel_periapsis = orbit.get_pqw_velocity_at_true_anomaly(0.0);
    /// let vel_apoapsis = orbit.get_pqw_velocity_at_true_anomaly(std::f64::consts::PI);
    ///
    /// let speed_periapsis = vel_periapsis.length();
    /// let speed_apoapsis = vel_apoapsis.length();
    ///
    /// assert!(speed_periapsis > speed_apoapsis);
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    #[doc(alias = "get_flat_velocity_at_angle")]
    #[doc(alias = "get_pqw_velocity_at_angle")]
    fn get_pqw_velocity_at_true_anomaly(&self, angle: f64) -> DVec2 {
        // TODO: PARABOLIC SUPPORT: This does not play well with parabolic trajectories.
        let outer_mult = (self.get_semi_major_axis() * self.get_gravitational_parameter())
            .abs()
            .sqrt()
            / self.get_altitude_at_true_anomaly(angle);

        let q_mult = (1.0 - self.get_eccentricity().powi(2)).abs().sqrt();

        let eccentric_anomaly = self.get_eccentric_anomaly_at_true_anomaly(angle);

        let trig_ecc_anom = if self.get_eccentricity() < 1.0 {
            eccentric_anomaly.sin_cos()
        } else {
            sinhcosh(eccentric_anomaly)
        };

        self.get_pqw_velocity_at_eccentric_anomaly_unchecked(outer_mult, q_mult, trig_ecc_anom)
    }

    /// Gets the velocity at the periapsis of the orbit
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// If you want to get the vector in the regular coordinate system instead, use
    /// [`get_velocity_at_periapsis`][OrbitTrait2D::get_velocity_at_periapsis] instead.
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_periapsis`][OrbitTrait2D::get_speed_at_periapsis] instead.  
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let vel = orbit.get_pqw_velocity_at_periapsis();
    ///
    /// assert_eq!(
    ///     vel,
    ///     DVec2::new(0.0, orbit.get_speed_at_periapsis())
    /// );
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_pqw_velocity_at_periapsis(&self) -> DVec2 {
        DVec2::new(0.0, self.get_speed_at_periapsis())
    }

    /// Gets the velocity at the apoapsis of the orbit
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// If you want to get the vector in the regular coordinate system instead, use
    /// [`get_velocity_at_apoapsis`][OrbitTrait2D::get_velocity_at_apoapsis] instead.
    ///
    /// # Open orbits (eccentricity >= 1)
    /// This function does not handle open orbits specially, and will return
    /// a non-physical value. You might want to use the getters for the velocity
    /// at the incoming and outgoing asymptotes:
    /// - [`get_pqw_velocity_at_incoming_asymptote`][OrbitTrait2D::get_pqw_velocity_at_incoming_asymptote]
    /// - [`get_pqw_velocity_at_outgoing_asymptote`][OrbitTrait2D::get_pqw_velocity_at_outgoing_asymptote]
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_apoapsis`][OrbitTrait2D::get_speed_at_apoapsis] instead.  
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let vel = orbit.get_pqw_velocity_at_apoapsis();
    ///
    /// assert_eq!(
    ///     vel,
    ///     DVec2::new(0.0, -orbit.get_speed_at_apoapsis())
    /// );
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_pqw_velocity_at_apoapsis(&self) -> DVec2 {
        DVec2::new(0.0, -self.get_speed_at_apoapsis())
    }

    /// Gets the velocity at the incoming asymptote of the trajectory
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// If you want to get the vector in the regular coordinate system instead, use
    /// [`get_velocity_at_incoming_asymptote`][OrbitTrait2D::get_velocity_at_incoming_asymptote]
    /// instead.
    ///
    /// # Unchecked Operation
    /// This function does not check that the orbit is open.  
    /// This function will return a NaN vector for closed orbits (e < 1).
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_infinity`][OrbitTrait2D::get_speed_at_infinity] instead.  
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(1.5);
    ///
    /// let speed = orbit.get_speed_at_infinity();
    /// let vel = orbit.get_pqw_velocity_at_incoming_asymptote();
    ///
    /// assert!((vel.length() - speed).abs() < 1e-15);
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_pqw_velocity_at_incoming_asymptote(&self) -> DVec2 {
        let asymptote_true_anom = -self.get_true_anomaly_at_asymptote();
        let (sin_true, cos_true) = asymptote_true_anom.sin_cos();
        -self.get_speed_at_infinity() * DVec2::new(cos_true, sin_true)
    }

    /// Gets the velocity at the outgoing asymptote of the trajectory
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// If you want to get the vector in the regular coordinate system instead, use
    /// [`get_velocity_at_outgoing_asymptote`][OrbitTrait2D::get_velocity_at_outgoing_asymptote]
    /// instead.
    ///
    /// # Unchecked Operation
    /// This function does not check that the orbit is open.  
    /// This function will return a NaN vector for closed orbits (e < 1).
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_infinity`][OrbitTrait2D::get_speed_at_infinity] instead.  
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    /// use glam::DVec2;
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(1.5);
    ///
    /// let speed = orbit.get_speed_at_infinity();
    /// let vel = orbit.get_pqw_velocity_at_outgoing_asymptote();
    ///
    /// assert!((vel.length() - speed).abs() < 1e-15);
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    #[doc(alias = "get_hyperbolic_excess_pqw_velocity")]
    fn get_pqw_velocity_at_outgoing_asymptote(&self) -> DVec2 {
        let asymptote_true_anom = self.get_true_anomaly_at_asymptote();
        let (sin_true, cos_true) = asymptote_true_anom.sin_cos();
        self.get_speed_at_infinity() * DVec2::new(cos_true, sin_true)
    }

    /// Gets the velocity at a given eccentric anomaly in the orbit
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// If you want to get the vector in the regular coordinate system instead, use
    /// [`get_velocity_at_eccentric_anomaly`][OrbitTrait2D::get_velocity_at_eccentric_anomaly] instead.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    ///
    /// # Parabolic Support
    /// This function doesn't consider parabolic trajectories yet.  
    /// `NaN`s or parabolic trajectories may be returned.
    ///
    /// # Performance
    /// This function is not too performant as it uses some trigonometric
    /// operations.  
    /// It is recommended to cache this value if you can.
    /// If you want to just get the speed, consider using the
    /// [`get_speed_at_eccentric_anomaly`][OrbitTrait2D::get_speed_at_eccentric_anomaly]
    /// function instead.
    ///
    /// Alternatively, if you already know some values (such as the altitude), consider
    /// using the unchecked version of the function instead:  
    /// [`get_pqw_velocity_at_eccentric_anomaly_unchecked`][OrbitTrait2D::get_pqw_velocity_at_eccentric_anomaly_unchecked]
    #[doc(alias = "get_flat_velocity_at_eccentric_anomaly")]
    fn get_pqw_velocity_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> DVec2 {
        // TODO: PARABOLIC SUPPORT: This does not play well with parabolic trajectories.
        let outer_mult = (self.get_semi_major_axis() * self.get_gravitational_parameter())
            .abs()
            .sqrt()
            / self.get_altitude_at_eccentric_anomaly(eccentric_anomaly);

        let q_mult = (1.0 - self.get_eccentricity().powi(2)).abs().sqrt();

        let trig_ecc_anom = if self.get_eccentricity() < 1.0 {
            eccentric_anomaly.sin_cos()
        } else {
            sinhcosh(eccentric_anomaly)
        };

        self.get_pqw_velocity_at_eccentric_anomaly_unchecked(outer_mult, q_mult, trig_ecc_anom)
    }

    /// Gets the velocity at a given eccentric anomaly in the orbit
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Unchecked Operation
    /// This function does not check the validity of the
    /// inputs passed to this function.  
    /// It is your responsibility to make sure the inputs passed in are valid.  
    /// Failing to do so may result in nonsensical outputs.
    ///
    /// # Parameters
    /// ## `outer_mult`
    /// This parameter is a multiplier for the entire 2D vector.  
    /// If the orbit is elliptic (e < 1), it should be calculated by
    /// the formula `sqrt(GM * a) / r`, where `GM` is the gravitational
    /// parameter, `a` is the semi-major axis, and `r` is the altitude of
    /// the orbit at that point.  
    /// If the orbit is hyperbolic (e > 1), it should instead be calculated by
    /// the formula `sqrt(-GM * a) / r`.  
    /// For the general case, the formula `sqrt(abs(GM * a)) / r` can be used instead.
    ///
    /// ## `q_mult`
    /// This parameter is a multiplier for the second element in the PQW vector.  
    /// For elliptic orbits, it should be calculated by the formula `sqrt(1 - e^2)`,
    /// where `e` is the eccentricity of the orbit.  
    /// For hyperbolic orbits, it should be calculated by the formula `sqrt(e^2 - 1)`,
    /// where `e` is the eccentricity of the orbit.  
    /// Alternatively, for the general case, you can use the formula `sqrt(abs(1 - e^2))`.
    ///
    /// ## `trig_ecc_anom`
    /// **For elliptic orbits**, this parameter should be a tuple containing the sine and cosine
    /// values of the eccentric anomaly, respectively.  
    /// **For hyperbolic orbits**, this parameter should be a tuple containing the **hyperbolic**
    /// sine and **hyperbolic** cosine values of the eccentric anomaly, respectively.
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// You can convert from the PQW system to the regular 3D space using
    /// [`transform_pqw_vector`][OrbitTrait2D::transform_pqw_vector].
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    ///
    /// # Parabolic Support
    /// This function doesn't consider parabolic trajectories yet.  
    /// `NaN`s or parabolic trajectories may be returned.
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, sinhcosh};
    ///
    /// # fn main() {
    /// let orbit = Orbit2D::default();
    ///
    /// let eccentric_anomaly: f64 = 1.25;
    ///
    /// let pqw_vel = orbit.get_pqw_velocity_at_eccentric_anomaly(eccentric_anomaly);
    ///
    /// let gm = orbit.get_gravitational_parameter();
    /// let a = orbit.get_semi_major_axis();
    /// let altitude = orbit.get_altitude_at_eccentric_anomaly(eccentric_anomaly);
    /// let outer_mult = (gm * a).sqrt() / altitude;
    ///
    /// let q_mult = (1.0 - orbit.get_eccentricity().powi(2)).sqrt();
    ///
    /// let trig_ecc_anom = eccentric_anomaly.sin_cos();
    ///
    /// let pqw_vel_2 = orbit
    ///     .get_pqw_velocity_at_eccentric_anomaly_unchecked(outer_mult, q_mult, trig_ecc_anom);
    ///
    /// assert_eq!(pqw_vel, pqw_vel_2);
    /// }
    /// ```
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, sinhcosh};
    ///
    /// # fn main() {
    /// let mut hyperbolic = Orbit2D::default();
    /// hyperbolic.set_eccentricity(3.0);
    ///
    /// let eccentric_anomaly: f64 = 2.35;
    ///
    /// let pqw_vel = hyperbolic.get_pqw_velocity_at_eccentric_anomaly(eccentric_anomaly);
    ///
    /// let gm = hyperbolic.get_gravitational_parameter();
    /// let a = hyperbolic.get_semi_major_axis();
    /// let altitude = hyperbolic.get_altitude_at_eccentric_anomaly(eccentric_anomaly);
    /// let outer_mult = (-gm * a).sqrt() / altitude;
    ///
    /// let q_mult = (hyperbolic.get_eccentricity().powi(2) - 1.0).sqrt();
    ///
    /// let trig_ecc_anom = sinhcosh(eccentric_anomaly);
    ///
    /// let pqw_vel_2 = hyperbolic
    ///     .get_pqw_velocity_at_eccentric_anomaly_unchecked(outer_mult, q_mult, trig_ecc_anom);
    ///
    /// assert_eq!(pqw_vel, pqw_vel_2);
    /// # }
    /// ```
    fn get_pqw_velocity_at_eccentric_anomaly_unchecked(
        &self,
        outer_mult: f64,
        q_mult: f64,
        trig_ecc_anom: (f64, f64),
    ) -> DVec2 {
        // ==== ELLIPTIC CASE: ====
        // https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
        // Equation 8:
        //                                   [      -sin E       ]
        // vector_o'(t) = sqrt(GM * a) / r * [ sqrt(1-e^2) cos E ]
        //                                   [         0         ]
        //
        // outer_mult = sqrt(GM * a) / r
        // trig_ecc_anom = eccentric_anomaly.sin_cos()
        // q_mult = sqrt(1 - e^2)
        //
        //
        // ==== HYPERBOLIC CASE: ====
        // https://space.stackexchange.com/a/54418
        //                                    [      -sinh F       ]
        // vector_o'(t) = sqrt(-GM * a) / r * [ sqrt(e^2-1) cosh F ]
        //                                    [         0          ]
        //
        // outer_mult = sqrt(GM * a) / r
        // trig_ecc_anom = (eccentric_anomaly.sinh(), eccentric_anomaly.cosh())
        // q_mult = sqrt(e^2 - 1)
        DVec2::new(
            outer_mult * -trig_ecc_anom.0,
            outer_mult * q_mult * trig_ecc_anom.1,
        )
    }

    /// Gets the velocity at a given time in the orbit
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// If you want to get the vector in the regular coordinate system instead, use
    /// [`get_velocity_at_time`][OrbitTrait2D::get_velocity_at_time] instead.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    ///
    /// # Time
    /// The time is expressed in seconds.
    ///
    /// # Performance
    /// This method involves converting the time into an eccentric anomaly,
    /// which uses numerical methods and so is not performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you already know the eccentric anomaly or the true anomaly,
    /// then you should use the
    /// [`get_pqw_velocity_at_eccentric_anomaly`][OrbitTrait2D::get_pqw_velocity_at_eccentric_anomaly]
    /// and
    /// [`get_pqw_velocity_at_true_anomaly`][OrbitTrait2D::get_pqw_velocity_at_true_anomaly]
    /// functions instead.  
    /// Those do not use numerical methods and therefore are a lot faster.
    #[doc(alias = "get_flat_velocity_at_time")]
    fn get_pqw_velocity_at_time(&self, time: f64) -> DVec2 {
        self.get_pqw_velocity_at_eccentric_anomaly(self.get_eccentric_anomaly_at_time(time))
    }

    /// Gets the position at a given angle (true anomaly) in the orbit
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// If you want to get the vector in the regular coordinate system instead, use
    /// [`get_position_at_true_anomaly`][OrbitTrait2D::get_position_at_true_anomaly] instead.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Performance
    /// This function is somewhat performant. However, if you already know
    /// the altitude beforehand, you might be interested in the unchecked
    /// version of this function:
    /// [`get_pqw_position_at_true_anomaly_unchecked`][OrbitTrait2D::get_pqw_position_at_true_anomaly_unchecked]  
    /// If you're looking to just get the altitude at a given angle,
    /// consider using the [`get_altitude_at_true_anomaly`][OrbitTrait2D::get_altitude_at_true_anomaly]
    /// function instead.
    ///
    /// # Example
    /// ```
    /// use glam::DVec2;
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.0);
    ///
    /// let pos = orbit.get_pqw_position_at_true_anomaly(0.0);
    ///
    /// assert_eq!(pos, DVec2::new(100.0, 0.0));
    /// ```
    #[doc(alias = "get_flat_position_at_angle")]
    #[doc(alias = "get_pqw_position_at_angle")]
    fn get_pqw_position_at_true_anomaly(&self, angle: f64) -> DVec2 {
        let alt = self.get_altitude_at_true_anomaly(angle);
        let (sin, cos) = angle.sin_cos();
        DVec2::new(alt * cos, alt * sin)
    }

    /// Gets the position at a certain point in the orbit
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Unchecked Operation
    /// This function does not check on the validity of the parameters.  
    /// Invalid values may lead to nonsensical results.
    ///
    /// # Parameters
    /// ## `altitude`
    /// The altitude at that certain point in the orbit.  
    /// The altitude is measured in meters, and measured from the
    /// center of the parent body (origin).
    /// ## `sincos_angle`
    /// A tuple containing the sine and cosine (respectively) of the true anomaly
    /// of the point in the orbit.
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// To convert from the PQW coordinates to regular 3D space, use the
    /// [`transform_pqw_vector`][OrbitTrait2D::transform_pqw_vector] function.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any performance
    /// issues.
    ///
    /// # Example
    /// ```
    /// use glam::DVec2;
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.0);
    ///
    /// let true_anomaly = 1.06;
    ///
    /// let pos = orbit.get_pqw_position_at_true_anomaly(true_anomaly);
    ///
    /// let altitude = orbit.get_altitude_at_true_anomaly(true_anomaly);
    /// let sincos_angle = true_anomaly.sin_cos();
    ///
    /// let pos2 = orbit.get_pqw_position_at_true_anomaly_unchecked(altitude, sincos_angle);
    ///
    /// assert_eq!(pos, pos2);
    /// ```
    fn get_pqw_position_at_true_anomaly_unchecked(
        &self,
        altitude: f64,
        sincos_angle: (f64, f64),
    ) -> DVec2 {
        DVec2::new(altitude * sincos_angle.1, altitude * sincos_angle.0)
    }

    // TODO: DOC: POST-PARABOLIC SUPPORT: Update doc
    /// Gets the position at a given time in the orbit
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// If you want to get the vector in the regular coordinate system instead, use
    /// [`get_position_at_time`][OrbitTrait2D::get_position_at_time] instead.
    ///
    /// # Time
    /// The time is expressed in seconds.
    ///
    /// # Parabolic Support
    /// **This function returns non-finite numbers for parabolic orbits**
    /// due to how the equation for true anomaly works.
    ///
    /// # Performance
    /// This involves calculating the true anomaly at a given time,
    /// and so is not performant.
    /// It is recommended to cache this value when possible.
    ///
    /// Alternatively, if you already know the eccentric anomaly or the true anomaly,
    /// consider using the
    /// [`get_pqw_position_at_eccentric_anomaly`][OrbitTrait2D::get_pqw_position_at_eccentric_anomaly]
    /// and
    /// [`get_pqw_position_at_true_anomaly`][OrbitTrait2D::get_pqw_position_at_true_anomaly]
    /// functions instead.
    /// Those do not use numerical methods and therefore are a lot faster.
    ///
    /// If you only want to get the altitude of the orbit, you can use the
    /// [`get_altitude_at_time`][OrbitTrait2D::get_altitude_at_time]
    /// function instead.
    #[doc(alias = "get_flat_position_at_time")]
    fn get_pqw_position_at_time(&self, time: f64) -> DVec2 {
        self.get_pqw_position_at_true_anomaly(self.get_true_anomaly_at_time(time))
    }

    // TODO: DOC: POST-PARABOLIC SUPPORT: Update doc
    /// Gets the position at a given eccentric anomaly in the orbit
    /// in the [perifocal coordinate system](https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
    ///
    /// # Perifocal Coordinate System
    /// This function returns a vector in the perifocal coordinate (PQW) system, where
    /// the first element points to the periapsis, and the second element has a
    /// true anomaly 90 degrees past the periapsis. The third element points perpendicular
    /// to the orbital plane, and is always zero in this case, and so it is omitted.
    ///
    /// Learn more about the PQW system: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// If you want to get the vector in the regular coordinate system instead, use
    /// [`get_position_at_eccentric_anomaly`][OrbitTrait2D::get_position_at_eccentric_anomaly] instead.
    ///
    /// # Time
    /// The time is expressed in seconds.
    ///
    /// # Parabolic Support
    /// **This function returns non-finite numbers for parabolic orbits**
    /// due to how the equation for true anomaly works.
    ///
    /// # Performance
    /// This function is not too performant as it uses a few trigonometric
    /// operations.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you already know the true anomaly,
    /// consider using the
    /// [`get_pqw_position_at_true_anomaly`][OrbitTrait2D::get_pqw_position_at_true_anomaly]
    /// function instead.
    #[doc(alias = "get_flat_position_at_eccentric_anomaly")]
    fn get_pqw_position_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> DVec2 {
        self.get_pqw_position_at_true_anomaly(
            self.get_true_anomaly_at_eccentric_anomaly(eccentric_anomaly),
        )
    }

    /// Gets the velocity at a given angle (true anomaly) in the orbit.
    ///
    /// # Performance
    /// This function is not too performant as it uses some trigonometric operations.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_true_anomaly`][OrbitTrait2D::get_speed_at_true_anomaly] instead.  
    /// Or, if you already have the eccentric anomaly, use
    /// [`get_velocity_at_eccentric_anomaly`][OrbitTrait2D::get_velocity_at_eccentric_anomaly]
    /// instead.
    /// These functions do less work and therefore are a lot faster.
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_true_anomaly`][OrbitTrait2D::get_state_vectors_at_true_anomaly]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let vel_periapsis = orbit.get_velocity_at_true_anomaly(0.0);
    /// let vel_apoapsis = orbit.get_velocity_at_true_anomaly(std::f64::consts::PI);
    ///
    /// let speed_periapsis = vel_periapsis.length();
    /// let speed_apoapsis = vel_apoapsis.length();
    ///
    /// assert!(speed_periapsis > speed_apoapsis)
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    #[doc(alias = "get_velocity_at_angle")]
    fn get_velocity_at_true_anomaly(&self, angle: f64) -> DVec2 {
        self.transform_pqw_vector(self.get_pqw_velocity_at_true_anomaly(angle))
    }

    /// Gets the velocity at the periapsis of the orbit.
    ///
    /// # Performance
    /// This function is not too performant as it uses some trigonometric operations.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_periapsis`][OrbitTrait2D::get_speed_at_periapsis] instead.  
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let speed = orbit.get_speed_at_periapsis();
    /// let vel = orbit.get_velocity_at_periapsis();
    ///
    /// assert!((vel.length() - speed).abs() < 1e-15);
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_velocity_at_periapsis(&self) -> DVec2 {
        self.transform_pqw_vector(self.get_pqw_velocity_at_periapsis())
    }

    /// Gets the velocity at the apoapsis of the orbit.
    ///
    /// # Performance
    /// This function is not too performant as it uses some trigonometric operations.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_apoapsis`][OrbitTrait2D::get_speed_at_apoapsis] instead.  
    ///
    /// # Open orbits (eccentricity >= 1)
    /// This function does not handle open orbits specially, and will return
    /// a non-physical value. You might want to use the getters for the velocity
    /// at the incoming and outgoing asymptotes:
    /// - [`get_velocity_at_incoming_asymptote`][OrbitTrait2D::get_velocity_at_incoming_asymptote]
    /// - [`get_velocity_at_outgoing_asymptote`][OrbitTrait2D::get_velocity_at_outgoing_asymptote]
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let speed = orbit.get_speed_at_apoapsis();
    /// let vel = orbit.get_velocity_at_apoapsis();
    ///
    /// assert!((vel.length() - speed).abs() < 1e-15);
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_velocity_at_apoapsis(&self) -> DVec2 {
        self.transform_pqw_vector(self.get_pqw_velocity_at_apoapsis())
    }

    /// Gets the velocity at the incoming asymptote of the trajectory.
    ///
    /// # Performance
    /// This function is not too performant as it uses some trigonometric operations.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_infinity`][OrbitTrait2D::get_speed_at_infinity] instead.  
    ///
    /// # Unchecked Operation
    /// This function does not check that the orbit is open.  
    /// This function will return a NaN vector for closed orbits (e < 1).
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(1.5);
    ///
    /// let speed = orbit.get_speed_at_infinity();
    /// let vel = orbit.get_velocity_at_incoming_asymptote();
    ///
    /// assert!((vel.length() - speed).abs() < 1e-15);
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_velocity_at_incoming_asymptote(&self) -> DVec2 {
        self.transform_pqw_vector(self.get_pqw_velocity_at_incoming_asymptote())
    }

    /// Gets the velocity at the outgoing asymptote of the trajectory.
    ///
    /// # Performance
    /// This function is not too performant as it uses some trigonometric operations.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_infinity`][OrbitTrait2D::get_speed_at_infinity] instead.  
    ///
    /// # Unchecked Operation
    /// This function does not check that the orbit is open.  
    /// This function will return a NaN vector for closed orbits (e < 1).
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(1.5);
    ///
    /// let speed = orbit.get_speed_at_infinity();
    /// let vel = orbit.get_velocity_at_outgoing_asymptote();
    ///
    /// assert!((vel.length() - speed).abs() < 1e-15);
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_velocity_at_outgoing_asymptote(&self) -> DVec2 {
        self.transform_pqw_vector(self.get_pqw_velocity_at_outgoing_asymptote())
    }

    /// Gets the velocity at a given eccentric anomaly in the orbit.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let vel_periapsis = orbit.get_velocity_at_true_anomaly(0.0);
    /// let vel_apoapsis = orbit.get_velocity_at_true_anomaly(std::f64::consts::PI);
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    ///
    /// # Performance
    /// This function is not too performant as it uses a few trigonometric
    /// operations.  
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you just want to get the speed, consider using the
    /// [`get_speed_at_eccentric_anomaly`][OrbitTrait2D::get_speed_at_eccentric_anomaly]
    /// function instead.
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_eccentric_anomaly`][OrbitTrait2D::get_state_vectors_at_eccentric_anomaly]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    ///
    /// This function benefits significantly from being in the
    /// [cached version of the orbit struct][crate::Orbit2D].
    fn get_velocity_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> DVec2 {
        self.transform_pqw_vector(self.get_pqw_velocity_at_eccentric_anomaly(eccentric_anomaly))
    }

    /// Gets the velocity at a given time in the orbit.
    ///
    /// # Time
    /// The time is expressed in seconds.
    ///
    /// # Performance
    /// The velocity is derived from the eccentric anomaly, which uses numerical
    /// methods and so is not performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_time`][OrbitTrait2D::get_speed_at_time] instead.  
    /// Or, if you already have the eccentric anomaly or true anomaly, use the
    /// [`get_velocity_at_eccentric_anomaly`][OrbitTrait2D::get_velocity_at_eccentric_anomaly]
    /// and
    /// [`get_velocity_at_true_anomaly`][OrbitTrait2D::get_velocity_at_true_anomaly]
    /// functions instead.  
    /// These functions do not require numerical methods and therefore are a lot faster.
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_time`][OrbitTrait2D::get_state_vectors_at_time]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_velocity_at_time(&self, time: f64) -> DVec2 {
        self.get_velocity_at_eccentric_anomaly(self.get_eccentric_anomaly_at_time(time))
    }

    /// Gets the altitude of the body from its parent at a given angle (true anomaly) in the orbit.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// Note that some angles, even within 0 to tau, are impossible for
    /// hyperbolic orbits and may result in invalid values.
    /// Check for the range of angles for a hyperbolic orbit using
    /// [`get_true_anomaly_at_asymptote`][OrbitTrait2D::get_true_anomaly_at_asymptote].
    ///
    /// # Altitude
    /// The altitude is measured in meters, and measured from the
    /// center of the parent body (origin).
    ///
    /// # Performance
    /// This function is performant, however, if you already
    /// know the orbit's semi-latus rectum or the cosine of the true anomaly,
    /// you can use the
    /// [`get_altitude_at_true_anomaly_unchecked`][OrbitTrait2D::get_altitude_at_true_anomaly_unchecked]
    /// function to skip a few steps in the calculation.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.0);
    ///
    /// let altitude = orbit.get_altitude_at_true_anomaly(0.0);
    ///
    /// assert_eq!(altitude, 100.0);
    /// ```
    #[doc(alias = "get_altitude_at_angle")]
    fn get_altitude_at_true_anomaly(&self, true_anomaly: f64) -> f64 {
        self.get_altitude_at_true_anomaly_unchecked(
            self.get_semi_latus_rectum(),
            true_anomaly.cos(),
        )
    }

    /// Gets the altitude of the body from its parent given the
    /// cosine of the true anomaly.  
    ///
    /// This function should only be used if you already know the semi-latus
    /// rectum or `cos(true_anomaly)` beforehand, and want to minimize
    /// duplicated work.
    ///
    /// # Unchecked Operation
    /// This function does not perform any checks on the validity
    /// of the `cos_true_anomaly` parameter. Invalid values result in
    /// possibly-nonsensical output values.
    ///
    /// # Altitude
    /// The altitude is measured in meters, and measured from the
    /// center of the parent body (origin).
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// Note that some angles, even within 0 to tau, are impossible for
    /// hyperbolic orbits and may result in invalid values.
    /// Check for the range of angles for a hyperbolic orbit using
    /// [`get_true_anomaly_at_asymptote`][OrbitTrait2D::get_true_anomaly_at_asymptote].
    ///
    /// # Performance
    /// This function, by itself, is performant and is unlikely
    /// to be the culprit of any performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let true_anomaly = 1.2345f64;
    ///
    /// // Precalculate some values...
    /// # let semi_latus_rectum = orbit.get_semi_latus_rectum();
    /// # let cos_true_anomaly = true_anomaly.cos();
    ///
    /// // Scenario 1: If you know just the semi-latus rectum
    /// let scenario_1 = orbit.get_altitude_at_true_anomaly_unchecked(
    ///     semi_latus_rectum, // We pass in our precalculated SLR...
    ///     true_anomaly.cos() // but calculate the cosine
    /// );
    ///
    /// // Scenario 2: If you know just the cosine of the true anomaly
    /// let scenario_2 = orbit.get_altitude_at_true_anomaly_unchecked(
    ///     orbit.get_semi_latus_rectum(), // We calculate the SLR...
    ///     cos_true_anomaly // but use our precalculated cosine
    /// );
    ///
    /// // Scenario 3: If you know both the semi-latus rectum:
    /// let scenario_3 = orbit.get_altitude_at_true_anomaly_unchecked(
    ///     semi_latus_rectum, // We pass in our precalculated SLR...
    ///     cos_true_anomaly // AND use our precalculated cosine
    /// );
    ///
    /// assert_eq!(scenario_1, scenario_2);
    /// assert_eq!(scenario_2, scenario_3);
    /// assert_eq!(scenario_3, orbit.get_altitude_at_true_anomaly(true_anomaly));
    /// ```
    fn get_altitude_at_true_anomaly_unchecked(
        &self,
        semi_latus_rectum: f64,
        cos_true_anomaly: f64,
    ) -> f64 {
        semi_latus_rectum / (1.0 + self.get_eccentricity() * cos_true_anomaly)
    }

    /// Gets the altitude at a given eccentric anomaly in the orbit.
    ///
    /// # Altitude
    /// The altitude is measured in meters, and measured from the
    /// center of the parent body (origin).
    ///
    /// # Performance
    /// This function is not too performant as it uses a few trigonometric operations.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you already know the true anomaly, use the
    /// [`get_altitude_at_true_anomaly`][OrbitTrait2D::get_altitude_at_true_anomaly]
    /// function instead.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.0);
    ///
    /// let altitude = orbit.get_altitude_at_eccentric_anomaly(0.0);
    ///
    /// assert_eq!(altitude, 100.0);
    /// ```
    fn get_altitude_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> f64 {
        self.get_altitude_at_true_anomaly(
            self.get_true_anomaly_at_eccentric_anomaly(eccentric_anomaly),
        )
    }

    // TODO: DOC: POST-PARABOLIC SUPPORT: Update doc
    /// Gets the altitude of the body from its parent at a given time in the orbit.
    ///
    /// Note that due to floating-point imprecision, values of extreme
    /// magnitude may not be accurate.
    ///
    /// # Time
    /// The time is expressed in seconds.
    ///
    /// # Altitude
    /// The altitude is measured in meters, and measured from the
    /// center of the parent body (origin).
    ///
    /// # Performance
    /// This involves calculating the true anomaly at a given time, and so is not very performant.  
    /// It is recommended to cache this value when possible.
    ///
    /// Alternatively, if you already know the eccentric anomaly or the true anomaly,
    /// consider using the
    /// [`get_altitude_at_eccentric_anomaly`][OrbitTrait2D::get_altitude_at_eccentric_anomaly]
    /// and
    /// [`get_altitude_at_true_anomaly`][OrbitTrait2D::get_altitude_at_true_anomaly]
    /// functions instead.  
    /// Those do not use numerical methods and therefore are a lot faster.
    ///
    /// # Parabolic Support
    /// **This function returns infinity for parabolic orbits** due to how the equation for
    /// true anomaly works.
    fn get_altitude_at_time(&self, time: f64) -> f64 {
        self.get_altitude_at_true_anomaly(self.get_true_anomaly_at_time(time))
    }

    // TODO: DOC: POST-PARABOLIC SUPPORT: Update doc
    /// Gets the 3D position at a given time in the orbit.
    ///
    /// # Time
    /// The time is expressed in seconds.
    ///
    /// # Performance
    /// This involves calculating the true anomaly at a given time,
    /// and so is not very performant.  
    /// It is recommended to cache this value when possible.
    ///
    /// This function benefits significantly from being in the
    /// [cached version of the orbit struct][crate::Orbit2D].  
    ///
    /// Alternatively, if you already know the true anomaly,
    /// consider using the
    /// [`get_position_at_true_anomaly`][OrbitTrait2D::get_position_at_true_anomaly]
    /// function instead.  
    /// That does not use numerical methods and therefore is a lot faster.
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_time`][OrbitTrait2D::get_state_vectors_at_time]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    ///
    /// # Parabolic Support
    /// **This function returns non-finite numbers for parabolic orbits**
    /// due to how the equation for true anomaly works.
    fn get_position_at_time(&self, time: f64) -> DVec2 {
        self.get_position_at_true_anomaly(self.get_true_anomaly_at_time(time))
    }

    // TODO: DOC: POST-PARABOLIC SUPPORT: Update doc
    /// Gets the 3D position and velocity at a given eccentric anomaly in the orbit.
    ///
    /// # Performance
    /// This function uses several trigonometric functions, and so it is not too performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// This is, however, faster than individually calling the position and velocity getters
    /// separately, as this will reuse calculations whenever possible.
    ///
    /// If you need *only one* of the vectors, though, you should instead call the dedicated
    /// getters:  
    /// [`get_velocity_at_eccentric_anomaly`][OrbitTrait2D::get_velocity_at_eccentric_anomaly]  
    /// [`get_position_at_eccentric_anomaly`][OrbitTrait2D::get_position_at_eccentric_anomaly]  
    ///
    /// This function should give similar performance to the getter from the true anomaly:  
    /// [`get_state_vectors_at_true_anomaly`][OrbitTrait2D::get_state_vectors_at_true_anomaly]
    ///
    /// In case you really want to, an unchecked version of this function is available:  
    /// [`get_state_vectors_from_unchecked_parts`][OrbitTrait2D::get_state_vectors_from_unchecked_parts]
    ///
    /// # Parabolic Support
    /// This function doesn't support parabolic trajectories yet.  
    /// `NaN`s or nonsensical values may be returned.
    fn get_state_vectors_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> StateVectors2D {
        let semi_major_axis = self.get_semi_major_axis();
        let sqrt_abs_gm_a = (semi_major_axis * self.get_gravitational_parameter())
            .abs()
            .sqrt();
        let true_anomaly = self.get_true_anomaly_at_eccentric_anomaly(eccentric_anomaly);
        let sincos_angle = true_anomaly.sin_cos();
        let eccentricity = self.get_eccentricity();

        let semi_latus_rectum = self.get_periapsis() * (1.0 + self.get_eccentricity());

        let altitude =
            self.get_altitude_at_true_anomaly_unchecked(semi_latus_rectum, sincos_angle.1);

        let q_mult = (1.0 - eccentricity.powi(2)).abs().sqrt();

        // TODO: PARABOLIC SUPPORT: This function doesn't yet consider parabolic orbits.
        let trig_ecc_anom = if eccentricity < 1.0 {
            eccentric_anomaly.sin_cos()
        } else {
            sinhcosh(eccentric_anomaly)
        };

        self.get_state_vectors_from_unchecked_parts(
            sqrt_abs_gm_a,
            altitude,
            q_mult,
            trig_ecc_anom,
            sincos_angle,
        )
    }

    /// Gets the 3D position and velocity at a given angle (true anomaly) in the orbit.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Performance
    /// This function uses several trigonometric functions, and so it is not too performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// This is, however, faster than individually calling the position and velocity getters
    /// separately, as this will reuse calculations whenever possible.
    ///
    /// If you need *only one* of the vectors, though, you should instead call the dedicated
    /// getters:  
    /// [`get_velocity_at_true_anomaly`][OrbitTrait2D::get_velocity_at_true_anomaly]
    /// [`get_position_at_true_anomaly`][OrbitTrait2D::get_position_at_true_anomaly]
    ///
    /// This function should give similar performance to the getter from the eccentric anomaly:  
    /// [`get_state_vectors_at_eccentric_anomaly`][OrbitTrait2D::get_state_vectors_at_true_anomaly]
    ///
    /// In case you really want to, an unchecked version of this function is available:  
    /// [`get_state_vectors_from_unchecked_parts`][OrbitTrait2D::get_state_vectors_from_unchecked_parts]
    ///
    /// # Parabolic Support
    /// This function doesn't support parabolic trajectories yet.  
    /// `NaN`s or nonsensical values may be returned.
    fn get_state_vectors_at_true_anomaly(&self, true_anomaly: f64) -> StateVectors2D {
        let semi_major_axis = self.get_semi_major_axis();
        let sqrt_abs_gm_a = (semi_major_axis * self.get_gravitational_parameter())
            .abs()
            .sqrt();
        let eccentric_anomaly = self.get_eccentric_anomaly_at_true_anomaly(true_anomaly);
        let sincos_angle = true_anomaly.sin_cos();
        let eccentricity = self.get_eccentricity();

        let semi_latus_rectum = self.get_periapsis() * (1.0 + self.get_eccentricity());

        let altitude =
            self.get_altitude_at_true_anomaly_unchecked(semi_latus_rectum, sincos_angle.1);

        let q_mult = (1.0 - eccentricity.powi(2)).abs().sqrt();

        // TODO: PARABOLIC SUPPORT: This function doesn't yet consider parabolic orbits.
        let trig_ecc_anom = if eccentricity < 1.0 {
            eccentric_anomaly.sin_cos()
        } else {
            sinhcosh(eccentric_anomaly)
        };

        self.get_state_vectors_from_unchecked_parts(
            sqrt_abs_gm_a,
            altitude,
            q_mult,
            trig_ecc_anom,
            sincos_angle,
        )
    }

    /// Gets the 3D position and velocity at a given mean anomaly in the orbit.
    ///
    /// # Performance
    /// This function involves converting the mean anomaly to an eccentric anomaly,
    /// which involves numerical approach methods and are therefore not performant.  
    /// It is recommended to cache this value if you can.  
    ///
    /// Alternatively, if you already know the eccentric anomaly or true anomaly,
    /// use the following functions instead, which do not use numerical methods and
    /// therefore are significantly faster:  
    /// [`get_state_vectors_at_eccentric_anomaly`][OrbitTrait2D::get_state_vectors_at_eccentric_anomaly]
    /// [`get_state_vectors_at_true_anomaly`][OrbitTrait2D::get_state_vectors_at_true_anomaly]
    ///
    /// This function is faster than individually calling the position and velocity getters
    /// separately, as this will reuse calculations whenever possible.
    ///
    /// # Parabolic Support
    /// This function doesn't support parabolic trajectories yet.  
    /// `NaN`s or nonsensical values may be returned.
    fn get_state_vectors_at_mean_anomaly(&self, mean_anomaly: f64) -> StateVectors2D {
        self.get_state_vectors_at_eccentric_anomaly(
            self.get_eccentric_anomaly_at_mean_anomaly(mean_anomaly),
        )
    }

    /// Gets the 3D position and velocity at a given time in the orbit.
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Performance
    /// This function involves converting a mean anomaly (derived from the time)
    /// into an eccentric anomaly.  
    /// This involves numerical approach methods and are therefore not performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you already know the eccentric anomaly or true anomaly,
    /// use the following functions instead, which do not use numerical methods and
    /// therefore are significantly faster:  
    /// [`get_state_vectors_at_eccentric_anomaly`][OrbitTrait2D::get_state_vectors_at_eccentric_anomaly]
    /// [`get_state_vectors_at_true_anomaly`][OrbitTrait2D::get_state_vectors_at_true_anomaly]
    ///
    /// If you only know the mean anomaly, then that may help with performance a little bit,
    /// in which case you can use
    /// [`get_state_vectors_at_mean_anomaly`][OrbitTrait2D::get_state_vectors_at_mean_anomaly]
    /// instead.
    ///
    /// This function is faster than individually calling the position and velocity getters
    /// separately, as this will reuse calculations whenever possible.
    ///
    /// If you need *only one* of the vectors, though, you should instead call the dedicated
    /// getters:  
    /// [`get_velocity_at_time`][OrbitTrait2D::get_velocity_at_time]
    /// [`get_position_at_time`][OrbitTrait2D::get_position_at_time]
    ///
    /// # Parabolic Support
    /// This function doesn't support parabolic trajectories yet.  
    /// `NaN`s or nonsensical values may be returned.
    fn get_state_vectors_at_time(&self, time: f64) -> StateVectors2D {
        self.get_state_vectors_at_mean_anomaly(self.get_mean_anomaly_at_time(time))
    }

    /// Gets the 3D position and velocity at a certain point in the orbit.
    ///
    /// # Unchecked Operation
    /// This function does not check the validity of the inputs.  
    /// Invalid inputs may lead to nonsensical results.
    ///
    /// # Parameters
    /// ## `sqrt_abs_gm_a`
    /// This parameter's value should be calculated using the formula:  
    /// `sqrt(abs(GM * a))`  
    /// where:
    /// - `GM` is the gravitational parameter (a.k.a. mu)
    /// - `a` is the semi-major axis of the orbit
    ///
    /// Alternatively, for elliptic orbits, this formula can be used:  
    /// `sqrt(GM * a)`
    ///
    /// As for hyperbolic orbits, this formula can be used:  
    /// `sqrt(-GM * a)`
    ///
    /// ## `altitude`
    /// The altitude at that point in the orbit.  
    /// The altitude is measured in meters, and measured from the
    /// center of the parent body (origin).
    ///
    /// ## `q_mult`
    /// This parameter is a multiplier for the second element in the velocity PQW vector.  
    /// For elliptic orbits, it should be calculated by the formula `sqrt(1 - e^2)`,
    /// where `e` is the eccentricity of the orbit.  
    /// For hyperbolic orbits, it should be calculated by the formula `sqrt(e^2 - 1)`,
    /// where `e` is the eccentricity of the orbit.  
    /// Alternatively, for the general case, you can use the formula `sqrt(abs(1 - e^2))`.
    ///
    /// ## `trig_ecc_anom`
    /// **For elliptic orbits**, this parameter should be a tuple containing the sine and cosine
    /// values of the eccentric anomaly, respectively.  
    /// **For hyperbolic orbits**, this parameter should be a tuple containing the **hyperbolic**
    /// sine and **hyperbolic** cosine values of the eccentric anomaly, respectively.
    ///
    /// ## `sincos_angle`
    /// This parameter should be calculated by passing the true anomaly into sin_cos():
    /// ```
    /// let true_anomaly: f64 = 1.25; // Example value
    /// let sincos_angle = true_anomaly.sin_cos();
    /// ```
    ///
    /// # Performance
    /// This function, by itself, is very performant and should not
    /// be the cause of any performance issues.
    ///
    /// # Parabolic Support
    /// This function doesn't support parabolic trajectories yet.  
    /// `NaN`s or nonsensical values may be returned.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, StateVectors2D};
    ///
    /// # fn main() {
    /// // Elliptic (circular) case
    ///
    /// let orbit = Orbit2D::default();
    /// let true_anomaly: f64 = 1.24; // Example value
    /// let eccentric_anomaly = orbit
    ///     .get_eccentric_anomaly_at_true_anomaly(true_anomaly);
    /// let sqrt_abs_gm_a = (
    ///     orbit.get_gravitational_parameter() *
    ///     orbit.get_semi_major_axis()
    /// ).abs().sqrt();
    /// let altitude = orbit.get_altitude_at_true_anomaly(true_anomaly);
    /// let q_mult = (
    ///     1.0 - orbit.get_eccentricity().powi(2)
    /// ).abs().sqrt();
    /// let trig_ecc_anom = eccentric_anomaly.sin_cos();
    /// let sincos_angle = true_anomaly.sin_cos();
    ///
    /// let sv = StateVectors2D {
    ///     position: orbit.get_position_at_true_anomaly(true_anomaly),
    ///     velocity: orbit.get_velocity_at_true_anomaly(true_anomaly),
    /// };
    ///
    /// let sv2 = orbit.get_state_vectors_from_unchecked_parts(
    ///     sqrt_abs_gm_a,
    ///     altitude,
    ///     q_mult,
    ///     trig_ecc_anom,
    ///     sincos_angle
    /// );
    ///
    /// assert_eq!(sv, sv2);
    /// # }
    /// ```
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, sinhcosh, StateVectors2D};
    ///
    /// # fn main() {
    /// // Hyperbolic case
    ///
    /// let mut orbit = Orbit2D::default();
    /// orbit.set_eccentricity(1.5);
    /// let true_anomaly: f64 = 1.24; // Example value
    /// let eccentric_anomaly = orbit
    ///     .get_eccentric_anomaly_at_true_anomaly(true_anomaly);
    /// let sqrt_abs_gm_a = (
    ///     orbit.get_gravitational_parameter() *
    ///     orbit.get_semi_major_axis()
    /// ).abs().sqrt();
    /// let altitude = orbit.get_altitude_at_true_anomaly(true_anomaly);
    /// let q_mult = (
    ///     1.0 - orbit.get_eccentricity().powi(2)
    /// ).abs().sqrt();
    /// let trig_ecc_anom = sinhcosh(eccentric_anomaly);
    /// let sincos_angle = true_anomaly.sin_cos();
    ///
    /// let sv = StateVectors2D {
    ///     position: orbit.get_position_at_true_anomaly(true_anomaly),
    ///     velocity: orbit.get_velocity_at_true_anomaly(true_anomaly),
    /// };
    ///
    /// let sv2 = orbit.get_state_vectors_from_unchecked_parts(
    ///     sqrt_abs_gm_a,
    ///     altitude,
    ///     q_mult,
    ///     trig_ecc_anom,
    ///     sincos_angle
    /// );
    ///
    /// assert_eq!(sv, sv2);
    /// # }
    /// ```
    fn get_state_vectors_from_unchecked_parts(
        &self,
        sqrt_abs_gm_a: f64,
        altitude: f64,
        q_mult: f64,
        trig_ecc_anom: (f64, f64),
        sincos_angle: (f64, f64),
    ) -> StateVectors2D {
        let outer_mult = sqrt_abs_gm_a / altitude;
        let pqw_velocity =
            self.get_pqw_velocity_at_eccentric_anomaly_unchecked(outer_mult, q_mult, trig_ecc_anom);
        let pqw_position = self.get_pqw_position_at_true_anomaly_unchecked(altitude, sincos_angle);
        let matrix = self.get_transformation_matrix();
        StateVectors2D {
            position: matrix.mul_vec2(pqw_position),
            velocity: matrix.mul_vec2(pqw_velocity),
        }
    }

    /// Gets the time of the orbit at a certain mean anomaly.
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Performance
    /// This function is performant and is unlikely to be the
    /// source of any performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let orbit = Orbit2D::new(
    ///     2.1, // Eccentricity
    ///     5.0, // Periapsis
    ///     2.9, // Argument of periapsis
    ///     1.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter
    /// );
    ///
    /// const TIME: f64 = 2.0;
    ///
    /// let mean_anomaly = orbit.get_mean_anomaly_at_time(TIME);
    /// let time_result = orbit.get_time_at_mean_anomaly(mean_anomaly);
    ///
    /// const TOLERANCE: f64 = 1e-15;
    ///
    /// assert!(
    ///     (time_result - TIME).abs() < TOLERANCE
    /// );
    /// ```
    fn get_time_at_mean_anomaly(&self, mean_anomaly: f64) -> f64 {
        // Per `get_mean_anomaly_at_time`:
        // M = t * sqrt(mu / |a^3|) + M_0
        // => M = t * sqrt(mu / |a^3|) + M_0
        // => M - M_0 = t * sqrt(mu / |a^3|)
        // => t * sqrt(mu / |a^3|) = M - M_0
        // => t = (M - M_0) / sqrt(mu / |a^3|)
        //
        // note: 1 / sqrt(mu / |a^3|) = sqrt(|a^3| / mu)
        //
        // => t = (M - M_0) * sqrt(|a^3| / mu)

        (mean_anomaly - self.get_mean_anomaly_at_epoch())
            * (self.get_semi_major_axis().powi(3).abs() / self.get_gravitational_parameter()).sqrt()
    }

    /// Gets the time of the orbit at a certain eccentric anomaly.
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Performance
    /// This function is not too performant as it performs some trigonometry.  
    /// Alternatively, if you already have the mean anomaly, you can instead use that
    /// along with [`get_time_at_mean_anomaly`][OrbitTrait2D::get_time_at_mean_anomaly].
    /// Or if you know `sin(eccentric_anomaly)` or `sinh(eccentric_anomaly)`
    /// beforehand, you may use
    /// [`get_mean_anomaly_at_elliptic_eccentric_anomaly`][OrbitTrait2D::get_mean_anomaly_at_elliptic_eccentric_anomaly]
    /// and
    /// [`get_mean_anomaly_at_hyperbolic_eccentric_anomaly`][OrbitTrait2D::get_mean_anomaly_at_hyperbolic_eccentric_anomaly]
    /// instead, for closed and hyperbolic orbits respectively.
    ///
    /// Those functions do not do trigonometry and are therefore a lot faster.
    fn get_time_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> f64 {
        self.get_time_at_mean_anomaly(self.get_mean_anomaly_at_eccentric_anomaly(eccentric_anomaly))
    }

    /// Gets the time of the orbit at a certain true anomaly.
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Performance
    /// This function is not too performant as it performs some trigonometry.
    /// Alternatively, if you already have the mean anomaly, you can instead use that
    /// along with [`get_time_at_mean_anomaly`][OrbitTrait2D::get_time_at_mean_anomaly].  
    /// Or if you already have the eccentric anomaly, you can instead use that
    /// along with [`get_time_at_eccentric_anomaly`][OrbitTrait2D::get_time_at_eccentric_anomaly].
    ///
    /// Those functions are faster as they do less trigonometry (and in the case of
    /// the first one: no trig whatsoever).
    fn get_time_at_true_anomaly(&self, true_anomaly: f64) -> f64 {
        self.get_time_at_eccentric_anomaly(self.get_eccentric_anomaly_at_true_anomaly(true_anomaly))
    }

    /// Transforms a position from the perifocal coordinate (PQW) system into
    /// 3D, using the orbital parameters.
    ///
    /// # Perifocal Coordinate (PQW) System
    /// The perifocal coordinate (PQW) system is a frame of reference using
    /// the basis vectors p-hat, q-hat, and w-hat, where p-hat points to the
    /// periapsis, q-hat has a true anomaly 90 degrees more than p-hat, and
    /// w-hat points perpendicular to the orbital plane.
    ///
    /// Learn more: <https://en.wikipedia.org/wiki/Perifocal_coordinate_system>
    ///
    /// # Performance
    /// This function performs 10x faster in the cached version of the
    /// [`Orbit2D`] struct, as it doesn't need to recalculate the transformation
    /// matrix needed to transform 2D vector.
    fn transform_pqw_vector(&self, position: DVec2) -> DVec2 {
        self.get_transformation_matrix().mul_vec2(position)
    }

    /// Gets the eccentricity of the orbit.
    ///
    /// The eccentricity of an orbit is a measure of how much it deviates
    /// from a perfect circle.
    ///
    /// An eccentricity of 0 means the orbit is a perfect circle.  
    /// Between 0 and 1, the orbit is elliptic, and has an oval shape.  
    /// An orbit with an eccentricity of 1 is said to be parabolic.  
    /// If it's greater than 1, the orbit is hyperbolic.
    ///
    /// For hyperbolic trajectories, the higher the eccentricity, the
    /// straighter the path.
    ///
    /// Wikipedia on conic section eccentricity: <https://en.wikipedia.org/wiki/Eccentricity_(mathematics)>  
    /// (Keplerian orbits are conic sections, so the concepts still apply)
    fn get_eccentricity(&self) -> f64;

    /// Gets whether or not the orbit is circular.
    ///
    /// A circular orbit has an eccentricity of 0.
    #[inline]
    fn is_circular(&self) -> bool {
        self.get_eccentricity() == 0.0
    }

    /// Gets whether or not the orbit is strictly elliptic.
    ///
    /// A strictly elliptic orbit has an eccentricity between
    /// 0 and 1.
    ///
    /// For getting whether or not the orbit is circular
    /// or elliptic, use [`is_closed`][OrbitTrait2D::is_closed]
    #[inline]
    fn is_elliptic(&self) -> bool {
        let eccentricity = self.get_eccentricity();

        eccentricity < 1.0 && eccentricity > 0.0
    }

    /// Gets whether or not the orbit is closed.
    ///
    /// A closed orbit can be either circular or elliptic,
    /// i.e. has an eccentricity of less than 1.
    ///
    /// For getting whether or not the orbit is circular (e = 0),
    /// use [`is_circular`][OrbitTrait2D::is_circular].
    ///
    /// For getting whether or not the orbit is strictly
    /// elliptic (0 < e < 1), use [`is_elliptic`][OrbitTrait2D::is_elliptic].
    #[inline]
    fn is_closed(&self) -> bool {
        self.get_eccentricity() < 1.0
    }

    /// Gets whether or not the trajectory is parabolic.
    ///
    /// A parabolic trajectory has an eccentricity of exactly 1.
    #[inline]
    fn is_parabolic(&self) -> bool {
        self.get_eccentricity() == 1.0
    }

    /// Gets whether or not the trajectory is hyperbolic.
    ///
    /// A hyperbolic trajectory has an eccentricity of greater than 1.
    ///
    /// For getting whether or not the trajectory is open — i.e.,
    /// has an eccentricity of *at least* 1 — use
    /// [`is_open`][OrbitTrait2D::is_open].
    #[inline]
    fn is_hyperbolic(&self) -> bool {
        self.get_eccentricity() > 1.0
    }

    /// Gets whether or not the trajectory is open.
    ///
    /// An open trajectory has an eccentricity of at least 1, i.e.,
    /// is either a parabolic trajectory or a hyperbolic trajectory.
    ///
    /// For getting whether or not a trajectory is parabolic (e = 1),
    /// use [`is_parabolic`][OrbitTrait2D::is_parabolic].
    ///
    /// For getting whether or not a trajectory is hyperbolic (e > 1),
    /// use [`is_hyperbolic`][OrbitTrait2D::is_hyperbolic].
    #[inline]
    fn is_open(&self) -> bool {
        self.get_eccentricity() >= 1.0
    }

    /// Sets the eccentricity of the orbit.
    ///
    /// The eccentricity of an orbit is a measure of how much it deviates
    /// from a perfect circle.
    ///
    /// An eccentricity of 0 means the orbit is a perfect circle.  
    /// Between 0 and 1, the orbit is elliptic, and has an oval shape.  
    /// An orbit with an eccentricity of 1 is said to be parabolic.  
    /// If it's greater than 1, the orbit is hyperbolic.
    ///
    /// For hyperbolic trajectories, the higher the eccentricity, the
    /// straighter the path.
    ///
    /// Wikipedia on conic section eccentricity: <https://en.wikipedia.org/wiki/Eccentricity_(mathematics)>  
    /// (Keplerian orbits are conic sections, so the concepts still apply)
    fn set_eccentricity(&mut self, eccentricity: f64);

    /// Gets the periapsis of the orbit.
    ///
    /// The periapsis of an orbit is the distance at the closest point
    /// to the parent body.
    ///
    /// More simply, this is the "minimum altitude" of an orbit.
    ///
    /// Wikipedia: <https://en.wikipedia.org/wiki/Apsis>
    fn get_periapsis(&self) -> f64;

    /// Sets the periapsis of the orbit.
    ///
    /// The periapsis of an orbit is the distance at the closest point
    /// to the parent body.
    ///
    /// More simply, this is the "minimum altitude" of an orbit.
    ///
    /// Wikipedia: <https://en.wikipedia.org/wiki/Apsis>
    fn set_periapsis(&mut self, periapsis: f64);

    /// Gets the argument of periapsis of the orbit in radians.
    ///
    /// Wikipedia:  
    /// The argument of periapsis is the angle from the body's
    /// ascending node to its periapsis, measured in the direction of
    /// motion.  
    /// <https://en.wikipedia.org/wiki/Argument_of_periapsis>
    ///
    /// In simple terms, it tells you how, and in which direction,
    /// the orbit "tilts".
    #[doc(alias = "get_argument_of_periapsis")]
    fn get_arg_pe(&self) -> f64;

    /// Sets the argument of periapsis of the orbit in radians.
    ///
    /// Wikipedia:  
    /// The argument of periapsis is the angle from the body's
    /// ascending node to its periapsis, measured in the direction of
    /// motion.  
    /// <https://en.wikipedia.org/wiki/Argument_of_periapsis>
    ///
    /// In simple terms, it tells you how, and in which direction,
    /// the orbit "tilts".
    #[doc(alias = "set_argument_of_periapsis")]
    fn set_arg_pe(&mut self, arg_pe: f64);

    /// Gets the mean anomaly of the orbit at a certain epoch.
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
    fn get_mean_anomaly_at_epoch(&self) -> f64;

    /// Sets the mean anomaly of the orbit at a certain epoch.
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
    fn set_mean_anomaly_at_epoch(&mut self, mean_anomaly: f64);

    /// Gets the gravitational parameter of the parent body.
    ///
    /// The gravitational parameter mu of the parent body equals a certain
    /// gravitational constant G times the mass of the parent body M.
    ///
    /// In other words, mu = GM.
    #[doc(alias = "get_mu")]
    fn get_gravitational_parameter(&self) -> f64;

    /// Sets the gravitational parameter of the parent body.
    ///
    /// The gravitational parameter mu of the parent body equals a certain
    /// gravitational constant G times the mass of the parent body M.
    ///
    /// In other words, mu = GM.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, MuSetterMode2D};
    ///
    /// let mut orbit = Orbit2D::new(
    ///     0.0, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Argument of Periapsis
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter (mu = GM)
    /// );
    ///
    /// orbit.set_gravitational_parameter(3.0, MuSetterMode2D::KeepElements);
    ///
    /// assert_eq!(orbit.get_eccentricity(), 0.0);
    /// assert_eq!(orbit.get_periapsis(), 1.0);
    /// assert_eq!(orbit.get_arg_pe(), 0.0);
    /// assert_eq!(orbit.get_mean_anomaly_at_epoch(), 0.0);
    /// assert_eq!(orbit.get_gravitational_parameter(), 3.0);
    /// ```
    #[doc(alias = "set_mu")]
    fn set_gravitational_parameter(&mut self, gravitational_parameter: f64, mode: MuSetterMode2D);

    /// Gets the time it takes to complete one revolution of the orbit.
    ///
    /// This function returns infinite values for parabolic trajectories and
    /// NaN for hyperbolic trajectories.
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Performance
    /// This function is performant and is unlikely to be the cause of any
    /// performance issues.
    fn get_orbital_period(&self) -> f64 {
        // T = 2pi * sqrt(a^3 / GM)
        // https://en.wikipedia.org/wiki/Orbital_period
        TAU * (self.get_semi_major_axis().powi(3) / self.get_gravitational_parameter()).sqrt()
    }
}

/// A struct representing a position and velocity at a point in the orbit.
///
/// The position and velocity vectors are two-dimensional.
///
/// The position vector is in meters, while the velocity vector is in
/// meters per second.
///
/// State vectors can be used to form an orbit using
/// [`to_cached_orbit`][Self::to_cached_orbit] or
/// [`to_compact_orbit`][Self::to_compact_orbit].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StateVectors2D {
    /// The 2D position at a point in the orbit, in meters.
    pub position: DVec2,
    /// The 2D velocity at a point in the orbit, in meters per second.
    pub velocity: DVec2,
}

impl StateVectors2D {
    /// Create a new [`CompactOrbit2D`] struct from the state
    /// vectors and the given mu and time values.
    ///
    /// # Mu
    /// Mu is also known as the gravitational parameter, and
    /// is equal to `GM`, where `G` is the gravitational constant,
    /// and `M` is the mass of the parent body.  
    /// It can be described as how strongly the parent body pulls on
    /// the orbiting body.
    ///
    /// Learn more about the gravitational parameter:
    /// <https://en.wikipedia.org/wiki/Standard_gravitational_parameter>
    ///
    /// # Time
    /// The time passed into the function is measured in seconds.
    ///
    /// # Performance
    /// This function is not too performant as it uses several trigonometric operations.  
    ///
    /// For single conversions, this is faster than
    /// [the cached orbit converter][Self::to_cached_orbit].  
    /// However, consider using the cached orbit instead if you want to use the same orbit for
    /// many calculations, as the caching speed benefits should outgrow the small initialization
    /// overhead.
    ///
    /// # Reference Frame
    /// This function expects a state vector where the position's origin (0.0, 0.0, 0.0)
    /// is the center of the parent body.
    ///
    /// # Parabolic Support
    /// This function does not yet support parabolic trajectories.
    /// Non-finite values may be returned for such cases.
    ///
    /// # Constraints
    /// The position must not be at the origin, and the velocity must not be at zero.  
    /// If this constraint is breached, you may get invalid values such as infinities
    /// or NaNs.
    ///
    /// # Examples
    /// Simple use-case:
    /// ```
    /// use keplerian_sim::{CompactOrbit2D, OrbitTrait2D};
    ///
    /// let orbit = CompactOrbit2D::default();
    /// let mu = orbit.get_gravitational_parameter();
    /// let time = 0.0;
    ///
    /// let sv = orbit.get_state_vectors_at_time(time);
    ///
    /// let new_orbit = sv.to_compact_orbit(mu, time);
    ///
    /// assert_eq!(orbit.get_eccentricity(), new_orbit.get_eccentricity());
    /// assert_eq!(orbit.get_periapsis(), new_orbit.get_periapsis());
    /// ```
    /// To simulate an instantaneous 0.1 m/s prograde burn at periapsis:
    /// ```
    /// use keplerian_sim::{CompactOrbit2D, OrbitTrait2D, StateVectors2D};
    /// use glam::DVec2;
    ///
    /// let orbit = CompactOrbit2D::default();
    /// let mu = orbit.get_gravitational_parameter();
    /// let time = 0.0;
    ///
    /// let sv = orbit.get_state_vectors_at_time(time);
    /// assert_eq!(
    ///     sv,
    ///     StateVectors2D {
    ///         position: DVec2::new(1.0, 0.0),
    ///         velocity: DVec2::new(0.0, 1.0),
    ///     }
    /// );
    ///
    /// let new_sv = StateVectors2D {
    ///     velocity: sv.velocity + DVec2::new(0.0, 0.1),
    ///     ..sv
    /// };
    ///
    /// let new_orbit = new_sv.to_compact_orbit(mu, time);
    ///
    /// assert_eq!(
    ///     new_orbit,
    ///     CompactOrbit2D::new(
    ///         0.2100000000000002, // eccentricity
    ///         1.0, // periapsis
    ///         0.0, // argument of periapsis
    ///         0.0, // mean anomaly
    ///         1.0, // gravitational parameter
    ///     )
    /// );
    /// ```
    #[must_use]
    pub fn to_compact_orbit(self, mu: f64, time: f64) -> CompactOrbit2D {
        // Reference:
        // https://orbital-mechanics.space/classical-orbital-elements/orbital-elements-and-the-state-vector.html
        // Adaptations to 2D are used
        // Note: That site doesn't use the same "base elements" and
        // conversions will need to be done at the end

        // Precalculated values
        let altitude = self.position.length();
        let altitude_recip = altitude.recip();
        let position_normal = self.position * altitude_recip;
        let mu_recip = mu.recip();

        // Step 1: Position and Velocity Magnitudes (i.e. speeds)
        // Apparently we don't use the results of this computation,
        // at least for 2D anyway. This section has been removed.

        // Step 2: Orbital Angular Momentum
        // angular_momentum_vector = position × velocity
        // since pos, vel are in 2D, we can guarantee it's in the form:
        // (px, py, 0) × (vx, vy, 0) = (0, 0, momentum)
        // momentum = px * vy - py * vx
        let angular_momentum =
            self.position.x * self.velocity.y - self.position.y * self.velocity.x;

        // Step 3: Inclination
        // Inclination is 0.

        // Step 4: Right Ascension of the Ascending Node
        // RAAN (LAN) is 0.

        // Step 5: Eccentricity
        // eccentricity_vector = velocity × angular_momentum_vector * mu_recip - position_normal
        // Given velocity is in the form (vx, vy, 0) and angular_momentum_vector is in the form
        // (0, 0, momentum), we can simplify:
        // velocity × angular_momentum_vector = (vy * momentum, -vx * momentum, 0)
        // => eccentricity_vector = (vy * momentum, -vx * momentum) * mu_recip - position_normal
        let eccentricity_vector = DVec2::new(
            self.velocity.y * angular_momentum,
            -self.velocity.x * angular_momentum,
        ) * mu_recip
            - position_normal;
        let eccentricity = eccentricity_vector.length();
        let eccentricity_recip = eccentricity.recip();
        let circular = eccentricity < 1e-6;

        // Step 6: Argument of Periapsis
        // In equatorial orbits, argument of periapsis is undefined because
        // the longitude of ascending node is undefined
        // However, the longitude of periapsis is defined.
        // Since longitude of periapsis = argument of periapsis + longitude of ascending node,
        // and we set longitude of ascending node to 0.0,
        // we can just set the argument of periapsis to the longitude of periapsis.

        // ASRI_306 on https://space.stackexchange.com/a/38316/ says (paraphrased):
        //
        //  If it is not circular but equatorial then,
        //      cos(arg_pe_true) = e_x / ||e||
        //  If it is circular but inclined then,
        //      cos(arg_pe) = (n . r) / (||n|| ||r||)
        //  If it is circular and equatorial then,
        //      cos(arg_pe_true) = r_x / ||r||
        //
        // I'm assuming the "arg_pe_true" means the longitude of periapsis,
        // which would be equal to the argument of periapsis when the
        // longitude of ascending node is zero (which it is in this case).
        let arg_pe = match circular {
            false => {
                // Not circular, equatorial: longitude of periapsis = acos(e_x/|e|) with sign from y
                let tmp = (eccentricity_vector.x * eccentricity_recip).acos();

                // Acos only returns values in [0, pi] instead of [0, 2pi]. To recover the
                // full range, we do a sign check on `e.y`, similar to the normal equation earlier,
                // except since the orbit lies on the XY plane we use the Y component instead of Z
                if eccentricity_vector.y >= 0.0 {
                    tmp
                } else {
                    TAU - tmp
                }
            }
            true => {
                // Circular and equatorial
                // This is a very weird case, so we just set it to zero and adjust
                // for this discrepancy in the true anomaly calculation instead.
                0.0
            }
        };

        // Step 7: True anomaly
        // For getting the true anomaly from 2D orbits, we get it manually from the P and Q basis
        // vectors in the PQW coordinate system
        // (see https://en.wikipedia.org/wiki/Perifocal_coordinate_system).
        //
        // Consider this excerpt from the transformation matrix getter from
        // another part of the codebase:
        //
        // matrix.e11 = cos_arg_pe * cos_lan - sin_arg_pe * cos_inc * sin_lan;
        // matrix.e12 = -(sin_arg_pe * cos_lan + cos_arg_pe * cos_inc * sin_lan);
        // matrix.e21 = cos_arg_pe * sin_lan + sin_arg_pe * cos_inc * cos_lan;
        // matrix.e22 = cos_arg_pe * cos_inc * cos_lan - sin_arg_pe * sin_lan;
        // matrix.e31 = sin_arg_pe * sin_inc;
        // matrix.e32 = cos_arg_pe * sin_inc;
        //
        // Note that since inc, lan = 0, we can simplify this:
        //
        // matrix.e11 = cos_arg_pe;
        // matrix.e12 = -sin_arg_pe;
        // matrix.e21 = sin_arg_pe;
        // matrix.e22 = cos_arg_pe;
        //
        // Here, `matrix.e*1` (namely e11, e21, e31) describes the P basis vector,
        // meanwhile `matrix.e*2` describes the Q basis vector.

        let (sin_arg_pe, cos_arg_pe) = arg_pe.sin_cos();

        let p_x = cos_arg_pe;
        let p_y = sin_arg_pe;

        let p = DVec2::new(p_x, p_y);

        let q_x = -sin_arg_pe;
        let q_y = cos_arg_pe;

        let q = DVec2::new(q_x, q_y);

        // Now that we have the P and Q basis vectors (of length 1), we can
        // project our position into the PQW reference frame
        let pos_p = self.position.dot(p);
        let pos_q = self.position.dot(q);

        // Then we can get the angle between the projected position and
        // the +X direction (or technically +P here because it's projected),
        // and since that direction points to the periapsis, that angle
        // is the true anomaly
        let true_anomaly = pos_q.atan2(pos_p).rem_euclid(TAU);

        // Now we convert those elements into our desired form
        // First we need to convert `h` (Orbital Angular Momentum)
        // into periapsis altitude, then we need to convert the true anomaly
        // to a mean anomaly, or to a "time at periapsis" value for parabolic
        // orbits (not implemented yet).
        // TODO: PARABOLIC SUPPORT: Implement that "time at periapsis" value

        // Part 1: Converting orbital angular momentum into periapsis altitude
        //
        // https://faculty.fiu.edu/~vanhamme/ast3213/orbits.pdf says:
        // r = (h^2 / mu) / (1 + e * cos(theta))
        // ...where:
        // r = altitude at a certain true anomaly in the orbit (theta)
        // h = the orbital angular momentum scalar vale
        // mu = the gravitational parameter
        // e = the eccentricity
        // theta = the true anomaly
        //
        // https://en.wikipedia.org/wiki/True_anomaly says:
        // [The true anomaly] is the angle between the direction of
        // periapsis and the current position of the body [...]
        //
        // This means that a true anomaly of zero means periapsis.
        // So if we substitute zero into theta into the earlier equation...
        //
        // r = (h^2 / mu) / (1 + e * cos(0))
        //   = (h^2 / mu) / (1 + e * 1)
        //   = (h^2 / mu) / (1 + e)
        //
        // This gives us the altitude at true anomaly 0 (periapsis).
        let periapsis = (angular_momentum.powi(2) * mu_recip) / (1.0 + eccentricity);

        // Part 2: converting true anomaly to mean anomaly
        // We first convert it to an eccentric anomaly:
        let eccentric_anomaly = if eccentricity < 1.0 {
            // let v = true_anomaly,
            //   e = eccentricity,
            //   E = eccentric anomaly
            //
            // https://en.wikipedia.org/wiki/True_anomaly#From_the_eccentric_anomaly:
            // tan(v / 2) = sqrt((1 + e)/(1 - e)) * tan(E / 2)
            // 1 = sqrt((1 + e)/(1 - e)) * tan(E / 2) / tan(v / 2)
            // 1 / tan(E / 2) = sqrt((1 + e)/(1 - e)) / tan(v / 2)
            // tan(E / 2) = tan(v / 2) / sqrt((1 + e)/(1 - e))
            // E / 2 = atan(tan(v / 2) / sqrt((1 + e)/(1 - e)))
            // E = 2 * atan(tan(v / 2) / sqrt((1 + e)/(1 - e)))
            // E = 2 * atan(tan(v / 2) * sqrt((1 - e)/(1 + e)))

            2.0 * ((true_anomaly * 0.5).tan()
                * ((1.0 - eccentricity) / (1.0 + eccentricity)).sqrt())
            .atan()
        } else {
            // From the presentation "Spacecraft Dynamics and Control"
            // by Matthew M. Peet
            // https://control.asu.edu/Classes/MAE462/462Lecture05.pdf
            // Slide 25 of 27
            // Section "The Method for Hyperbolic Orbits"
            //
            // tan(f/2) = sqrt((e+1)/(e-1))*tanh(H/2)
            // 1 / tanh(H/2) = sqrt((e+1)/(e-1)) / tan(f/2)
            // tanh(H/2) = tan(f/2) / sqrt((e+1)/(e-1))
            // tanh(H/2) = tan(f/2) * sqrt((e-1)/(e+1))
            // H/2 = atanh(tan(f/2) * sqrt((e-1)/(e+1)))
            // H = 2 atanh(tan(f/2) * sqrt((e-1)/(e+1)))
            2.0 * ((true_anomaly * 0.5).tan()
                * ((eccentricity - 1.0) / (eccentricity + 1.0)).sqrt())
            .atanh()
        };

        // Then use Kepler's Equation to convert eccentric anomaly
        // to mean anomaly:
        let mean_anomaly = if eccentricity < 1.0 {
            // https://en.wikipedia.org/wiki/Kepler%27s_equation#Equation
            //
            //      M = E - e sin E
            //
            // where:
            //   M = mean anomaly
            //   E = eccentric anomaly
            //   e = eccentricity
            eccentric_anomaly - eccentricity * eccentric_anomaly.sin()
        } else {
            // https://en.wikipedia.org/wiki/Kepler%27s_equation#Hyperbolic_Kepler_equation
            //
            //      M = e sinh(H) - H
            //
            // where:
            //   M = mean anomaly
            //   e = eccentricity
            //   H = hyperbolic eccentric anomaly
            eccentricity * eccentric_anomaly.sinh() - eccentric_anomaly
        };

        // We now have the actual mean anomaly (`M`) at the current time (`t`)
        // We want to get the mean anomaly *at epoch* (`M_0`)
        // This means offsetting the mean anomaly using the given current timestamp (`t`)
        //
        // M = t * sqrt(mu / |a^3|) + M_0
        // M - M_0 = t * sqrt(mu / |a^3|)
        // -M_0 = t * sqrt(mu / |a^3|) - M
        // M_0 = M - t * sqrt(mu / |a^3|)
        //
        // a = r_p / (1 - e)
        let semi_major_axis: f64 = periapsis / (1.0 - eccentricity);
        let offset = time * (mu / semi_major_axis.powi(3).abs()).sqrt();
        let mean_anomaly_at_epoch = mean_anomaly - offset;
        let mean_anomaly_at_epoch = if eccentricity < 1.0 {
            mean_anomaly_at_epoch.rem_euclid(TAU)
        } else {
            mean_anomaly_at_epoch
        };

        CompactOrbit2D::new(eccentricity, periapsis, arg_pe, mean_anomaly_at_epoch, mu)
    }

    /// Create a new [`Orbit2D`] struct from the state
    /// vectors and a given mu value.
    ///
    /// # Mu
    /// Mu is also known as the gravitational parameter, and
    /// is equal to `GM`, where `G` is the gravitational constant,
    /// and `M` is the mass of the parent body.  
    /// It can be described as how strongly the parent body pulls on
    /// the orbiting body.
    ///
    /// Learn more about the gravitational parameter:
    /// <https://en.wikipedia.org/wiki/Standard_gravitational_parameter>
    ///
    /// # Time
    /// The time passed into the function is measured in seconds.
    ///
    /// # Performance
    /// This function is not too performant as it uses several trigonometric operations.  
    ///
    /// For single conversions, this is slower than
    /// [the compact orbit converter][Self::to_compact_orbit], as there are some extra
    /// values that will be calculated and cached.  
    /// However, if you're going to use this same orbit for many calculations, this should
    /// be better off in the long run as the caching performance benefits should outgrow
    /// the small initialization cost.
    ///
    /// # Reference Frame
    /// This function expects a state vector where the position's origin (0.0, 0.0, 0.0)
    /// is the center of the parent body.
    ///
    /// # Parabolic Support
    /// This function does not yet support parabolic trajectories.
    /// Non-finite values may be returned for such cases.
    ///
    /// # Constraints
    /// The position must not be at the origin, and the velocity must not be at zero.  
    /// If this constraint is breached, you may get invalid values such as infinities
    /// or NaNs.
    ///
    /// # Examples
    /// Simple use-case:
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D};
    ///
    /// let orbit = Orbit2D::default();
    /// let mu = orbit.get_gravitational_parameter();
    /// let time = 0.0;
    ///
    /// let sv = orbit.get_state_vectors_at_time(time);
    ///
    /// let new_orbit = sv.to_cached_orbit(mu, time);
    ///
    /// assert_eq!(orbit.get_eccentricity(), new_orbit.get_eccentricity());
    /// assert_eq!(orbit.get_periapsis(), new_orbit.get_periapsis());
    /// ```
    /// To simulate an instantaneous 0.1 m/s prograde burn at periapsis:
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, StateVectors2D};
    /// use glam::DVec2;
    ///
    /// let orbit = Orbit2D::default();
    /// let mu = orbit.get_gravitational_parameter();
    /// let time = 0.0;
    ///
    /// let sv = orbit.get_state_vectors_at_time(time);
    /// assert_eq!(
    ///     sv,
    ///     StateVectors2D {
    ///         position: DVec2::new(1.0, 0.0),
    ///         velocity: DVec2::new(0.0, 1.0),
    ///     }
    /// );
    ///
    /// let new_sv = StateVectors2D {
    ///     velocity: sv.velocity + DVec2::new(0.0, 0.1),
    ///     ..sv
    /// };
    ///
    /// let new_orbit = new_sv.to_cached_orbit(mu, time);
    ///
    /// assert_eq!(
    ///     new_orbit,
    ///     Orbit2D::new(
    ///         0.2100000000000002, // eccentricity
    ///         1.0, // periapsis
    ///         0.0, // argument of periapsis
    ///         0.0, // mean anomaly
    ///         1.0, // gravitational parameter
    ///     )
    /// )
    /// ```
    #[must_use]
    pub fn to_cached_orbit(self, mu: f64, time: f64) -> Orbit2D {
        self.to_compact_orbit(mu, time).into()
    }

    /// Create a new custom orbit struct from the state vectors
    /// and a given mu value.
    ///
    /// # Mu
    /// Mu is also known as the gravitational parameter, and
    /// is equal to `GM`, where `G` is the gravitational constant,
    /// and `M` is the mass of the parent body.  
    /// It can be described as how strongly the parent body pulls on
    /// the orbiting body.
    ///
    /// Learn more about the gravitational parameter:
    /// <https://en.wikipedia.org/wiki/Standard_gravitational_parameter>
    ///
    /// # Time
    /// The time passed into the function is measured in seconds.
    ///
    /// # Performance
    /// This function is not too performant as it uses several trigonometric operations.
    ///
    /// The performance also depends on how fast the specified orbit type can convert
    /// between the [`CompactOrbit2D`] form into itself, and so we cannot guarantee any
    /// performance behaviors.
    ///
    /// # Reference Frame
    /// This function expects a state vector where the position's origin (0.0, 0.0, 0.0)
    /// is the center of the parent body.
    ///
    /// # Parabolic Support
    /// This function does not yet support parabolic trajectories.
    /// Non-finite values may be returned for such cases.
    ///
    /// # Constraints
    /// The position must not be at the origin, and the velocity must not be at zero.  
    /// If this constraint is breached, you may get invalid values such as infinities
    /// or NaNs.
    #[must_use]
    pub fn to_custom_orbit<O>(self, mu: f64, time: f64) -> O
    where
        O: From<CompactOrbit2D> + OrbitTrait2D,
    {
        self.to_compact_orbit(mu, time).into()
    }
}

/// A mode to describe how the gravitational parameter setter should behave.
///
/// This is used to describe how the setter should behave when setting the
/// gravitational parameter of the parent body.
///
/// # Which mode should I use?
/// The mode you should use depends on what you expect from setting the mu value
/// to a different value.
///
/// If you just want to set the mu value naïvely (without touching the
/// other orbital elements), you can use the `KeepElements` variant.
///
/// If this is part of a simulation and you want to keep the current position
/// (not caring about the velocity), you can use the `KeepPositionAtTime` variant.
///
/// If you want to keep the current position and velocity, you can use either
/// the `KeepKnownStateVectors` or `KeepStateVectorsAtTime` modes, the former
/// being more performant if you already know the state vectors beforehand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MuSetterMode2D {
    /// Keep all the other orbital parameters the same.
    ///
    /// **This will change the position and velocity of the orbiting body abruptly,
    /// if you use the time-based functions.** It will not, however, change the trajectory
    /// of the orbit.
    ///
    /// # Performance
    /// This mode is the fastest of the mu setter modes as it is simply an
    /// assignment operation.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, MuSetterMode2D};
    ///
    /// let mut orbit = Orbit2D::new(
    ///     0.0, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Argument of Periapsis
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter (mu = GM)
    /// );
    ///
    /// orbit.set_gravitational_parameter(3.0, MuSetterMode2D::KeepElements);
    ///
    /// assert_eq!(orbit.get_eccentricity(), 0.0);
    /// assert_eq!(orbit.get_periapsis(), 1.0);
    /// assert_eq!(orbit.get_arg_pe(), 0.0);
    /// assert_eq!(orbit.get_mean_anomaly_at_epoch(), 0.0);
    /// assert_eq!(orbit.get_gravitational_parameter(), 3.0);
    /// ```
    KeepElements,
    /// Keep the overall shape of the orbit, but modify the mean anomaly at epoch
    /// such that the position at the given time is the same.
    ///
    /// **This will change the velocity of the orbiting body abruptly, if you use
    /// the time-based position/velocity getter functions.**
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Performance
    /// This mode is slower than the `KeepElements` mode as it has to compute a new
    /// mean anomaly at epoch. However, this isn't too expensive and only costs a
    /// few squareroot operations.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, MuSetterMode2D};
    ///
    /// let mut orbit = Orbit2D::new(
    ///     0.0, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Argument of Periapsis
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter (mu = GM)
    /// );
    ///
    /// orbit.set_gravitational_parameter(
    ///     3.0,
    ///     MuSetterMode2D::KeepPositionAtTime(0.4),
    /// );
    ///
    /// assert_eq!(orbit.get_eccentricity(), 0.0);
    /// assert_eq!(orbit.get_periapsis(), 1.0);
    /// assert_eq!(orbit.get_arg_pe(), 0.0);
    /// assert_eq!(orbit.get_mean_anomaly_at_epoch(), -0.2928203230275509);
    /// assert_eq!(orbit.get_gravitational_parameter(), 3.0);
    /// ```
    KeepPositionAtTime(f64),
    /// Keep the position and velocity of the orbit at a certain time
    /// roughly unchanged, using known [StateVectors2D] to avoid
    /// duplicate calculations.
    ///
    /// **This will change the orbit's overall trajectory.**
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Unchecked Operation
    /// This mode does not check whether or not the state vectors and time values given
    /// match up. Mismatched values may result in undesired behavior and NaNs.  
    /// Use the [`KeepStateVectorsAtTime`][MuSetterMode2D::KeepStateVectorsAtTime]
    /// mode if you don't want this unchecked operation.
    ///
    /// # Performance
    /// This mode uses some trigonometry, and therefore is not very performant.  
    /// Consider using another mode if performance is an issue.  
    ///
    /// This is, however, significantly more performant than the numerical approach
    /// used in the [`KeepStateVectorsAtTime`][MuSetterMode2D::KeepStateVectorsAtTime]
    /// mode.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, MuSetterMode2D};
    ///
    /// let mut orbit = Orbit2D::new(
    ///     0.0, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Argument of Periapsis
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter (mu = GM)
    /// );
    ///
    /// let time = 0.75;
    ///
    /// let state_vectors = orbit.get_state_vectors_at_time(time);
    ///
    /// orbit.set_gravitational_parameter(
    ///     3.0,
    ///     MuSetterMode2D::KeepKnownStateVectors {
    ///         state_vectors,
    ///         time
    ///     }
    /// );
    ///
    /// let new_state_vectors = orbit.get_state_vectors_at_time(time);
    ///
    /// println!("Old state vectors: {state_vectors:?}");
    /// println!("New state vectors: {new_state_vectors:?}");
    /// ```
    KeepKnownStateVectors {
        /// The state vectors describing the point in the orbit where you want the
        /// position and velocity to remain the same.
        ///
        /// Must correspond to the time value, or undesired behavior and NaNs may occur.
        state_vectors: StateVectors2D,

        /// The time value of the point in the orbit
        /// where you want the position and velocity to remain the same.
        ///
        /// The time is measured in seconds.
        ///
        /// Must correspond to the given state vectors, or undesired behavior and NaNs may occur.
        time: f64,
    },
    /// Keep the position and velocity of the orbit at a certain time
    /// roughly unchanged.
    ///
    /// **This will change the orbit's overall trajectory.**
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Performance
    /// This mode uses numerical approach methods, and therefore is not performant.  
    /// Consider using another mode if performance is an issue.  
    ///
    /// Alternatively, if you already know the state vectors (position and velocity)
    /// of the point you want to keep, use the
    /// [`KeepKnownStateVectors`][MuSetterMode2D::KeepKnownStateVectors]
    /// mode instead. This skips the numerical method used to obtain the eccentric anomaly
    /// and some more trigonometry.
    ///
    /// If you only know the eccentric anomaly and true anomaly, it's more performant
    /// to derive state vectors from those first and then use the aforementioned
    /// [`KeepKnownStateVectors`][MuSetterMode2D::KeepKnownStateVectors] mode. This can
    /// be done using the [`Orbit2D::get_state_vectors_at_eccentric_anomaly`] function, for
    /// example.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, MuSetterMode2D};
    ///
    /// let old_orbit = Orbit2D::new(
    ///     0.0, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Argument of Periapsis
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter (mu = GM)
    /// );
    ///
    /// let mut new_orbit = old_orbit.clone();
    ///
    /// const TIME: f64 = 1.5;
    ///
    /// new_orbit.set_gravitational_parameter(
    ///     3.0,
    ///     MuSetterMode2D::KeepStateVectorsAtTime(TIME)
    /// );
    ///
    /// let old_state_vectors = old_orbit.get_state_vectors_at_time(TIME);
    /// let new_state_vectors = new_orbit.get_state_vectors_at_time(TIME);
    ///
    /// println!("Old state vectors: {old_state_vectors:?}");
    /// println!("New state vectors: {new_state_vectors:?}");
    /// ```
    KeepStateVectorsAtTime(f64),
}
