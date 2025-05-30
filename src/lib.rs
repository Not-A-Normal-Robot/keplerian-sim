//! # Keplerian Orbital Mechanics
//! This library crate contains logic for Keplerian orbits, similar to the ones
//! you'd find in a game like Kerbal Space Program.  
//!
//! Keplerian orbits are special in that they are more stable and predictable than
//! Newtonian orbits. In fact, unlike Newtonian orbits, Keplerian orbits don't use
//! time steps to calculate the next position of an object. Keplerian orbits use
//! state vectors to determine the object's *full trajectory* at any given time.  
//! This way, you don't need to worry about lag destabilizing Keplerian orbits.  
//!
//! However, Keplerian orbits are significantly more complex to calculate than
//! just using Newtonian physics. It's also a two-body simulation, meaning that
//! it doesn't account for external forces like gravity from other bodies or the
//! engines of a spacecraft.
//!
//! The way Kerbal Space Program handles this is to have an "on-rails" physics
//! system utilizing Keplerian orbits, and an "active" physics system utilizing
//! Newtonian two-body physics.
//!
//! ## Getting started
//! This crate provides four main structs:
//! - [`Orbit`]: A struct representing an orbit around a celestial body.
//!   Each instance of this struct has some cached data to speed up
//!   certain calculations, and has a larger memory footprint.
//! - [`CompactOrbit`]: A struct representing an orbit around a celestial body.
//!   This struct has a smaller memory footprint than the regular `Orbit` struct,
//!   but some calculations may take 2~10x slower because it doesn't save any
//!   cached calculations.
//! - [`Body`]: A struct representing a celestial body. This struct contains
//!   information about the body's mass, radius, and orbit.
//! - [`Universe`]: A struct representing the entire simulation. This struct
//!   contains a list of all the bodies in the simulation, and can calculate
//!   the absolute position of any body at any given time.
//!   To do this, it stores parent-child relationships between bodies.
//!
//! We also provide a [`body_presets`] module, which contains some preset celestial
//! bodies to use in your simulation. It contains many celestial bodies, like
//! the Sun, the Moon, and all the planets in the Solar System.
//!
//! ## Example
//!
//! ```rust
//! use keplerian_sim::{Orbit, OrbitTrait};
//!
//! # fn main() {
//! // Create a perfectly circular orbit with a radius of 1 meter
//! let orbit = Orbit::new_default();
//! assert_eq!(orbit.get_position_at_time(0.0), (1.0, 0.0, 0.0));
//! # }
//! #
//! ```

#![warn(missing_docs)]

mod body;
pub mod body_presets;
mod cached_orbit;
mod compact_orbit;
mod universe;

pub use body::Body;
pub use cached_orbit::Orbit;
pub use compact_orbit::CompactOrbit;
pub use universe::Universe;

/// A struct representing a 3x2 matrix.
///
/// This struct is used to store the transformation matrix
/// for transforming a 2D vector into a 3D vector.
///
/// Namely, it is used in the [`tilt_flat_position`][OrbitTrait::tilt_flat_position]
/// method to tilt a 2D position into 3D, using the orbital parameters.
///
/// Each element is named `eXY`, where `X` is the row and `Y` is the column.
///
/// # Example
/// ```
/// use keplerian_sim::Matrix3x2;
///
/// let matrix: Matrix3x2<f64> = Matrix3x2 {
///    e11: 1.0, e12: 0.0,
///    e21: 0.0, e22: 1.0,
///    e31: 0.0, e32: 0.0,
/// };
///
/// let vec = (1.0, 2.0);
///
/// let result = matrix.dot_vec(vec);
///
/// assert_eq!(result, (1.0, 2.0, 0.0));
/// ```
#[allow(missing_docs)]
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix3x2<T> {
    // Element XY
    pub e11: T,
    pub e12: T,
    pub e21: T,
    pub e22: T,
    pub e31: T,
    pub e32: T,
}

/// A struct representing 3D orbital state vectors.  
///
/// This struct consists of a 3D position vector and a 3D velocity vector.
///
/// This represents the position and velocity of a certain point in the orbit.
///
/// # Example
/// ```
/// use keplerian_sim::StateVectors3D;
///
/// let v1 = StateVectors3D::ZERO;
/// let v2 = StateVectors3D {
///     position: (0.0, 0.0, 0.0),
///     velocity: (0.0, 0.0, 0.0)
/// };
///
/// assert_eq!(v1, v2);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StateVectors3D {
    /// A 3D position vector.
    pub position: Vec3,

    /// A 3D velocity vector.
    pub velocity: Vec3,
}

impl StateVectors3D {
    pub const ZERO: StateVectors3D = StateVectors3D {
        position: (0.0, 0.0, 0.0),
        velocity: (0.0, 0.0, 0.0),
    };
}

impl<T: Copy> Copy for Matrix3x2<T> {}
impl<T: Eq> Eq for Matrix3x2<T> {}

impl<T: Copy> Matrix3x2<T> {
    /// Create a new Matrix3x2 instance where each
    /// element is initialized with the same value.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::Matrix3x2;
    ///
    /// let matrix = Matrix3x2::filled_with(0.0);
    ///
    /// assert_eq!(matrix, Matrix3x2 {
    ///    e11: 0.0, e12: 0.0,
    ///    e21: 0.0, e22: 0.0,
    ///    e31: 0.0, e32: 0.0,
    /// });
    /// ```
    pub fn filled_with(element: T) -> Matrix3x2<T> {
        return Matrix3x2 {
            e11: element,
            e12: element,
            e21: element,
            e22: element,
            e31: element,
            e32: element,
        };
    }
}

impl<T> Matrix3x2<T>
where
    T: Copy + core::ops::Mul<Output = T> + core::ops::Add<Output = T>,
{
    /// Computes a dot product between this matrix and a 2D vector.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::Matrix3x2;
    ///
    /// let matrix: Matrix3x2<f64> = Matrix3x2 {
    ///     e11: 1.0, e12: 0.0,
    ///     e21: 0.0, e22: 1.0,
    ///     e31: 1.0, e32: 1.0,
    /// };
    ///
    /// let vec = (1.0, 2.0);
    ///
    /// let result = matrix.dot_vec(vec);
    ///
    /// assert_eq!(result, (1.0, 2.0, 3.0));
    /// ```
    pub fn dot_vec(&self, vec: (T, T)) -> (T, T, T) {
        return (
            vec.0 * self.e11 + vec.1 * self.e12,
            vec.0 * self.e21 + vec.1 * self.e22,
            vec.0 * self.e31 + vec.1 * self.e32,
        );
    }
}

type Vec3 = (f64, f64, f64);
type Vec2 = (f64, f64);

/// A trait that defines the methods that a Keplerian orbit must implement.
///
/// This trait is implemented by both [`Orbit`] and [`CompactOrbit`].
///
/// # Examples
/// ```
/// use keplerian_sim::{Orbit, OrbitTrait, CompactOrbit};
///
/// fn accepts_orbit(orbit: &impl OrbitTrait) {
///     println!("That's an orbit!");
/// }
///
/// fn main() {
///     let orbit = Orbit::new_default();
///     accepts_orbit(&orbit);
///
///     let compact = CompactOrbit::new_default();
///     accepts_orbit(&compact);
/// }
/// ```
///
/// This example will fail to compile:
///
/// ```compile_fail
/// # use keplerian_sim::{Orbit, OrbitTrait, CompactOrbit};
/// #
/// # fn accepts_orbit(orbit: &impl OrbitTrait) {
/// #     println!("That's an orbit!");
/// # }
/// #
/// # fn main() {
/// #     let orbit = Orbit::new_default();
/// #     accepts_orbit(&orbit);
/// #  
/// #     let compact = CompactOrbit::new_default();
/// #     accepts_orbit(&compact);
/// let not_orbit = (0.0, 1.0);
/// accepts_orbit(&not_orbit);
/// # }
/// ```
pub trait OrbitTrait {
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
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
    /// orbit.set_periapsis(50.0);
    /// orbit.set_apoapsis_force(100.0);
    /// let sma = orbit.get_semi_major_axis();
    /// let expected = 75.0;
    /// assert!((sma - expected).abs() < 1e-6);
    /// ```
    fn get_semi_major_axis(&self) -> f64;

    /// Gets the semi-minor axis of the orbit.
    ///
    /// In an elliptic orbit, the semi-minor axis is half of the maximum "width"
    /// of the orbit.
    ///
    /// Learn more: <https://en.wikipedia.org/wiki/Semi-major_and_semi-minor_axes>
    fn get_semi_minor_axis(&self) -> f64;

    /// Gets the semi-latus rectum of the orbit.
    ///
    /// Learn more: <https://en.wikipedia.org/wiki/Ellipse#Semi-latus_rectum>  
    /// <https://en.wikipedia.org/wiki/Conic_section#Conic_parameters>
    fn get_semi_latus_rectum(&self) -> f64;

    /// Gets the linear eccentricity of the orbit, in meters.
    ///
    /// In an elliptic orbit, the linear eccentricity is the distance
    /// between its center and either of its two foci (focuses).
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
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
    fn get_linear_eccentricity(&self) -> f64;

    /// Gets the apoapsis of the orbit.  
    /// Returns infinity for parabolic orbits.  
    /// Returns negative values for hyperbolic orbits.  
    ///
    /// # Examples
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
    /// orbit.set_eccentricity(0.5); // Elliptic
    /// assert!(orbit.get_apoapsis() > 0.0);
    ///
    /// orbit.set_eccentricity(1.0); // Parabolic
    /// assert!(orbit.get_apoapsis().is_infinite());
    ///
    /// orbit.set_eccentricity(2.0); // Hyperbolic
    /// assert!(orbit.get_apoapsis() < 0.0);
    /// ```
    fn get_apoapsis(&self) -> f64;

    /// Sets the apoapsis of the orbit.  
    /// Errors when the apoapsis is less than the periapsis, or less than zero.  
    /// If you want a setter that does not error, use `set_apoapsis_force`, which will
    /// try its best to interpret what you might have meant, but may have
    /// undesirable behavior.
    ///
    /// # Examples
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
    /// orbit.set_periapsis(50.0);
    ///
    /// assert!(
    ///     orbit.set_apoapsis(100.0)
    ///         .is_ok()
    /// );
    ///
    /// let result = orbit.set_apoapsis(25.0);
    /// assert!(result.is_err());
    /// assert!(
    ///     result.unwrap_err() ==
    ///     keplerian_sim::ApoapsisSetterError::ApoapsisLessThanPeriapsis
    /// );
    ///
    /// let result = orbit.set_apoapsis(-25.0);
    /// assert!(result.is_err());
    /// assert!(
    ///     result.unwrap_err() ==
    ///     keplerian_sim::ApoapsisSetterError::ApoapsisNegative
    /// );
    /// ```
    fn set_apoapsis(&mut self, apoapsis: f64) -> Result<(), ApoapsisSetterError>;

    /// Sets the apoapsis of the orbit, with a best-effort attempt at interpreting
    /// possibly-invalid values.  
    /// This function will not error, but may have undesirable behavior:
    /// - If the given apoapsis is less than the periapsis but more than zero,
    ///   the orbit will be flipped and the periapsis will be set to the given apoapsis.
    /// - If the given apoapsis is less than zero, the orbit will be hyperbolic
    ///   instead.
    ///
    /// If these behaviors are undesirable, consider creating a custom wrapper around
    /// `set_eccentricity` instead.
    fn set_apoapsis_force(&mut self, apoapsis: f64);

    /// Gets the transformation matrix needed to tilt a 2D vector into the
    /// tilted orbital plane.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let orbit = Orbit::new_default();
    /// let matrix = orbit.get_transformation_matrix();
    ///
    /// assert_eq!(matrix, keplerian_sim::Matrix3x2 {
    ///     e11: 1.0, e12: 0.0,
    ///     e21: 0.0, e22: 1.0,
    ///     e31: 0.0, e32: 0.0,
    /// });
    /// ```
    fn get_transformation_matrix(&self) -> Matrix3x2<f64>;

    /// Gets the eccentric anomaly at a given mean anomaly in the orbit.
    ///
    /// The method to get the eccentric anomaly often uses numerical
    /// methods like Newton's method, and so it is not very performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// When the orbit is open (has an eccentricity of at least 1),
    /// the [hyperbolic eccentric anomaly](https://space.stackexchange.com/questions/27602/what-is-hyperbolic-eccentric-anomaly-f)
    /// would be returned instead.
    ///
    /// The eccentric anomaly is an angular parameter that defines the position
    /// of a body that is moving along an elliptic Kepler orbit.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Eccentric_anomaly)
    fn get_eccentric_anomaly(&self, mean_anomaly: f64) -> f64;

    // TODO: Improve documentation on this
    // namely: is this performant?
    /// Gets the eccentric anomaly at a given true anomaly in the orbit.
    ///
    /// When the orbit is open (has an eccentricity of at least 1),
    /// the [hyperbolic eccentric anomaly](https://space.stackexchange.com/questions/27602/what-is-hyperbolic-eccentric-anomaly-f)
    /// would be returned instead.
    ///
    /// The eccentric anomaly is an angular parameter that defines the position
    /// of a body that is moving along an elliptic Kepler orbit.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Eccentric_anomaly)
    fn get_eccentric_anomaly_at_true_anomaly(&self, true_anomaly: f64) -> f64;

    /// Gets the eccentric anomaly at a given time in the orbit.
    ///
    /// The method to get the eccentric anomaly often uses numerical
    /// methods like Newton's method, and so it is not very performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// When the orbit is open (has an eccentricity of at least 1),
    /// the [hyperbolic eccentric anomaly](https://space.stackexchange.com/questions/27602/what-is-hyperbolic-eccentric-anomaly-f)
    /// would be returned instead.
    ///
    /// The eccentric anomaly is an angular parameter that defines the position
    /// of a body that is moving along an elliptic Kepler orbit.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Eccentric_anomaly)
    fn get_eccentric_anomaly_at_time(&self, t: f64) -> f64 {
        self.get_eccentric_anomaly(self.get_mean_anomaly_at_time(t))
    }

    /// Gets the true anomaly at a given eccentric anomaly in the orbit.
    ///
    /// This function is faster than the function which takes mean anomaly as input,
    /// as the eccentric anomaly is hard to calculate.
    ///
    /// This function returns +/- pi for parabolic orbits due to how the equation works,
    /// and so **may result in infinities when combined with other functions**.
    fn get_true_anomaly_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> f64;

    /// Gets the true anomaly at a given mean anomaly in the orbit.
    ///
    /// The true anomaly is derived from the eccentric anomaly, which
    /// uses numerical methods and so is not very performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// The true anomaly is the angle between the direction of periapsis
    /// and the current position of the body, as seen from the main focus
    /// of the ellipse.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/True_anomaly)
    ///
    /// This function returns +/- pi for parabolic orbits due to how the equation works,
    /// and so **may result in infinities when combined with other functions**.
    fn get_true_anomaly(&self, mean_anomaly: f64) -> f64 {
        self.get_true_anomaly_at_eccentric_anomaly(self.get_eccentric_anomaly(mean_anomaly))
    }

    /// Gets the true anomaly at a given time in the orbit.
    ///
    /// The true anomaly is derived from the eccentric anomaly, which
    /// uses numerical methods and so is not very performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// The true anomaly is the angle between the direction of periapsis
    /// and the current position of the body, as seen from the main focus
    /// of the ellipse.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/True_anomaly)
    ///
    /// This function returns +/- pi for parabolic orbits due to how the equation works,
    /// and so **may result in infinities when combined with other functions**.
    fn get_true_anomaly_at_time(&self, t: f64) -> f64 {
        self.get_true_anomaly(self.get_mean_anomaly_at_time(t))
    }

    /// Gets the mean anomaly at a given time in the orbit.
    ///
    /// The mean anomaly is the fraction of an elliptical orbit's period
    /// that has elapsed since the orbiting body passed periapsis,
    /// expressed as an angle which can be used in calculating the position
    /// of that body in the classical two-body problem.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Mean_anomaly)
    fn get_mean_anomaly_at_time(&self, t: f64) -> f64;

    /// Gets the mean anomaly at a given eccentric anomaly in the orbit.
    ///
    /// # Performance
    /// This function is a wrapper around
    /// [`get_mean_anomaly_at_elliptic_eccentric_anomaly`][Self::get_mean_anomaly_at_elliptic_eccentric_anomaly]
    /// and
    /// [`get_mean_anomaly_at_hyperbolic_eccentric_anomaly`][Self::get_mean_anomaly_at_hyperbolic_eccentric_anomaly].  
    /// It does some trigonometry, but if you know `sin(eccentric_anomaly)` or `sinh(eccentric_anomaly)`
    /// beforehand, this can be skipped by directly using those inner functions.
    ///
    /// The mean anomaly is the fraction of an elliptical orbit's period
    /// that has elapsed since the orbiting body passed periapsis,
    /// expressed as an angle which can be used in calculating the position
    /// of that body in the classical two-body problem.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Mean_anomaly)
    fn get_mean_anomaly_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> f64 {
        // TODO: PARABOLA SUPPORT: This function doesn't consider parabolas yet.
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
    /// # Unchecked Operation
    /// This function does no checks on the validity of the value given
    /// as `sin_eccentric_anomaly`. It also doesn't check if the orbit is elliptic.  
    /// If invalid values are passed in, you will receive a possibly-nonsensical value as output.  
    ///
    /// The mean anomaly is the fraction of an elliptical orbit's period
    /// that has elapsed since the orbiting body passed periapsis,
    /// expressed as an angle which can be used in calculating the position
    /// of that body in the classical two-body problem.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Mean_anomaly)
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
    /// # Unchecked Operation
    /// This function does no checks on the validity of the value given
    /// as `sinh_eccentric_anomaly`. It also doesn't check if the orbit is hyperbolic.  
    /// If invalid values are passed in, you will receive a possibly-nonsensical value as output.  
    ///
    /// The mean anomaly is the fraction of an elliptical orbit's period
    /// that has elapsed since the orbiting body passed periapsis,
    /// expressed as an angle which can be used in calculating the position
    /// of that body in the classical two-body problem.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Mean_anomaly)
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
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Mean_anomaly)
    fn get_mean_anomaly_at_true_anomaly(&self, true_anomaly: f64) -> f64 {
        let ecc_anom = self.get_eccentric_anomaly_at_true_anomaly(true_anomaly);
        self.get_mean_anomaly_at_eccentric_anomaly(ecc_anom)
    }

    /// Gets the 3D position at a given angle (true anomaly) in the orbit.
    ///
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.0);
    ///
    /// let pos = orbit.get_position_at_angle(0.0);
    ///
    /// assert_eq!(pos, (100.0, 0.0, 0.0));
    /// ```
    fn get_position_at_angle(&self, angle: f64) -> Vec3 {
        let (x, y) = self.get_flat_position_at_angle(angle);
        self.tilt_flat_position(x, y)
    }

    /// Gets the speed at a given angle (true anomaly) in the orbit.
    ///
    /// The speed is derived from the vis-viva equation, and so is
    /// a lot faster than the velocity calculation.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.
    /// Anything out of range will get wrapped around.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let speed_periapsis = orbit.get_speed_at_angle(0.0);
    /// let speed_apoapsis = orbit.get_speed_at_angle(std::f64::consts::PI);
    ///
    /// assert!(speed_periapsis > speed_apoapsis);
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_speed_at_angle(&self, angle: f64) -> f64 {
        // https://en.wikipedia.org/wiki/Vis-viva_equation
        // v^2 = GM (2/r - 1/a)
        // v = sqrt(GM * (2/r - 1/a))

        let r = self.get_altitude_at_angle(angle);
        let a = self.get_semi_major_axis();

        return ((2.0 / r - a.recip()) * self.get_gravitational_parameter()).sqrt();
    }

    /// Gets the speed at a given time in the orbit.
    ///
    /// # Performance
    /// The time will be converted into an eccentric anomaly, which uses
    /// numerical methods and so is not very performant.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you know the eccentric anomaly or the true anomaly,
    /// then you should use the [`get_speed_at_angle`][OrbitTrait::get_speed_at_angle]
    /// method instead.  
    /// It does not use numerical methods and therefore is a lot faster.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_speed_at_time(&self, t: f64) -> f64 {
        self.get_speed_at_angle(self.get_true_anomaly_at_time(t))
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
    fn get_speed_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> f64 {
        self.get_speed_at_angle(self.get_true_anomaly_at_eccentric_anomaly(eccentric_anomaly))
    }

    /// Gets the velocity at a given angle (true anomaly) in the orbit if
    /// it had an inclination and longitude of ascending node of 0.
    ///
    /// # Flat
    /// This ignores "orbital tilting" parameters, namely the inclination and
    /// the longitude of ascending node.
    ///
    /// # Performance
    /// The velocity is derived from the eccentric anomaly, which uses numerical
    /// methods and so is not very performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_angle`][OrbitTrait::get_speed_at_angle] instead.  
    /// And if you already know the eccentric anomaly, use
    /// [`get_flat_velocity_at_eccentric_anomaly`][OrbitTrait::get_flat_velocity_at_eccentric_anomaly]
    /// instead.
    /// These functions do not require numerical methods and therefore are a lot faster.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let vel_periapsis = orbit.get_flat_velocity_at_angle(0.0);
    /// let vel_apoapsis = orbit.get_flat_velocity_at_angle(std::f64::consts::PI);
    ///
    /// fn get_magnitude(vector: (f64, f64)) -> f64 {
    ///     (vector.0 * vector.0 + vector.1 * vector.1).sqrt()
    /// }
    ///
    /// let speed_periapsis = get_magnitude(vel_periapsis);
    /// let speed_apoapsis = get_magnitude(vel_apoapsis);
    ///
    /// assert!(speed_periapsis > speed_apoapsis);
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_flat_velocity_at_angle(&self, angle: f64) -> Vec2 {
        let eccentric_anomaly = self.get_eccentric_anomaly_at_true_anomaly(angle);

        return self.get_flat_velocity_at_eccentric_anomaly(eccentric_anomaly);
    }

    /// Gets the velocity at a given eccentric anomaly in the orbit if
    /// it had an inclination and longitude of ascending node of 0.
    ///
    /// # Flat
    /// This ignores "orbital tilting" parameters, namely the inclination and
    /// the longitude of ascending node.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_flat_velocity_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> Vec2 {
        // TODO: PARABOLA SUPPORT: This does not play well with parabolic trajectories.
        if self.get_eccentricity() < 1.0 {
            // https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
            // Equation 8:
            //                                   [      -sin E       ]
            // vector_o'(t) = sqrt(GM * a) / r * [ sqrt(1-e^2) cos E ]
            //                                   [         0         ]

            let multiplier = (self.get_semi_major_axis() * self.get_gravitational_parameter())
                .sqrt()
                / self.get_altitude_at_eccentric_anomaly(eccentric_anomaly);

            let (sin, cos) = eccentric_anomaly.sin_cos();

            (
                -sin * multiplier,
                (1.0 - self.get_eccentricity().powi(2)).sqrt() * cos * multiplier,
            )
        } else {
            // https://space.stackexchange.com/a/54418
            //                                    [        sinh F       ]
            // vector_o'(t) = sqrt(-GM * a) / r * [ -sqrt(e^2-1) cosh F ]
            //                                    [          0          ]
            let multiplier = (-self.get_semi_major_axis() * self.get_gravitational_parameter())
                .sqrt()
                / self.get_altitude_at_eccentric_anomaly(eccentric_anomaly);

            let (sinh, cosh) = sinhcosh(eccentric_anomaly);

            (
                sinh * multiplier,
                -(self.get_eccentricity().powi(2) - 1.0).sqrt() * cosh * multiplier,
            )
        }
    }

    /// Gets the velocity at a given time in the orbit if
    /// it had an inclination and longitude of ascending node of 0.
    ///
    /// # Flat
    /// This ignores "orbital tilting" parameters, namely the inclination and
    /// the longitude of ascending node.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_flat_velocity_at_time(&self, t: f64) -> Vec2 {
        self.get_flat_velocity_at_eccentric_anomaly(self.get_eccentric_anomaly_at_time(t))
    }

    // TODO: Make all docstrings sectioned like these
    /// Gets the 2D position at a given angle (true anomaly) in the orbit.
    ///
    /// This ignores "orbital tilting" parameters, namely the inclination and
    /// the longitude of ascending node.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.0);
    ///
    /// let pos = orbit.get_flat_position_at_angle(0.0);
    ///
    /// assert_eq!(pos, (100.0, 0.0));
    /// ```
    fn get_flat_position_at_angle(&self, angle: f64) -> Vec2 {
        let alt = self.get_altitude_at_angle(angle);
        return (alt * angle.cos(), alt * angle.sin());
    }

    /// Gets the 2D position at a given time in the orbit.
    ///
    /// This involves calculating the true anomaly at a given time,
    /// and so is not very performant.
    /// It is recommended to cache this value when possible.
    ///
    /// This ignores "orbital tilting" parameters, namely the inclination
    /// and longitude of ascending node.
    ///
    /// For closed orbits (with an eccentricity less than 1), the
    /// `t` (time) value ranges from 0 to 1.  
    /// Anything out of range will get wrapped around.
    ///
    /// For open orbits (with an eccentricity of at least 1), the
    /// `t` (time) value is unbounded.  
    /// Note that due to floating-point imprecision, values of extreme
    /// magnitude may not be accurate.
    ///
    /// **This function returns non-finite numbers for parabolic orbits**
    /// due to how the equation for true anomaly works.
    fn get_flat_position_at_time(&self, t: f64) -> Vec2 {
        self.get_flat_position_at_angle(self.get_true_anomaly_at_time(t))
    }

    /// Gets the 2D position at a given eccentric anomaly in the orbit.
    ///
    /// This ignores "orbital tilting" parameters, namely the inclination
    /// and longitude of ascending node.
    ///
    /// For closed orbits (with an eccentricity less than 1), the
    /// `t` (time) value ranges from 0 to 1.  
    /// Anything out of range will get wrapped around.
    ///
    /// For open orbits (with an eccentricity of at least 1), the
    /// `t` (time) value is unbounded.  
    /// Note that due to floating-point imprecision, values of extreme
    /// magnitude may not be accurate.
    ///
    /// **This function returns non-finite numbers for parabolic orbits**
    /// due to how the equation for true anomaly works.
    fn get_flat_position_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> Vec2 {
        self.get_flat_position_at_angle(
            self.get_true_anomaly_at_eccentric_anomaly(eccentric_anomaly),
        )
    }

    /// Gets the velocity at a given angle (true anomaly) in the orbit.
    ///
    /// # Performance
    /// The velocity is derived from the eccentric anomaly, which uses numerical
    /// methods and so is not very performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_angle`][OrbitTrait::get_speed_at_angle] instead.  
    /// Or, if you already have the eccentric anomaly, use
    /// [`get_velocity_at_eccentric_anomaly`][OrbitTrait::get_velocity_at_eccentric_anomaly]
    /// instead.
    /// These functions do not require numerical methods and therefore are a lot faster.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let vel_periapsis = orbit.get_velocity_at_angle(0.0);
    /// let vel_apoapsis = orbit.get_velocity_at_angle(std::f64::consts::PI);
    ///
    /// fn get_magnitude(vector: (f64, f64, f64)) -> f64 {
    ///     (vector.0 * vector.0 + vector.1 * vector.1 + vector.2 * vector.2).sqrt()
    /// }
    ///
    /// let speed_periapsis = get_magnitude(vel_periapsis);
    /// let speed_apoapsis = get_magnitude(vel_apoapsis);
    ///
    /// assert!(speed_periapsis > speed_apoapsis)
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_velocity_at_angle(&self, angle: f64) -> Vec3 {
        let (x, y) = self.get_flat_velocity_at_angle(angle);
        self.tilt_flat_position(x, y)
    }

    /// Gets the velocity at a given eccentric anomaly in the orbit.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.5);
    ///
    /// let vel_periapsis = orbit.get_velocity_at_angle(0.0);
    /// let vel_apoapsis = orbit.get_velocity_at_angle(std::f64::consts::PI);
    /// ```
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_velocity_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> Vec3 {
        let (x, y) = self.get_flat_velocity_at_eccentric_anomaly(eccentric_anomaly);

        self.tilt_flat_position(x, y)
    }

    /// Gets the velocity at a given time in the orbit.
    ///
    /// # Performance
    /// The velocity is derived from the eccentric anomaly, which uses numerical
    /// methods and so is not very performant.  
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_time`][OrbitTrait::get_speed_at_time] instead.  
    /// Or, if you already have the eccentric anomaly, use
    /// [`get_velocity_at_eccentric_anomaly`][OrbitTrait::get_velocity_at_eccentric_anomaly]
    /// instead.
    /// These functions do not require numerical methods and therefore are a lot faster.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_velocity_at_time(&self, t: f64) -> Vec3 {
        self.get_velocity_at_eccentric_anomaly(self.get_eccentric_anomaly_at_time(t))
    }

    /// Gets the altitude of the body from its parent at a given angle (true anomaly) in the orbit.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.0);
    ///
    /// let altitude = orbit.get_altitude_at_angle(0.0);
    ///
    /// assert_eq!(altitude, 100.0);
    /// ```
    fn get_altitude_at_angle(&self, angle: f64) -> f64;

    /// Gets the altitude of the body from its parent at a given eccentric anomaly in the orbit.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::new_default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.0);
    ///
    /// let altitude = orbit.get_altitude_at_eccentric_anomaly(0.0);
    ///
    /// assert_eq!(altitude, 100.0);
    /// ```
    fn get_altitude_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> f64 {
        self.get_altitude_at_angle(self.get_true_anomaly_at_eccentric_anomaly(eccentric_anomaly))
    }

    /// Gets the altitude of the body from its parent at a given time in the orbit.
    ///
    /// This involves calculating the true anomaly at a given time, and so is not very performant.  
    /// It is recommended to cache this value when possible.
    ///
    /// Note that due to floating-point imprecision, values of extreme
    /// magnitude may not be accurate.
    ///
    /// **This function returns infinity for parabolic orbits** due to how the equation for
    /// true anomaly works.
    fn get_altitude_at_time(&self, t: f64) -> f64 {
        self.get_altitude_at_angle(self.get_true_anomaly_at_time(t))
    }

    /// Gets the 3D position at a given time in the orbit.
    ///
    /// This involves calculating the true anomaly at a given time,
    /// and so is not very performant.  
    /// It is recommended to cache this value when possible.
    ///
    /// For closed orbits (with an eccentricity less than 1), the
    /// `t` (time) value ranges from 0 to 1.  
    /// Anything out of range will get wrapped around.
    ///
    /// For open orbits (with an eccentricity of at least 1), the
    /// `t` (time) value is unbounded.  
    /// Note that due to floating-point imprecision, values of extreme
    /// magnitude may not be accurate.
    ///
    /// **This function returns non-finite numbers for parabolic orbits**
    /// due to how the equation for true anomaly works.
    fn get_position_at_time(&self, t: f64) -> Vec3 {
        self.get_position_at_angle(self.get_true_anomaly_at_time(t))
    }

    /// Tilts a 2D position into 3D, using the orbital parameters.
    ///
    /// This uses the "orbital tilting" parameters, namely the inclination
    /// and longitude of ascending node, to tilt that position into the same
    /// plane that the orbit resides in.
    ///
    /// This function performs 10x faster in the cached version of the
    /// [`Orbit`] struct, as it doesn't need to recalculate the transformation
    /// matrix needed to transform 2D vector.
    fn tilt_flat_position(&self, x: f64, y: f64) -> Vec3 {
        self.get_transformation_matrix().dot_vec((x, y))
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

    /// Gets the inclination of the orbit in radians.
    ///
    /// The inclination of an orbit is the angle between the plane of the
    /// orbit and the reference plane.
    ///
    /// In simple terms, it tells you how "tilted" the orbit is.
    ///
    /// Wikipedia: <https://en.wikipedia.org/wiki/Orbital_inclination>
    fn get_inclination(&self) -> f64;

    /// Sets the inclination of the orbit in radians.
    ///
    /// The inclination of an orbit is the angle between the plane of the
    /// orbit and the reference plane.
    ///
    /// In simple terms, it tells you how "tilted" the orbit is.
    ///
    /// Wikipedia: <https://en.wikipedia.org/wiki/Orbital_inclination>
    fn set_inclination(&mut self, inclination: f64);

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
    fn set_arg_pe(&mut self, arg_pe: f64);

    /// Gets the longitude of ascending node of the orbit in radians.
    ///
    /// Wikipedia:  
    /// The longitude of ascending node is the angle from a specified
    /// reference direction, called the origin of longitude, to the direction
    /// of the ascending node, as measured in a specified reference plane.  
    /// <https://en.wikipedia.org/wiki/Longitude_of_the_ascending_node>
    ///
    /// In simple terms, it tells you how, and in which direction,
    /// the orbit "tilts".
    fn get_long_asc_node(&self) -> f64;

    /// Sets the longitude of ascending node of the orbit in radians.
    ///
    /// Wikipedia:  
    /// The longitude of ascending node is the angle from a specified
    /// reference direction, called the origin of longitude, to the direction
    /// of the ascending node, as measured in a specified reference plane.  
    /// <https://en.wikipedia.org/wiki/Longitude_of_the_ascending_node>
    ///
    /// In simple terms, it tells you how, and in which direction,
    /// the orbit "tilts".
    fn set_long_asc_node(&mut self, long_asc_node: f64);

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
    fn get_gravitational_parameter(&self) -> f64;

    /// Sets the gravitational parameter of the parent body.
    ///
    /// The gravitational parameter mu of the parent body equals a certain
    /// gravitational constant G times the mass of the parent body M.
    ///
    /// In other words, mu = GM.
    fn set_gravitational_parameter(&mut self, gravitational_parameter: f64, mode: MuSetterMode);
}

/// A mode to describe how the gravitational parameter setter should behave.
///
/// This is used to describe how the setter should behave when setting the
/// gravitational parameter of the parent body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MuSetterMode {
    /// Keep all the other orbital parameters the same.
    ///
    /// **This will change the position and velocity of the orbiting body abruptly,
    /// if you use the time-based functions.** It will not, however, change the trajectory
    /// of the orbit.
    KeepElements,
    /// Keep the overall shape of the orbit, but modify the mean anomaly at epoch
    /// such that the position at the given time t is the same.
    ///
    /// **This will change the velocity of the orbiting body abruptly, if you use
    /// the time-based functions.**
    KeepPositionAtTime(f64),
    /// Keep the position and velocity of the orbit at a certain time t the same.
    ///
    /// **This will change the orbit's overall trajectory.**
    KeepPositionAndVelocityAtTime(f64),
    /// Keep the overall shape of the orbit, but modify the mean anomaly at epoch
    /// such that the position at the given angle (in radians) is the same.
    ///
    /// **This will change the velocity of the orbiting body abruptly, if you use
    /// the time-based functions.**
    KeepPositionAtAngle(f64),
    /// Keep the position and velocity of the orbit at a certain angle t the same.
    ///
    /// **This will change the orbit's overall trajectory.**
    KeepPositionAndVelocityAtAngle(f64),
}

/// An error to describe why setting the periapsis of an orbit failed.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ApoapsisSetterError {
    /// ### Attempt to set apoapsis to a value less than periapsis.
    /// By definition, an orbit's apoapsis is the highest point in the orbit,
    /// and its periapsis is the lowest point in the orbit.  
    /// Therefore, it doesn't make sense for the apoapsis to be lower than the periapsis.
    ApoapsisLessThanPeriapsis,

    /// ### Attempt to set apoapsis to a negative value.
    /// By definition, the apoapsis is the highest point in the orbit.  
    /// You can't be a negative distance away from the center of mass of the parent body.  
    /// Therefore, it doesn't make sense for the apoapsis to be lower than zero.
    ApoapsisNegative,
}

#[cfg(test)]
mod tests;

#[inline]
fn keplers_equation(mean_anomaly: f64, eccentric_anomaly: f64, eccentricity: f64) -> f64 {
    return eccentric_anomaly - (eccentricity * eccentric_anomaly.sin()) - mean_anomaly;
}
#[inline]
fn keplers_equation_derivative(eccentric_anomaly: f64, eccentricity: f64) -> f64 {
    return 1.0 - (eccentricity * eccentric_anomaly.cos());
}
#[inline]
fn keplers_equation_second_derivative(eccentric_anomaly: f64, eccentricity: f64) -> f64 {
    return eccentricity * eccentric_anomaly.sin();
}

/// Get the hyperbolic sine and cosine of a number.
///
/// Usually faster than calling `x.sinh()` and `x.cosh()` separately.
///
/// Returns a tuple which contains:
/// - 0: The hyperbolic sine of the number.
/// - 1: The hyperbolic cosine of the number.
fn sinhcosh(x: f64) -> (f64, f64) {
    let e_x = x.exp();
    let e_neg_x = (-x).exp();

    return ((e_x - e_neg_x) * 0.5, (e_x + e_neg_x) * 0.5);
}

/// Solve a cubic equation to get its real root.
///
/// The cubic equation is in the form of:
/// ax^3 + bx^2 + cx + d
///
/// The cubic equation is assumed to be monotone.  
/// If it isn't monotone (i.e., the discriminant
/// is negative), it may return an incorrect value
/// or NaN.
fn solve_monotone_cubic(a: f64, b: f64, c: f64, d: f64) -> f64 {
    // Normalize coefficients so that a = 1
    // ax^3 + bx^2 + cx + d
    // ...where b, c, d are the normalized coefficients,
    // and a = 1
    let b = b / a;
    let c = c / a;
    let d = d / a;

    // Depress the cubic equation
    // t^3 + pt + q = 0
    // ...where:
    // p = (3ac - b^2) / (3a^2)
    // q = (2b^3 - 9abc + 27da^2) / (27a^3)
    // ...since a = 1, we can simplify them to:
    // p = (3c - b^2) / 3
    // q = (2b^3 - 9bc + 27d) / 27
    let b_sq = b * b;

    let p = (3.0 * c - b_sq) / 3.0;
    let q = (2.0 * b_sq * b - 9.0 * b * c + 27.0 * d) / 27.0;

    let q_div_two = q / 2.0;
    let p_div_three = p / 3.0;
    let p_div_three_cubed = p_div_three * p_div_three * p_div_three;
    let discriminant = q_div_two * q_div_two + p_div_three_cubed;

    if discriminant < 0.0 {
        // Function is not monotone
        return f64::NAN;
    }

    let t = {
        let sqrt_discriminant = discriminant.sqrt();
        let neg_q_div_two = -q_div_two;
        let u = (neg_q_div_two + sqrt_discriminant).cbrt();
        let v = (neg_q_div_two - sqrt_discriminant).cbrt();
        u + v
    };

    // x_i = t_i - b / 3a
    // here, a = 1
    return t - b / 3.0;
}

mod generated_sinh_approximator;
