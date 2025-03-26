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
//! use glam::DVec3;
//!
//! use keplerian_sim::{Orbit, OrbitTrait};
//!
//! # fn main() {
//! // Create a perfectly circular orbit with a radius of 1 meter
//! let orbit = Orbit::default();
//! assert_eq!(orbit.get_position_at_time(0.0), DVec3::new(1.0, 0.0, 0.0));
//! # }
//! #
//! ```

#![warn(missing_docs)]

mod body;
pub mod body_presets;
mod compact_orbit;
mod generated_sinh_approximator;
mod orbit;
mod state_vectors;
mod universe;

use std::f64::consts::{PI, TAU};

pub use body::Body;
pub use compact_orbit::CompactOrbit;
use glam::{DVec2, DVec3};
pub use orbit::Orbit;
pub use state_vectors::StateVectors;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use universe::{Universe, G};

/// A constant used to get the initial seed for the eccentric anomaly.
///
/// It's very arbitrary, but according to some testing, a value just
/// below 1 works better than exactly 1.
///
/// Source:
/// "Two fast and accurate routines for solving the elliptic Kepler
/// equation for all values of the eccentricity and mean anomaly"
/// by Daniele Tommasini and David N. Olivieri,
/// section 2.1.2, 'The "rational seed"'
///
/// https://doi.org/10.1051/0004-6361/202141423
const B: f64 = 0.999999;

/// A constant used for the Laguerre method.
///
/// The paper "An improved algorithm due to
/// laguerre for the solution of Kepler's equation."
/// says:
///
/// > Similar experimentation has been done with values of n both greater and smaller
/// > than n = 5. The speed of convergence seems to be very insensitive to the choice of n.
/// > No value of n was found to yield consistently better convergence properties than the
/// > choice of n = 5 though specific cases were found where other choices would give
/// > faster convergence.
const N_U32: u32 = 5;

/// A constant used for the Laguerre method.
///
/// The paper "An improved algorithm due to
/// laguerre for the solution of Kepler's equation."
/// says:
///
/// > Similar experimentation has been done with values of n both greater and smaller
/// > than n = 5. The speed of convergence seems to be very insensitive to the choice of n.
/// > No value of n was found to yield consistently better convergence properties than the
/// > choice of n = 5 though specific cases were found where other choices would give
/// > faster convergence.
const N_F64: f64 = 5.0;

/// The maximum number of iterations for the numerical approach algorithms.
///
/// This is used to prevent infinite loops in case the method fails to converge.
const NUMERIC_MAX_ITERS: u32 = 1000;

const PI_SQUARED: f64 = PI * PI;

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
/// use glam::{DVec2, DVec3};
///
/// use keplerian_sim::Matrix3x2;
///
/// let matrix = Matrix3x2 {
///    e11: 1.0, e12: 0.0,
///    e21: 0.0, e22: 1.0,
///    e31: 0.0, e32: 0.0,
/// };
///
/// let vec = DVec2::new(1.0, 2.0);
///
/// let result = matrix.dot_vec(&vec);
///
/// assert_eq!(result, DVec3::new(1.0, 2.0, 0.0));
/// ```
#[allow(missing_docs)]
#[derive(Copy, Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Matrix3x2 {
    // Element XY
    pub e11: f64,
    pub e12: f64,
    pub e21: f64,
    pub e22: f64,
    pub e31: f64,
    pub e32: f64,
}

impl Matrix3x2 {
    /// Computes a dot product between this matrix and a 2D vector.
    ///
    /// # Example
    /// ```
    /// use glam::{DVec2, DVec3};
    ///
    /// use keplerian_sim::Matrix3x2;
    ///
    /// let matrix = Matrix3x2 {
    ///     e11: 1.0, e12: 0.0,
    ///     e21: 0.0, e22: 1.0,
    ///     e31: 1.0, e32: 1.0,
    /// };
    ///
    /// let vec = DVec2::new(1.0, 2.0);
    ///
    /// let result = matrix.dot_vec(&vec);
    ///
    /// assert_eq!(result, DVec3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn dot_vec(&self, vec: &DVec2) -> DVec3 {
        DVec3::new(
            vec.x * self.e11 + vec.y * self.e12,
            vec.x * self.e21 + vec.y * self.e22,
            vec.x * self.e31 + vec.y * self.e32,
        )
    }
}

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
///     let orbit = Orbit::default();
///     accepts_orbit(&orbit);
///
///     let compact = CompactOrbit::default();
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
/// #     let orbit = Orbit::default();
/// #     accepts_orbit(&orbit);
/// #  
/// #     let compact = CompactOrbit::default();
/// #     accepts_orbit(&compact);
///       let not_orbit = (0.0, 1.0);
///       accepts_orbit(&not_orbit);
/// # }
/// ```
pub trait OrbitTrait: Default {
    /// Creates a new `OrbitTrait` instance with the given parameters.
    ///
    /// Note: This function uses eccentricity instead of apoapsis.  
    /// If you want to provide an apoapsis instead, consider using the
    /// [`OrbitTrait::with_apoapsis`] function instead.
    ///
    /// ### Parameters
    /// - `eccentricity`: The eccentricity of the orbit.
    /// - `periapsis`: The periapsis of the orbit, in meters.
    /// - `inclination`: The inclination of the orbit, in radians.
    /// - `arg_pe`: The argument of periapsis of the orbit, in radians.
    /// - `long_asc_node`: The longitude of ascending node of the orbit, in radians.
    /// - `mean_anomaly`: The mean anomaly of the orbit, in radians.
    /// - `parent_mass`: The mass of the parent body, in kilograms.
    fn new(
        eccentricity: f64,
        periapsis: f64,
        inclination: f64,
        arg_pe: f64,
        long_asc_node: f64,
        mean_anomaly: f64,
        parent_mass: f64,
    ) -> Self;

    /// Creates a new orbit with the given parameters.
    ///
    /// Note: This function uses apoapsis instead of eccentricity.  
    /// Because of this, it's not recommended to initialize
    /// parabolic or hyperbolic 'orbits' with this function.  
    /// If you're looking to initialize a parabolic or hyperbolic
    /// trajectory, consider using the [`OrbitTrait::new`] function instead.
    ///
    /// ### Parameters
    /// - `apoapsis`: The apoapsis of the orbit, in meters.
    /// - `periapsis`: The periapsis of the orbit, in meters.
    /// - `inclination`: The inclination of the orbit, in radians.
    /// - `arg_pe`: The argument of periapsis of the orbit, in radians.
    /// - `long_asc_node`: The longitude of ascending node of the orbit, in radians.
    /// - `mean_anomaly`: The mean anomaly of the orbit, in radians.
    /// - `parent_mass`: The mass of the parent body, in kilograms.
    fn with_apoapsis(
        apoapsis: f64,
        periapsis: f64,
        inclination: f64,
        arg_pe: f64,
        long_asc_node: f64,
        mean_anomaly: f64,
        parent_mass: f64,
    ) -> Self;

    /// Returns the mass of the parent body, in kilograms.
    fn get_parent_mass(&self) -> f64;

    /// Returns g * mass.
    fn get_mu(&self, g: f64) -> f64 {
        g * self.get_parent_mass()
    }

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
    /// use keplerian_sim::{G, Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
    /// orbit.set_periapsis(50.0, G);
    /// orbit.set_apoapsis_force(100.0, G);
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
    fn get_semi_latus_rectum(&self) -> f64 {
        let eccentricity = self.get_eccentricity();
        if eccentricity == 1.0 {
            2.0 * self.get_periapsis()
        } else {
            self.get_semi_major_axis() * (1.0 - eccentricity * eccentricity)
        }
    }

    /// Gets the linear eccentricity of the orbit, in meters.
    ///
    /// In an elliptic orbit, the linear eccentricity is the distance
    /// between its center and either of its two foci (focuses).
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{G, Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
    /// orbit.set_periapsis(50.0, G);
    /// orbit.set_apoapsis_force(100.0, G);
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
    /// use keplerian_sim::{G, Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
    /// orbit.set_eccentricity(0.5, G); // Elliptic
    /// assert!(orbit.get_apoapsis() > 0.0);
    ///
    /// orbit.set_eccentricity(1.0, G); // Parabolic
    /// assert!(orbit.get_apoapsis().is_infinite());
    ///
    /// orbit.set_eccentricity(2.0, G); // Hyperbolic
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
    /// use keplerian_sim::{G, Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
    /// orbit.set_periapsis(50.0, G);
    ///
    /// assert!(
    ///     orbit.set_apoapsis(100.0, G)
    ///         .is_ok()
    /// );
    ///
    /// let result = orbit.set_apoapsis(25.0, G);
    /// assert!(result.is_err());
    /// assert!(
    ///     result.unwrap_err() ==
    ///     keplerian_sim::ApoapsisSetterError::ApoapsisLessThanPeriapsis
    /// );
    ///
    /// let result = orbit.set_apoapsis(-25.0, G);
    /// assert!(result.is_err());
    /// assert!(
    ///     result.unwrap_err() ==
    ///     keplerian_sim::ApoapsisSetterError::ApoapsisNegative
    /// );
    /// ```
    fn set_apoapsis(&mut self, apoapsis: f64, g: f64) -> Result<(), ApoapsisSetterError>;

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
    fn set_apoapsis_force(&mut self, apoapsis: f64, g: f64);

    /// Gets the transformation matrix needed to tilt a 2D vector into the
    /// tilted orbital plane.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let orbit = Orbit::default();
    /// let matrix = orbit.get_transformation_matrix();
    ///
    /// assert_eq!(matrix, keplerian_sim::Matrix3x2 {
    ///     e11: 1.0, e12: 0.0,
    ///     e21: 0.0, e22: 1.0,
    ///     e31: 0.0, e32: 0.0,
    /// });
    /// ```
    fn get_transformation_matrix(&self) -> Matrix3x2;

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
    fn get_eccentric_anomaly(&self, mean_anomaly: f64) -> f64 {
        if self.get_eccentricity() < 1.0 {
            self.get_eccentric_anomaly_elliptic(mean_anomaly)
        } else {
            self.get_eccentric_anomaly_hyperbolic(mean_anomaly)
        }
    }

    /// Gets the true anomaly at a given eccentric anomaly in the orbit.
    ///
    /// This function is faster than the function which takes mean anomaly as input,
    /// as the eccentric anomaly is hard to calculate.
    ///
    /// This function returns +/- pi for parabolic orbits due to how the equation works,
    /// and so **may result in infinities when combined with other functions**.
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

    /// Gets the mean anomaly at a given time in the orbit.
    ///
    /// The mean anomaly is the fraction of an elliptical orbit's period
    /// that has elapsed since the orbiting body passed periapsis,
    /// expressed as an angle which can be used in calculating the position
    /// of that body in the classical two-body problem.
    ///
    /// \- [Wikipedia](https://en.wikipedia.org/wiki/Mean_anomaly)
    fn get_mean_anomaly_at_time(&self, t: f64) -> f64 {
        t * TAU + self.get_mean_anomaly_at_epoch()
    }

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

    /// Returns the distance in meters at time t.
    ///
    /// \- [Source](https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf)
    fn get_distance_at_time(&self, t: f64) -> f64 {
        self.get_semi_major_axis()
            * (1. - self.get_eccentricity() * self.get_eccentric_anomaly_at_time(t).cos())
    }

    /// Gets the 3D position at a given angle (true anomaly) in the orbit.
    ///
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Example
    /// ```
    /// use glam::DVec3;
    ///
    /// use keplerian_sim::{G, Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
    /// orbit.set_periapsis(100.0, G);
    /// orbit.set_eccentricity(0.0, G);
    ///
    /// let pos = orbit.get_position_at_angle(0.0);
    ///
    /// assert_eq!(pos, DVec3::new(100.0, 0.0, 0.0));
    /// ```
    fn get_position_at_angle(&self, angle: f64) -> DVec3 {
        self.tilt_flat_position(&self.get_flat_position_at_angle(angle))
    }

    /// Gets the 2D position at a given angle (true anomaly) in the orbit.
    ///
    /// This ignores "orbital tilting" parameters, namely the inclination and
    /// the longitude of ascending node.
    ///
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Example
    /// ```
    /// use glam::DVec2;
    /// use keplerian_sim::{G, Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
    /// orbit.set_periapsis(100.0, G);
    /// orbit.set_eccentricity(0.0, G);
    ///
    /// let pos = orbit.get_flat_position_at_angle(0.0);
    ///
    /// assert_eq!(pos, DVec2::new(100.0, 0.0));
    /// ```
    fn get_flat_position_at_angle(&self, angle: f64) -> DVec2 {
        let alt = self.get_altitude_at_angle(angle);
        DVec2::new(alt * angle.cos(), alt * angle.sin())
    }

    /// Gets the altitude of the body from its parent at a given angle (true anomaly) in the orbit.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{G, Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
    /// orbit.set_periapsis(100.0, G);
    /// orbit.set_eccentricity(0.0, G);
    ///
    /// let altitude = orbit.get_altitude_at_angle(0.0);
    ///
    /// assert_eq!(altitude, 100.0);
    /// ```
    fn get_altitude_at_angle(&self, true_anomaly: f64) -> f64 {
        (self.get_semi_latus_rectum() / (1.0 + self.get_eccentricity() * true_anomaly.cos())).abs()
    }

    /// Gets the altitude of the body from its parent at a given time in the orbit.
    ///
    /// This involves calculating the true anomaly at a given time, and so is not very performant.  
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
    fn get_position_at_time(&self, t: f64) -> DVec3 {
        self.get_position_at_angle(self.get_true_anomaly_at_time(t))
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
    fn get_flat_position_at_time(&self, t: f64) -> DVec2 {
        self.get_flat_position_at_angle(self.get_true_anomaly_at_time(t))
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
    fn tilt_flat_position(&self, position: &DVec2) -> DVec3 {
        self.get_transformation_matrix().dot_vec(position)
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
    fn set_eccentricity(&mut self, eccentricity: f64, g: f64);

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
    fn set_periapsis(&mut self, periapsis: f64, g: f64);

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
    fn set_inclination(&mut self, inclination: f64, g: f64);

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
    fn set_arg_pe(&mut self, arg_pe: f64, g: f64);

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
    fn set_long_asc_node(&mut self, long_asc_node: f64, g: f64);

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
    fn set_mean_anomaly_at_epoch(&mut self, mean_anomaly: f64, g: f64);

    /// Get an initial guess for the hyperbolic eccentric anomaly of an orbit.
    ///
    /// From the paper:  
    /// "A new method for solving the hyperbolic Kepler equation"  
    /// by Baisheng Wu et al.  
    /// Quote:
    /// "we divide the hyperbolic eccentric anomaly interval into two parts:
    /// a finite interval and an infinite interval. For the finite interval,
    /// we apply a piecewise Pade approximation to establish an initial
    /// approximate solution of HKE. For the infinite interval, an analytical
    /// initial approximate solution is constructed."
    fn get_approx_hyp_ecc_anomaly(&self, mean_anomaly: f64) -> f64 {
        let sign = mean_anomaly.signum();
        let mean_anomaly = mean_anomaly.abs();
        const SINH_5: f64 = 74.20321057778875;

        let eccentricity = self.get_eccentricity();

        // (Paragraph after Eq. 5 in the aforementioned paper)
        //   The [mean anomaly] interval [0, e_c sinh(5) - 5) can
        //   be separated into fifteen subintervals corresponding to
        //   those intervals of F in [0, 5), see Eq. (4).
        sign * if mean_anomaly < eccentricity * SINH_5 - 5.0 {
            // We use the Pade approximation of sinh of order
            // [3 / 2], in `crate::generated_sinh_approximator`.
            // We can then rearrange the equation to a cubic
            // equation in terms of (F - a) and solve it.
            //
            // To quote the paper:
            //   Replacing sinh(F) in [the hyperbolic Kepler
            //   equation] with its piecewise Pade approximation
            //   defined in Eq. (4) [`crate::generated_sinh_approximator`]
            //   yields:
            //     e_c P(F) - F = M_h                          (6)
            //
            //   Eq. (6) can be written as a cubic equation in u = F - a, as
            //     (e_c p_3 - q_2)u^3 +
            //     (e_c p_2 - (M_h + a)q_2 - q_1) u^2 +
            //     (e_c p_1 - (M_h + a)q_1 - 1)u +
            //     e_c s - M_h - a = 0                         (7)
            //
            //   Solving Eq. (7) and picking the real root F = F_0 in the
            //   corresponding subinterval results in an initial approximate
            //   solution to [the hyperbolic Kepler equation].
            //
            // For context:
            // - `e_c` is eccentricity
            // - `p_*`, `q_*`, `a`, and `s` is derived from the Pade approximation
            //   arguments, which can be retrieved using the
            //   `generated_sinh_approximator::get_sinh_approx_params` function
            // - `M_h` is the mean anomaly
            // - `F` is the eccentric anomaly

            use crate::generated_sinh_approximator::get_sinh_approx_params;
            let params = get_sinh_approx_params(mean_anomaly);

            // We first get the value of each coefficient in the cubic equation:
            // Au^3 + Bu^2 + Cu + D = 0
            let mean_anom_plus_a = mean_anomaly + params.a;
            let coeff_a = eccentricity * params.p_3 - params.q_2;
            let coeff_b = eccentricity * params.p_2 - mean_anom_plus_a * params.q_2 - params.q_1;
            let coeff_c = eccentricity * params.p_1 - mean_anom_plus_a * params.q_1 - 1.0;
            let coeff_d = eccentricity * params.s - mean_anomaly - params.a;

            // Then we solve it to get the value of u = F - a
            let u = solve_monotone_cubic(coeff_a, coeff_b, coeff_c, coeff_d);

            u + params.a
        } else {
            // Equation 13
            // A *very* rough guess, with an error that may exceed 1%.
            let rough_guess = (2.0 * mean_anomaly / eccentricity).ln();

            /*
            A fourth-order Schröder iteration of the second kind
            is performed to create a better guess.
            ...Apparently it's not a well-known thing, but the aforementioned paper
            referenced this other paper about Schröder iterations:
            https://doi.org/10.1016/j.cam.2019.02.035

            To do the Schröder iteration, we need to compute a delta value
            to be added to the rough guess. Part of Equation 15 from the paper is below.

            delta = (
                    6 * [e_c^2 / (4 * M_h) + F_a] / (e_c * c_a - 1) +
                    3 * [e_c * s_a / (e_c * c_a - 1)]{[e_c^2 / (4 * M_h) + F_a] / (e_c * c_a - 1)}^2
                ) / (
                    6 +
                    6 * [e_c * s_a / (e_c * c_a - 1)]{[e_c^2 / (4 * M_h) + F_a] / (e_c * c_a - 1)} +
                    [e_c * c_a / (e_c * c_a - 1)]{[e_c^2 / (4 * M_h) + F_a] / (e_c * c_a - 1)}^2
                )
            ...where:
            e_c = eccentricity
            F_a = rough guess
            c_a = cosh(F_a) = 0.5 * [2 * M_h / e_c + e_c / (2 * M_h)],
            s_a = sinh(F_a) = 0.5 * [2 * M_h / e_c - e_c / (2 * M_h)]

            Although the equation may look intimidating, there are a lot of repeated values.
            We can simplify the equation by extracting the repeated values.

            Let:
                alpha = e_c^2 / (4 * M_h) + F_a
                beta  = 1 / (e_c * c_a - 1)
                gamma = alpha * beta

            The equation gets simplified into:

            delta = (
                    6 * gamma +
                    3 * e_c * s_a * beta * gamma^2
                ) / (
                    6 +
                    6 * e_c * s_a * beta * gamma +
                    e_c * c_a * beta * gamma^2
                )

            Then we can refine the rough guess into the initial guess:
            F_0 = F_a + delta
            */

            let (c_a, s_a) = {
                // c_a and s_a has a lot of repeated values, so we can
                // optimize by calculating them together.
                // c_a, s_a = 0.5 * [2 * M_h / e_c +- e_c / (2 * M_h)]
                //
                // define "left"  = 2 * M_h / e_c
                // define "right" = e_c / (2 * M_h)

                let left = 2.0 * mean_anomaly / eccentricity;
                let right = eccentricity / (2.0 * mean_anomaly);

                (0.5 * (left + right), 0.5 * (left - right))
            };

            let alpha = eccentricity * eccentricity / (4.0 * mean_anomaly) + rough_guess;

            let beta = (eccentricity * c_a - 1.0).recip();

            let gamma = alpha * beta;
            let gamma_sq = gamma * gamma;

            let delta = (6.0 * alpha * beta + 3.0 * (eccentricity * s_a * beta) * gamma_sq)
                / (6.0
                    + 6.0 * (eccentricity * s_a * beta) * gamma
                    + (eccentricity * c_a * beta) * gamma_sq);

            rough_guess + delta
        }
    }

    /// From the paper:  
    /// "A new method for solving the hyperbolic Kepler equation"  
    /// by Baisheng Wu et al.  
    fn get_eccentric_anomaly_hyperbolic(&self, mean_anomaly: f64) -> f64 {
        let mut ecc_anom = self.get_approx_hyp_ecc_anomaly(mean_anomaly);

        /*
        Do a fourth-order Schröder iteration of the second kind

        Equation 25 of "A new method for solving the hyperbolic Kepler equation"
        by Baisheng Wu et al.
        Slightly restructured:

        F_1^(4) = F_0 - (
            (6h/h' - 3h^2 h'' / h'^3) /
            (6 - 6h h'' / h'^2 + h^2 h'''/h'^3)
        )

        ...where:
        e_c = eccentricity
        F_0 = initial guess
        h   = e_c sinh(F_0) - F_0 - M_h
        h'  = e_c cosh(F_0) - 1
        h'' = e_c sinh(F_0)
            = h + F_0 + M_h
        h'''= h' + 1

        Rearranging for efficiency:
        h'''= e_c cosh(F_0)
        h'  = h''' - 1
        h'' = e_c sinh(F_0)
        h   = h'' - F_0 - M_h

        Factoring out 1/h':

        let r = 1 / h'

        F_1^(4) = F_0 - (
            (6hr - 3h^2 h'' r^3) /
            (6 - 6h h'' r^2 + h^2 h''' r^3)
        )

        Since sinh and cosh are very similar algebraically,
        it may be better to calculate them together.

        Paper about Schröder iterations:
        https://doi.org/10.1016/j.cam.2019.02.035
         */

        let eccentricity = self.get_eccentricity();

        for _ in 0..NUMERIC_MAX_ITERS {
            let (sinh_eca, cosh_eca) = sinhcosh(ecc_anom);

            let hppp = eccentricity * cosh_eca;
            let hp = hppp - 1.0;
            let hpp = eccentricity * sinh_eca;
            let h = hpp - ecc_anom - mean_anomaly;

            let h_sq = h * h;
            let r = hp.recip();
            let r_sq = r * r;
            let r_cub = r_sq * r;

            let denominator = 6.0 - 6.0 * h * hpp * r_sq + h_sq * hppp * r_cub;

            if denominator.abs() < 1e-30 || !denominator.is_finite() {
                // dangerously close to div-by-zero, break out
                #[cfg(debug_assertions)]
                eprintln!(
                    "Hyperbolic eccentric anomaly solver: denominator is too small or not finite"
                );
                break;
            }

            let numerator = 6.0 * h * r - 3.0 * h_sq * hpp * r_cub;
            let delta = numerator / denominator;

            ecc_anom -= delta;

            if delta.abs() < 1e-12 {
                break;
            }
        }

        ecc_anom
    }

    /// "An improved algorithm due to laguerre for the solution of Kepler's equation."
    /// by Bruce A. Conway
    /// https://doi.org/10.1007/bf01230852
    fn get_eccentric_anomaly_elliptic(&self, mut mean_anomaly: f64) -> f64 {
        let mut sign = 1.0;
        // Use the symmetry and periodicity of the eccentric anomaly
        // Equation 2 from the paper
        // "Two fast and accurate routines for solving
        // the elliptic Kepler equation for all values
        // of the eccentricity and mean anomaly"
        mean_anomaly %= TAU;
        if mean_anomaly > PI {
            // return self.get_eccentric_anomaly_elliptic(mean_anomaly - TAU);
            mean_anomaly -= TAU;
        }
        if mean_anomaly < 0.0 {
            // return -self.get_eccentric_anomaly_elliptic(-mean_anomaly);
            mean_anomaly = -mean_anomaly;
            sign = -1.0;
        }

        // Starting guess
        // Section 2.1.2, 'The "rational seed"',
        // equation 19, from the paper
        // "Two fast and accurate routines for solving
        // the elliptic Kepler equation for all values
        // of the eccentricity and mean anomaly"
        //
        // E_0 = M + (4beM(pi - M)) / (8eM + 4e(e-pi) + pi^2)
        // where:
        // e = eccentricity
        // M = mean anomaly
        // pi = the constant PI
        // b = the constant B
        let eccentricity = self.get_eccentricity();
        let mut eccentric_anomaly = mean_anomaly
            + (4.0 * eccentricity * B * mean_anomaly * (PI - mean_anomaly))
                / (8.0 * eccentricity * mean_anomaly
                    + 4.0 * eccentricity * (eccentricity - PI)
                    + PI_SQUARED);

        // Laguerre's method
        //
        // i = 2, 3, ..., n
        //
        // D = sqrt((n-1)^2(f'(x_i))^2 - n(n-1)f(x_i)f''(x_i))
        //
        // x_i+1 = x_i - (nf(x_i) / (f'(x_i) +/- D))
        // ...where the "+/-" is chosen to so that abs(denominator) is maximized
        for _ in 2..N_U32 {
            let f = keplers_equation(mean_anomaly, eccentric_anomaly, eccentricity);
            let fp = keplers_equation_derivative(eccentric_anomaly, eccentricity);
            let fpp = keplers_equation_second_derivative(eccentric_anomaly, eccentricity);

            let n = N_F64;
            let n_minus_1 = n - 1.0;
            let d = ((n_minus_1 * n_minus_1) * fp * fp - n * n_minus_1 * f * fpp)
                .abs()
                .sqrt()
                .copysign(fp);

            let denominator = n * f / (fp + d.max(1e-30));
            eccentric_anomaly -= denominator;

            if denominator.abs() < 1e-30 || !denominator.is_finite() {
                // dangerously close to div-by-zero, break out
                break;
            }
        }

        eccentric_anomaly * sign
    }

    /// Returns the position and velocity of an orbiting object relative to its parent.
    ///
    /// ### Parameters
    /// - `t`: A value between 0 and 1 for closed orbits, and unbounded for open orbits.
    /// - `g`: The gravitational constant.
    /// 
    /// \- [Source](https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf)
    fn get_state_vectors(&self, t: f64, g: f64) -> StateVectors {
        let distance = self.get_distance_at_time(t);
        let flat_velocity = (self.get_sqrt_mu_sma(g) / distance) * self.get_velocity_unit_vector();
        StateVectors {
            position: self.get_position_at_time(t),
            velocity: self.tilt_flat_position(&flat_velocity),
        }
    }

    /// Returns `(mu * semi_major_axis).sqrt()`.
    fn get_sqrt_mu_sma(&self, g: f64) -> f64;

    /// Returns a unit vector used to calculate the velocity.
    fn get_velocity_unit_vector(&self) -> DVec2;
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
    eccentric_anomaly - (eccentricity * eccentric_anomaly.sin()) - mean_anomaly
}
#[inline]
fn keplers_equation_derivative(eccentric_anomaly: f64, eccentricity: f64) -> f64 {
    1.0 - (eccentricity * eccentric_anomaly.cos())
}
#[inline]
fn keplers_equation_second_derivative(eccentric_anomaly: f64, eccentricity: f64) -> f64 {
    eccentricity * eccentric_anomaly.sin()
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

    ((e_x - e_neg_x) * 0.5, (e_x + e_neg_x) * 0.5)
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
    t - b / 3.0
}

/// See: https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
fn get_sqrt_mu_sma<T: OrbitTrait>(orbit: &T, g: f64) -> f64 {
    (orbit.get_mu(g) * orbit.get_semi_major_axis()).sqrt()
}

/// See: https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
fn get_velocity_unit_vector<T: OrbitTrait>(orbit: &T) -> DVec2 {
    let eccentricity: f64 = orbit.get_eccentricity();
    let eccentric_anomaly = orbit.get_eccentric_anomaly(orbit.get_mean_anomaly_at_epoch());
    DVec2 {
        x: -eccentric_anomaly.sin(),
        y: (1. - eccentricity.powi(2).sqrt() * eccentric_anomaly.cos()),
    }
}
