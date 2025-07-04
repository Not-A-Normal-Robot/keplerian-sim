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
mod cached_orbit;
mod compact_orbit;
mod universe;

use std::f64::consts::{PI, TAU};

pub use body::Body;
pub use cached_orbit::Orbit;
pub use compact_orbit::CompactOrbit;
use glam::{DVec2, DVec3};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

pub use universe::Universe;

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
/// <https://doi.org/10.1051/0004-6361/202141423>
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
const N_F64: f64 = N_U32 as f64;

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
/// Namely, it is used in the [`transform_pqw_vector`][OrbitTrait::transform_pqw_vector]
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
/// let result = matrix.dot_vec(vec);
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
    /// let result = matrix.dot_vec(vec);
    ///
    /// assert_eq!(result, DVec3::new(1.0, 2.0, 3.0));
    /// ```
    pub fn dot_vec(&self, vec: DVec2) -> DVec3 {
        DVec3::new(
            vec.x * self.e11 + vec.y * self.e12,
            vec.x * self.e21 + vec.y * self.e22,
            vec.x * self.e31 + vec.y * self.e32,
        )
    }
}

/// A struct representing a position and velocity at a point in the orbit.
///
/// The position and velocity vectors are three-dimensional.
///
/// The position vector is in meters, while the velocity vector is in
/// meters per second.
///
/// State vectors can be used to form an orbit, see
/// [`to_orbit`][Self::to_compact_orbit] for more information.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StateVectors {
    /// The 3D position at a point in the orbit, in meters.
    pub position: DVec3,
    /// The 3D velocity at a point in the orbit, in meters per second.
    pub velocity: DVec3,
}

impl StateVectors {
    /// Create a new [`CompactOrbit`] struct from the state
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
    /// # Performance
    /// This function is not too performant as it uses several trigonometric operations.  
    ///
    /// For single conversions, this is faster than
    /// [the cached orbit converter][Self::to_cached_orbit].  
    /// However, consider using the cached orbit converter if you want to use the same orbit for
    /// many calculations, as the caching speed benefits should outgrow the small initialization
    /// overhead.
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
    /// use keplerian_sim::{CompactOrbit, OrbitTrait};
    ///
    /// let orbit = CompactOrbit::default();
    /// let mu = orbit.get_gravitational_parameter();
    ///
    /// let sv = orbit.get_state_vectors_at_time(0.0);
    ///
    /// let new_orbit = sv.to_compact_orbit(mu);
    ///
    /// assert_eq!(orbit.get_eccentricity(), new_orbit.get_eccentricity());
    /// assert_eq!(orbit.get_periapsis(), new_orbit.get_periapsis());
    /// ```
    /// To simulate a burn:
    /// ```
    /// use keplerian_sim::{CompactOrbit, OrbitTrait, StateVectors};
    /// use glam::DVec3;
    ///
    /// let orbit = CompactOrbit::default();
    /// let mu = orbit.get_gravitational_parameter();
    ///
    /// let sv = orbit.get_state_vectors_at_time(0.0);
    /// assert_eq!(
    ///     sv,
    ///     StateVectors {
    ///         position: DVec3::new(1.0, 0.0, 0.0),
    ///         velocity: DVec3::new(0.0, 1.0, 0.0),
    ///     }
    /// );
    ///
    /// let new_sv = StateVectors {
    ///     velocity: sv.velocity + DVec3::new(0.0, 0.1, 0.0),
    ///     ..sv
    /// };
    ///
    /// let new_orbit = new_sv.to_compact_orbit(mu);
    ///
    /// panic!("{new_orbit:?}");
    /// ```
    #[must_use]
    pub fn to_compact_orbit(self, mu: f64) -> CompactOrbit {
        // Reference:
        // https://orbital-mechanics.space/classical-orbital-elements/orbital-elements-and-the-state-vector.html
        // Note: That site doesn't use the same "base elements" and
        // conversions will need to be done at the end

        // Precalculated values
        let altitude = self.position.length();
        let altitude_recip = altitude.recip();
        let position_normal = self.position * altitude_recip;
        let mu_recip = mu.recip();

        // Step 1: Position and Velocity Magnitudes (i.e. speeds)
        let radial_speed = self.velocity.dot(position_normal);

        // Step 2: Orbital Angular Momentum
        let angular_momentum_vector = self.position.cross(self.velocity);
        let angular_momentum = angular_momentum_vector.length();

        // Step 3: Inclination
        let inclination = (angular_momentum_vector.z / angular_momentum).acos();

        // Step 4: Right Ascension of the Ascending Node
        // Here we use René Schwarz's simplification of the cross product
        // between (0, 0, 1) and the angular momentum vector, outlined in:
        // https://downloads.rene-schwarz.com/download/M002-Cartesian_State_Vectors_to_Keplerian_Orbit_Elements.pdf
        let asc_vec2 = DVec2::new(-angular_momentum_vector.y, angular_momentum_vector.x);
        let asc_len = asc_vec2.length();
        let asc_len_recip = asc_len.recip();
        let asc_vec3 = asc_vec2.extend(0.0);

        let long_asc_node = {
            let tmp = (asc_vec3.x * asc_len_recip).acos();
            if asc_vec3.y >= 0.0 {
                tmp
            } else {
                TAU - tmp
            }
        };

        // Step 5: Eccentricity
        let eccentricity_vector =
            self.velocity.cross(angular_momentum_vector) * mu_recip - position_normal;
        let eccentricity = eccentricity_vector.length();
        let eccentricity_recip = eccentricity.recip();

        // Step 6: Argument of Periapsis
        let arg_pe = {
            let tmp =
                (eccentricity_vector.dot(asc_vec3) * eccentricity_recip * asc_len_recip).acos();
            if eccentricity_vector.z >= 0.0 {
                tmp
            } else {
                TAU - tmp
            }
        };

        // Step 7: True anomaly
        let true_anomaly = {
            let tmp =
                (eccentricity_vector.dot(self.position) * eccentricity_recip * altitude_recip)
                    .acos();
            if radial_speed >= 0.0 {
                tmp
            } else {
                TAU - tmp
            }
        };

        // Now we convert those elements into our desired form
        // First we need to convert `h` (Orbital Angular Momentum)
        // into periapsis altitude, then we need to convert the true anomaly
        // to a mean anomaly, or to a "time at periapsis" value for parabolic
        // orbits.
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

    /// Create a new [`Orbit`] struct from the state
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
    /// # Parabolic Support
    /// This function does not yet support parabolic trajectories.
    /// Non-finite values may be returned for such cases.
    ///
    /// # Constraints
    /// The position must not be at the origin, and the velocity must not be at zero.  
    /// If this constraint is breached, you may get invalid values such as infinities
    /// or NaNs.
    #[must_use]
    pub fn to_cached_orbit(self, mu: f64) -> Orbit {
        self.to_compact_orbit(mu).into()
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
    /// # Performance
    /// This function is not too performant as it uses several trigonometric operations.
    ///
    /// The performance also depends on how fast the specified orbit type can convert
    /// between the [`CompactOrbit`] form into itself, and so we cannot guarantee any
    /// performance behaviors.
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
    pub fn to_custom_orbit<O>(self, mu: f64) -> O
    where
        O: From<CompactOrbit> + OrbitTrait,
    {
        self.to_compact_orbit(mu).into()
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
    /// let mut orbit = Orbit::default();
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

    /// Gets the semi-latus rectum of the orbit.
    ///
    /// Learn more: <https://en.wikipedia.org/wiki/Ellipse#Semi-latus_rectum>  
    /// <https://en.wikipedia.org/wiki/Conic_section#Conic_parameters>
    ///
    /// # Performance
    /// This function is very performant and should not be the cause of any
    /// performance issues.
    fn get_semi_latus_rectum(&self) -> f64 {
        let eccentricity = self.get_eccentricity();
        if eccentricity == 1.0 {
            2.0 * self.get_periapsis()
        } else {
            self.get_semi_major_axis() * (1.0 - eccentricity.powi(2))
        }
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
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut base = Orbit::default();
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

    /// Gets the transformation matrix needed to tilt a 2D vector into the
    /// tilted orbital plane.
    ///
    /// # Performance
    /// For [`CompactOrbit`], this will perform a few trigonometric operations.  
    /// If you need this value often, consider using [the cached orbit struct][crate::Orbit] instead.
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

    // TODO: POST-PARABOLIC SUPPORT: Add note about parabolic eccentric anomaly (?), remove parabolic support sections
    /// Gets the eccentric anomaly at a given mean anomaly in the orbit.
    ///
    /// When the orbit is open (has an eccentricity of at least 1),
    /// the [hyperbolic eccentric anomaly](https://space.stackexchange.com/questions/27602/what-is-hyperbolic-eccentric-anomaly-f)
    /// would be returned instead.
    ///
    /// # Parabolic Support
    /// This function doesn't yet support parabolic trajectories. It may return `NaN`s
    /// or nonsensical values.
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
        if self.get_eccentricity() < 1.0 {
            self.get_eccentric_anomaly_elliptic(mean_anomaly)
        } else {
            self.get_eccentric_anomaly_hyperbolic(mean_anomaly)
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
    fn get_approx_hyperbolic_eccentric_anomaly(&self, mean_anomaly: f64) -> f64 {
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
    fn get_eccentric_anomaly_hyperbolic(&self, mean_anomaly: f64) -> f64 {
        let mut ecc_anom = self.get_approx_hyperbolic_eccentric_anomaly(mean_anomaly);

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
    fn get_eccentric_anomaly_at_time(&self, t: f64) -> f64 {
        self.get_eccentric_anomaly_at_mean_anomaly(self.get_mean_anomaly_at_time(t))
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
    /// [`get_true_anomaly_at_eccentric_anomaly`][OrbitTrait::get_true_anomaly_at_eccentric_anomaly]
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
    /// [`get_true_anomaly_at_eccentric_anomaly`][OrbitTrait::get_true_anomaly_at_eccentric_anomaly]
    /// instead.
    ///
    /// If you already know the mean anomaly, consider using
    /// [`get_true_anomaly_at_mean_anomaly`][OrbitTrait::get_true_anomaly_at_mean_anomaly]
    /// instead.  
    /// It won't help performance much, but it's not zero.
    fn get_true_anomaly_at_time(&self, t: f64) -> f64 {
        self.get_true_anomaly_at_mean_anomaly(self.get_mean_anomaly_at_time(t))
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
    fn get_mean_anomaly_at_time(&self, t: f64) -> f64 {
        t * (self.get_gravitational_parameter() / self.get_semi_major_axis().powi(3).abs()).sqrt()
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
    /// [`get_mean_anomaly_at_elliptic_eccentric_anomaly`][OrbitTrait::get_mean_anomaly_at_elliptic_eccentric_anomaly]
    /// and
    /// [`get_mean_anomaly_at_hyperbolic_eccentric_anomaly`][OrbitTrait::get_mean_anomaly_at_hyperbolic_eccentric_anomaly].  
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
    /// [`get_mean_anomaly_at_eccentric_anomaly`][OrbitTrait::get_mean_anomaly_at_eccentric_anomaly]
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
    /// This function benefits significantly from being in the
    /// [cached version of the orbit struct][crate::Orbit].  
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_true_anomaly`][OrbitTrait::get_state_vectors_at_true_anomaly]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    ///
    /// If you already know the altitude at the angle, you can
    /// rotate the altitude using the true anomaly, then tilt
    /// it using the [`transform_pqw_vector`][OrbitTrait::transform_pqw_vector]
    /// function instead.
    ///
    /// # Example
    /// ```
    /// use glam::DVec3;
    ///
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
    /// orbit.set_periapsis(100.0);
    /// orbit.set_eccentricity(0.0);
    ///
    /// let pos = orbit.get_position_at_true_anomaly(0.0);
    ///
    /// assert_eq!(pos, DVec3::new(100.0, 0.0, 0.0));
    /// ```
    #[doc(alias = "get_position_at_angle")]
    fn get_position_at_true_anomaly(&self, angle: f64) -> DVec3 {
        self.transform_pqw_vector(self.get_pqw_position_at_true_anomaly(angle))
    }

    /// Gets the 3D position at a given eccentric anomaly in the orbit.
    ///
    /// # Performance
    /// This function benefits significantly from being in the
    /// [cached version of the orbit struct][crate::Orbit].  
    /// This function is not too performant as it uses a few trigonometric
    /// operations. It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you already know the true anomaly, you can use the
    /// [`get_position_at_true_anomaly`][OrbitTrait::get_position_at_true_anomaly]
    /// function instead.  
    /// Or, if you only need the altitude, use the
    /// [`get_altitude_at_eccentric_anomaly`][OrbitTrait::get_altitude_at_eccentric_anomaly]
    /// function instead.
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_eccentric_anomaly`][OrbitTrait::get_state_vectors_at_eccentric_anomaly]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    fn get_position_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> DVec3 {
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
    /// you can use the [`get_speed_at_altitude`][OrbitTrait::get_speed_at_altitude]
    /// function instead to skip some calculations.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// const PERIAPSIS: f64 = 100.0;
    ///
    /// let mut orbit = Orbit::default();
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
        // v = sqrt(GM * (2/r - 1/a))

        let r = altitude;
        let a = self.get_semi_major_axis();

        ((2.0 / r - a.recip()) * self.get_gravitational_parameter()).sqrt()
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
    /// [`get_speed_at_eccentric_anomaly`][OrbitTrait::get_speed_at_eccentric_anomaly]
    /// and
    /// [`get_speed_at_true_anomaly`][OrbitTrait::get_speed_at_true_anomaly]
    /// functions instead.  
    /// Those do not use numerical methods and therefore are a lot faster.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_speed_at_time(&self, t: f64) -> f64 {
        self.get_speed_at_true_anomaly(self.get_true_anomaly_at_time(t))
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
    /// [`get_speed_at_true_anomaly`][OrbitTrait::get_speed_at_true_anomaly]
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
    /// [`get_velocity_at_true_anomaly`][OrbitTrait::get_velocity_at_true_anomaly] instead.
    ///
    /// # Performance
    /// This function is not too performant as it uses some trigonometric operations.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you only want to know the speed, use
    /// [`get_speed_at_true_anomaly`][OrbitTrait::get_speed_at_true_anomaly] instead.  
    /// And if you already know the eccentric anomaly, use
    /// [`get_pqw_velocity_at_eccentric_anomaly`][OrbitTrait::get_pqw_velocity_at_eccentric_anomaly]
    /// instead.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
    /// [`get_velocity_at_eccentric_anomaly`][OrbitTrait::get_velocity_at_eccentric_anomaly] instead.
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
    /// [`get_speed_at_eccentric_anomaly`][OrbitTrait::get_speed_at_eccentric_anomaly]
    /// function instead.
    ///
    /// Alternatively, if you already know some values (such as the altitude), consider
    /// using the unchecked version of the function instead:  
    /// [`get_pqw_velocity_at_eccentric_anomaly_unchecked`][OrbitTrait::get_pqw_velocity_at_eccentric_anomaly_unchecked]
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
    /// inputs passed to this function, and it also doesn't check
    /// that the orbit is elliptic.  
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
    /// [`transform_pqw_vector`][OrbitTrait::transform_pqw_vector].
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
    /// use keplerian_sim::{Orbit, OrbitTrait, sinhcosh};
    ///
    /// # fn main() {
    /// let orbit = Orbit::default();
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
    /// use keplerian_sim::{Orbit, OrbitTrait, sinhcosh};
    ///
    /// # fn main() {
    /// let mut hyperbolic = Orbit::default();
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
    /// [`get_velocity_at_time`][OrbitTrait::get_velocity_at_time] instead.
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
    /// [`get_pqw_velocity_at_eccentric_anomaly`][OrbitTrait::get_pqw_velocity_at_eccentric_anomaly]
    /// and
    /// [`get_pqw_velocity_at_true_anomaly`][OrbitTrait::get_pqw_velocity_at_true_anomaly]
    /// functions instead.  
    /// Those do not use numerical methods and therefore are a lot faster.
    #[doc(alias = "get_flat_velocity_at_time")]
    fn get_pqw_velocity_at_time(&self, t: f64) -> DVec2 {
        self.get_pqw_velocity_at_eccentric_anomaly(self.get_eccentric_anomaly_at_time(t))
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
    /// [`get_position_at_true_anomaly`][OrbitTrait::get_position_at_true_anomaly] instead.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Performance
    /// This function is somewhat performant. However, if you already know
    /// the altitude beforehand, you might be interested in the unchecked
    /// version of this function:
    /// [`get_pqw_position_at_true_anomaly_unchecked`][OrbitTrait::get_pqw_position_at_true_anomaly_unchecked]  
    /// If you're looking to just get the altitude at a given angle,
    /// consider using the [`get_altitude_at_true_anomaly`][OrbitTrait::get_altitude_at_true_anomaly]
    /// function instead.
    ///
    /// # Example
    /// ```
    /// use glam::DVec2;
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
    /// [`transform_pqw_vector`][OrbitTrait::transform_pqw_vector] function.
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
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
    /// [`get_position_at_time`][OrbitTrait::get_position_at_time] instead.
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
    /// [`get_pqw_position_at_eccentric_anomaly`][OrbitTrait::get_pqw_position_at_eccentric_anomaly]
    /// and
    /// [`get_pqw_position_at_true_anomaly`][OrbitTrait::get_pqw_position_at_true_anomaly]
    /// functions instead.
    /// Those do not use numerical methods and therefore are a lot faster.
    #[doc(alias = "get_flat_position_at_time")]
    fn get_pqw_position_at_time(&self, t: f64) -> DVec2 {
        self.get_pqw_position_at_true_anomaly(self.get_true_anomaly_at_time(t))
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
    /// [`get_position_at_eccentric_anomaly`][OrbitTrait::get_position_at_eccentric_anomaly] instead.
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
    /// [`get_pqw_position_at_true_anomaly`][OrbitTrait::get_pqw_position_at_true_anomaly]
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
    /// [`get_speed_at_true_anomaly`][OrbitTrait::get_speed_at_true_anomaly] instead.  
    /// Or, if you already have the eccentric anomaly, use
    /// [`get_velocity_at_eccentric_anomaly`][OrbitTrait::get_velocity_at_eccentric_anomaly]
    /// instead.
    /// These functions do not require numerical methods and therefore are a lot faster.
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_true_anomaly`][OrbitTrait::get_state_vectors_at_true_anomaly]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
    fn get_velocity_at_true_anomaly(&self, angle: f64) -> DVec3 {
        self.transform_pqw_vector(self.get_pqw_velocity_at_true_anomaly(angle))
    }

    /// Gets the velocity at a given eccentric anomaly in the orbit.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
    /// [`get_speed_at_eccentric_anomaly`][OrbitTrait::get_speed_at_eccentric_anomaly]
    /// function instead.
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_eccentric_anomaly`][OrbitTrait::get_state_vectors_at_eccentric_anomaly]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    ///
    /// This function benefits significantly from being in the
    /// [cached version of the orbit struct][crate::Orbit].
    fn get_velocity_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> DVec3 {
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
    /// [`get_speed_at_time`][OrbitTrait::get_speed_at_time] instead.  
    /// Or, if you already have the eccentric anomaly or true anomaly, use the
    /// [`get_velocity_at_eccentric_anomaly`][OrbitTrait::get_velocity_at_eccentric_anomaly]
    /// and
    /// [`get_velocity_at_true_anomaly`][OrbitTrait::get_velocity_at_true_anomaly]
    /// functions instead.  
    /// These functions do not require numerical methods and therefore are a lot faster.
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_time`][OrbitTrait::get_state_vectors_at_time]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    ///
    /// # Speed vs. Velocity
    /// Speed is not to be confused with velocity.  
    /// Speed tells you how fast something is moving,
    /// while velocity tells you how fast *and in what direction* it's moving in.
    fn get_velocity_at_time(&self, t: f64) -> DVec3 {
        self.get_velocity_at_eccentric_anomaly(self.get_eccentric_anomaly_at_time(t))
    }

    /// Gets the altitude of the body from its parent at a given angle (true anomaly) in the orbit.
    ///
    ///
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Performance
    /// This function is performant, however, if you already
    /// know the orbit's semi-latus rectum or the cosine of the true anomaly,
    /// you can use the
    /// [`get_altitude_at_true_anomaly_unchecked`][OrbitTrait::get_altitude_at_true_anomaly_unchecked]
    /// function to skip a few steps in the calculation.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
    /// # Angle
    /// The angle is expressed in radians, and ranges from 0 to tau.  
    /// Anything out of range will get wrapped around.
    ///
    /// # Performance
    /// This function, by itself, is performant and is unlikely
    /// to be the culprit of any performance issues.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
        (semi_latus_rectum / (1.0 + self.get_eccentricity() * cos_true_anomaly)).abs()
    }

    /// Gets the altitude at a given eccentric anomaly in the orbit.
    ///
    /// # Performance
    /// This function is not too performant as it uses a few trigonometric operations.
    /// It is recommended to cache this value if you can.
    ///
    /// Alternatively, if you already know the true anomaly, use the
    /// [`get_altitude_at_true_anomaly`][OrbitTrait::get_altitude_at_true_anomaly]
    /// function instead.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit, OrbitTrait};
    ///
    /// let mut orbit = Orbit::default();
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
    /// # Performance
    /// This involves calculating the true anomaly at a given time, and so is not very performant.  
    /// It is recommended to cache this value when possible.
    ///
    /// Alternatively, if you already know the eccentric anomaly or the true anomaly,
    /// consider using the
    /// [`get_altitude_at_eccentric_anomaly`][OrbitTrait::get_altitude_at_eccentric_anomaly]
    /// and
    /// [`get_altitude_at_true_anomaly`][OrbitTrait::get_altitude_at_true_anomaly]
    /// functions instead.  
    /// Those do not use numerical methods and therefore are a lot faster.
    ///
    /// # Parabolic Support
    /// **This function returns infinity for parabolic orbits** due to how the equation for
    /// true anomaly works.
    fn get_altitude_at_time(&self, t: f64) -> f64 {
        self.get_altitude_at_true_anomaly(self.get_true_anomaly_at_time(t))
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
    /// [cached version of the orbit struct][crate::Orbit].  
    ///
    /// Alternatively, if you already know the true anomaly,
    /// consider using the
    /// [`get_position_at_true_anomaly`][OrbitTrait::get_position_at_true_anomaly]
    /// function instead.  
    /// That does not use numerical methods and therefore is a lot faster.
    ///
    /// If you want to get both the position and velocity vectors, you can
    /// use the
    /// [`get_state_vectors_at_time`][OrbitTrait::get_state_vectors_at_time]
    /// function instead. It prevents redundant calculations and is therefore
    /// faster than calling the position and velocity functions separately.
    ///
    /// # Parabolic Support
    /// **This function returns non-finite numbers for parabolic orbits**
    /// due to how the equation for true anomaly works.
    fn get_position_at_time(&self, t: f64) -> DVec3 {
        self.get_position_at_true_anomaly(self.get_true_anomaly_at_time(t))
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
    /// [`get_velocity_at_eccentric_anomaly`][OrbitTrait::get_velocity_at_eccentric_anomaly]  
    /// [`get_position_at_eccentric_anomaly`][OrbitTrait::get_position_at_eccentric_anomaly]  
    ///
    /// This function should give similar performance to the getter from the true anomaly:  
    /// [`get_state_vectors_at_true_anomaly`][OrbitTrait::get_state_vectors_at_true_anomaly]
    ///
    /// In case you really want to, an unchecked version of this function is available:  
    /// [`get_state_vectors_from_unchecked_parts`][OrbitTrait::get_state_vectors_from_unchecked_parts]
    ///
    /// # Parabolic Support
    /// This function doesn't support parabolic trajectories yet.  
    /// `NaN`s or nonsensical values may be returned.
    fn get_state_vectors_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> StateVectors {
        let semi_major_axis = self.get_semi_major_axis();
        let sqrt_abs_gm_a = (semi_major_axis * self.get_gravitational_parameter())
            .abs()
            .sqrt();
        let true_anomaly = self.get_true_anomaly_at_eccentric_anomaly(eccentric_anomaly);
        let sincos_angle = true_anomaly.sin_cos();
        let eccentricity = self.get_eccentricity();

        // 1 minus e^2
        let _1me2 = 1.0 - eccentricity.powi(2);

        // inlined version of [self.get_semi_latus_rectum] with pre-known values
        let semi_latus_rectum = if eccentricity == 1.0 {
            2.0 * self.get_periapsis()
        } else {
            semi_major_axis * _1me2
        };

        let altitude =
            self.get_altitude_at_true_anomaly_unchecked(semi_latus_rectum, sincos_angle.1);

        let q_mult = _1me2.abs().sqrt();

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
    /// [`get_velocity_at_true_anomaly`][OrbitTrait::get_velocity_at_true_anomaly]
    /// [`get_position_at_true_anomaly`][OrbitTrait::get_position_at_true_anomaly]
    ///
    /// This function should give similar performance to the getter from the eccentric anomaly:  
    /// [`get_state_vectors_at_eccentric_anomaly`][OrbitTrait::get_state_vectors_at_true_anomaly]
    ///
    /// In case you really want to, an unchecked version of this function is available:  
    /// [`get_state_vectors_from_unchecked_parts`][OrbitTrait::get_state_vectors_from_unchecked_parts]
    ///
    /// # Parabolic Support
    /// This function doesn't support parabolic trajectories yet.  
    /// `NaN`s or nonsensical values may be returned.
    fn get_state_vectors_at_true_anomaly(&self, true_anomaly: f64) -> StateVectors {
        let semi_major_axis = self.get_semi_major_axis();
        let sqrt_abs_gm_a = (semi_major_axis * self.get_gravitational_parameter())
            .abs()
            .sqrt();
        let eccentric_anomaly = self.get_eccentric_anomaly_at_true_anomaly(true_anomaly);
        let sincos_angle = true_anomaly.sin_cos();
        let eccentricity = self.get_eccentricity();

        // 1 minus e^2
        let _1me2 = 1.0 - eccentricity.powi(2);

        // inlined version of [self.get_semi_latus_rectum] with pre-known values
        let semi_latus_rectum = if eccentricity == 1.0 {
            2.0 * self.get_periapsis()
        } else {
            semi_major_axis * _1me2
        };

        let altitude =
            self.get_altitude_at_true_anomaly_unchecked(semi_latus_rectum, sincos_angle.1);

        let q_mult = _1me2.abs().sqrt();

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
    /// [`get_state_vectors_at_eccentric_anomaly`][OrbitTrait::get_state_vectors_at_eccentric_anomaly]
    /// [`get_state_vectors_at_true_anomaly`][OrbitTrait::get_state_vectors_at_true_anomaly]
    ///
    /// This function is faster than individually calling the position and velocity getters
    /// separately, as this will reuse calculations whenever possible.
    ///
    /// # Parabolic Support
    /// This function doesn't support parabolic trajectories yet.  
    /// `NaN`s or nonsensical values may be returned.
    fn get_state_vectors_at_mean_anomaly(&self, mean_anomaly: f64) -> StateVectors {
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
    /// [`get_state_vectors_at_eccentric_anomaly`][OrbitTrait::get_state_vectors_at_eccentric_anomaly]
    /// [`get_state_vectors_at_true_anomaly`][OrbitTrait::get_state_vectors_at_true_anomaly]
    ///
    /// If you only know the mean anomaly, then that may help with performance a little bit,
    /// in which case you can use
    /// [`get_state_vectors_at_mean_anomaly`][OrbitTrait::get_state_vectors_at_mean_anomaly]
    /// instead.
    ///
    /// This function is faster than individually calling the position and velocity getters
    /// separately, as this will reuse calculations whenever possible.
    ///
    /// If you need *only one* of the vectors, though, you should instead call the dedicated
    /// getters:  
    /// [`get_velocity_at_time`][OrbitTrait::get_velocity_at_time]
    /// [`get_position_at_time`][OrbitTrait::get_position_at_time]
    ///
    /// # Parabolic Support
    /// This function doesn't support parabolic trajectories yet.  
    /// `NaN`s or nonsensical values may be returned.
    fn get_state_vectors_at_time(&self, t: f64) -> StateVectors {
        self.get_state_vectors_at_mean_anomaly(self.get_mean_anomaly_at_time(t))
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
    /// The altitude at that point in the orbit, in meters.
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
    /// use keplerian_sim::{Orbit, OrbitTrait, StateVectors};
    ///
    /// # fn main() {
    /// // Elliptic (circular) case
    ///
    /// let orbit = Orbit::default();
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
    /// let sv = StateVectors {
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
    /// use keplerian_sim::{Orbit, OrbitTrait, sinhcosh, StateVectors};
    ///
    /// # fn main() {
    /// // Hyperbolic case
    ///
    /// let mut orbit = Orbit::default();
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
    /// let sv = StateVectors {
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
    ) -> StateVectors {
        let outer_mult = sqrt_abs_gm_a / altitude;
        let pqw_velocity =
            self.get_pqw_velocity_at_eccentric_anomaly_unchecked(outer_mult, q_mult, trig_ecc_anom);
        let pqw_position = self.get_pqw_position_at_true_anomaly_unchecked(altitude, sincos_angle);
        StateVectors {
            position: self.transform_pqw_vector(pqw_position),
            velocity: self.transform_pqw_vector(pqw_velocity),
        }
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
    /// [`Orbit`] struct, as it doesn't need to recalculate the transformation
    /// matrix needed to transform 2D vector.
    fn transform_pqw_vector(&self, position: DVec2) -> DVec3 {
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
    #[doc(alias = "set_mu")]
    fn set_gravitational_parameter(&mut self, gravitational_parameter: f64, mode: MuSetterMode);

    /// Gets the time it takes to complete one revolution of the orbit.
    ///
    /// This function returns infinite values for parabolic trajectories and
    /// NaN for hyperbolic trajectories.
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Performance
    /// This function is performant and should not be the cause of any
    /// performance issues.
    fn get_orbital_period(&self) -> f64 {
        // T = 2pi * sqrt(a^3 / GM)
        // https://en.wikipedia.org/wiki/Orbital_period
        TAU * (self.get_semi_major_axis().powi(3) / self.get_gravitational_parameter()).sqrt()
    }
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
pub fn sinhcosh(x: f64) -> (f64, f64) {
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
///
/// # Performance
/// This function involves several divisions,
/// squareroots, and cuberoots, and therefore is not
/// very performant. It is recommended to cache this value if you can.
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

mod generated_sinh_approximator;
