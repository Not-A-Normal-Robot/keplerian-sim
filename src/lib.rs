//! # Keplerian Orbital Mechanics
//! This library crate contains logic for Keplerian orbits, similar to the ones
//! you'd find in a game like Kerbal Space Program.  
//!
//! Keplerian orbits are special in that they are more stable and predictable than
//! Newtonian orbits. In fact, unlike Newtonian orbits, Keplerian orbits don't use
//! time steps to calculate the next position of an object. Keplerian orbits use
//! state vectors to determine the object's *full trajectory* at any given time.\
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
//!
//! We used to have `body`, `universe`, and `body_presets` modules, however these
//! were removed from the main library because some programs have different
//! needs on what to store on each body. The code was moved to the `simulate`
//! example file in the repository:
//! <https://github.com/Not-A-Normal-Robot/keplerian-sim/blob/0d60ed756dc6b09c60d779167cfa0e3346e09213/examples/simulate.rs>
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
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("Either std or libm must be used for math operations");

use glam::{DVec2, DVec3};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "libm")]
mod math;
#[cfg(feature = "libm")]
#[allow(unused_imports)]
use math::F64Math;

mod dim2;
mod dim3;
/// Re-exports for dependencies.
pub mod reexports;
mod solvers;

pub use dim2::{CompactOrbit2D, MuSetterMode2D, Orbit2D, OrbitTrait2D, StateVectors2D};
pub use dim3::{CompactOrbit, MuSetterMode, Orbit, OrbitTrait, StateVectors};

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
const B: f64 = 0.999_999;

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
    /// The zero matrix.
    ///
    /// Multiplying a vector with this results in a vector
    /// of zero length.
    pub const ZERO: Self = Self {
        e11: 0.0,
        e12: 0.0,
        e21: 0.0,
        e22: 0.0,
        e31: 0.0,
        e32: 0.0,
    };

    /// The identity matrix.
    ///
    /// Multiplying a vector with this results in the same
    /// vector.
    pub const IDENTITY: Self = Self {
        e11: 1.0,
        e22: 1.0,
        ..Self::ZERO
    };

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
    #[must_use]
    pub fn dot_vec(&self, vec: DVec2) -> DVec3 {
        DVec3::new(
            vec.x * self.e11 + vec.y * self.e12,
            vec.x * self.e21 + vec.y * self.e22,
            vec.x * self.e31 + vec.y * self.e32,
        )
    }
}

#[cfg(feature = "mint")]
impl From<Matrix3x2> for mint::RowMatrix3x2<f64> {
    fn from(value: Matrix3x2) -> Self {
        mint::RowMatrix3x2::<f64> {
            x: mint::Vector2::<f64> {
                x: value.e11,
                y: value.e12,
            },
            y: mint::Vector2::<f64> {
                x: value.e21,
                y: value.e22,
            },
            z: mint::Vector2::<f64> {
                x: value.e31,
                y: value.e32,
            },
        }
    }
}

#[cfg(feature = "mint")]
impl mint::IntoMint for Matrix3x2 {
    type MintType = mint::RowMatrix3x2<f64>;
}

#[cfg(feature = "mint")]
impl From<mint::RowMatrix3x2<f64>> for Matrix3x2 {
    fn from(value: mint::RowMatrix3x2<f64>) -> Self {
        Self {
            e11: value.x.x,
            e12: value.x.y,
            e21: value.y.x,
            e22: value.y.y,
            e31: value.z.x,
            e32: value.z.y,
        }
    }
}

/// An error to describe why setting the periapsis of an orbit failed.
#[derive(PartialEq, Eq, Debug, Clone, Copy)]
pub enum ApoapsisSetterError {
    /// ### Attempt to set apoapsis to a value less than periapsis.
    /// By definition, an orbit's apoapsis is the highest point in the orbit,
    /// and its periapsis is the lowest point in the orbit.\
    /// Therefore, it doesn't make sense for the apoapsis to be lower than the periapsis.
    ApoapsisLessThanPeriapsis,

    /// ### Attempt to set apoapsis to a negative value.
    /// By definition, the apoapsis is the highest point in the orbit.\
    /// You can't be a negative distance away from the center of mass of the parent body.\
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
#[must_use]
pub fn sinhcosh(x: f64) -> (f64, f64) {
    let e_x = x.exp();
    let e_neg_x = e_x.recip();

    ((e_x - e_neg_x) * 0.5, (e_x + e_neg_x) * 0.5)
}

/// Solve a cubic equation to get its real root.
///
/// The cubic equation is in the form of:
/// ax^3 + bx^2 + cx + d
///
/// The cubic equation is assumed to be monotone.\
/// If it isn't monotone (i.e., the discriminant
/// is negative), it may return an incorrect value
/// or NaN.
///
/// # Performance
/// This function involves several divisions,
/// squareroots, and cuberoots, and therefore is not
/// very performant. It is recommended to cache this value if you can.
#[expect(clippy::many_single_char_names)]
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
