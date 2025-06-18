//! This module contains presets for stars.
//!
//! "A star is a luminous spheroid of plasma held together by self-gravity."  
//!
//! \- [Wikipedia](https://en.wikipedia.org/wiki/Star)

use crate::{Body, Orbit};

/// Returns the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
pub fn the_sun(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            2.36518e20, 2.36518e20,
            // I can't seem to find the orientation of the Sun's orbit
            0.0, 0.0, 0.0, 0.0, mu,
        )
    });

    Body::new("The Sun".to_string(), 1.989e30, 6.9634e5, orbit)
}
