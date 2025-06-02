//! This module contains presets for stars.
//!
//! "A star is a luminous spheroid of plasma held together by self-gravity."  
//!
//! \- [Wikipedia](https://en.wikipedia.org/wiki/Star)

use crate::{Body, Orbit};

/// Returns the Sun.
///
/// `include_orbit`: Whether to include the orbit of the Sun around Sagittarius A*.
pub fn the_sun(include_orbit: bool) -> Body {
    let orbit = if include_orbit {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            2.36518e20,
            2.36518e20,
            // I can't seem to find the orientation of the Sun's orbit
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ))
    } else {
        None
    };

    return Body::new("The Sun".to_string(), 1.989e30, 6.9634e5, orbit);
}
