//! This module contains presets for common planets.
//!
//! A planet is a celestial body that  
//! (a) is in orbit around the Sun,  
//! (b) has sufficient mass for its self-gravity to overcome
//!     rigid body forces so that it assumes a hydrostatic
//!     equilibrium (nearly round) shape, and  
//! (c) has cleared the neighbourhood around its orbit.
//!
//! \- [International Astronomical Union](https://en.wikipedia.org/wiki/IAU_definition_of_planet#Final_definition)

use crate::{Body, Orbit};

/// Returns Mercury, the closest planet to the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
pub fn mercury(parent_mu: Option<f64>) -> Body {
    let orbit = if let Some(mu) = parent_mu {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            6.982e7,
            4.6e7,
            7.005_f64.to_radians(),
            29.124_f64.to_radians(),
            48.331_f64.to_radians(),
            174.796_f64.to_radians(),
            mu,
        ))
    } else {
        None
    };

    return Body::new("Mercury".to_string(), 3.3011e23, 2.4397e6, orbit);
}

/// Returns Venus, the second planet from the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
pub fn venus(parent_mu: Option<f64>) -> Body {
    let orbit = if let Some(mu) = parent_mu {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            1.0894e8,
            1.0748e8,
            3.39458_f64.to_radians(),
            54.884_f64.to_radians(),
            76.680_f64.to_radians(),
            50.115_f64.to_radians(),
            mu,
        ))
    } else {
        None
    };

    return Body::new("Venus".to_string(), 4.8675e24, 6.0518e6, orbit);
}

/// Returns Earth, the third planet from the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
pub fn earth(parent_mu: Option<f64>) -> Body {
    let orbit = if let Some(mu) = parent_mu {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            1.52097597e11,
            1.47098450e11,
            0.00005_f64.to_radians(),
            114.20783_f64.to_radians(),
            -11.26064_f64.to_radians(),
            358.617_f64.to_radians(),
            mu,
        ))
    } else {
        None
    };

    return Body::new("Earth".to_string(), 5.972e24, 6.371e6, orbit);
}

/// Returns Mars, the fourth planet from the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
pub fn mars(parent_mu: Option<f64>) -> Body {
    let orbit = if let Some(mu) = parent_mu {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            2.49261e11,
            2.0665e11,
            1.850_f64.to_radians(),
            286.5_f64.to_radians(),
            49.57854_f64.to_radians(),
            19.412_f64.to_radians(),
            mu,
        ))
    } else {
        None
    };

    return Body::new("Mars".to_string(), 6.4171e23, 3.3895e6, orbit);
}

/// Returns Jupiter, the fifth planet from the Sun.  
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
pub fn jupiter(parent_mu: Option<f64>) -> Body {
    let orbit = if let Some(mu) = parent_mu {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            8.16363e11,
            7.40595e11,
            1.303_f64.to_radians(),
            273.867_f64.to_radians(),
            100.464_f64.to_radians(),
            20.02_f64.to_radians(),
            mu,
        ))
    } else {
        None
    };

    return Body::new("Jupiter".to_string(), 1.8982e27, 6.9911e7, orbit);
}

/// Returns Saturn, the sixth planet from the Sun.  
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
pub fn saturn(parent_mu: Option<f64>) -> Body {
    let orbit = if let Some(mu) = parent_mu {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            1.5145e12,
            1.35255e12,
            2.485_f64.to_radians(),
            339.392_f64.to_radians(),
            113.665_f64.to_radians(),
            317.020_f64.to_radians(),
            mu,
        ))
    } else {
        None
    };

    return Body::new("Saturn".to_string(), 5.6834e26, 58232.0, orbit);
}

/// Returns Uranus, the seventh planet from the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
pub fn uranus(parent_mu: Option<f64>) -> Body {
    let orbit = if let Some(mu) = parent_mu {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            3.00639e12,
            2.73556e12,
            0.773_f64.to_radians(),
            96.998857_f64.to_radians(),
            74.006_f64.to_radians(),
            142.2386_f64.to_radians(),
            mu,
        ))
    } else {
        None
    };

    return Body::new("Uranus".to_string(), 8.681e25, 2.5362e7, orbit);
}

/// Returns Neptune, the eighth planet from the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
pub fn neptune(parent_mu: Option<f64>) -> Body {
    let orbit = if let Some(mu) = parent_mu {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            4.54e12,
            4.46e12,
            1.77_f64.to_radians(),
            273.187_f64.to_radians(),
            131.783_f64.to_radians(),
            259.883_f64.to_radians(),
            mu,
        ))
    } else {
        None
    };

    return Body::new("Neptune".to_string(), 1.02409e26, 2.4341e7, orbit);
}
