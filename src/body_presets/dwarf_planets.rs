//! This module contains presets for dwarf planets.
//!
//! A dwarf planet is a celestial body that:  
//! (a) is in orbit around the Sun,  
//! (b) has sufficient mass for its self-gravity to
//!     overcome rigid body forces so that it assumes
//!     a hydrostatic equilibrium (nearly round) shape,  
//! (c) has not cleared the neighbourhood around its orbit, and  
//! (d) is not a satellite.
//!
//! \- [International Astronomical Union](https://en.wikipedia.org/wiki/IAU_definition_of_planet#Final_definition)

use crate::{Body, Orbit, OrbitTrait};

use super::{MASS_ERIS, MASS_PLUTO, MASS_QUAOAR, MASS_SUN};

/// Returns 1 Ceres, a dwarf planet in the asteroid belt.  
///
/// `include_orbit`: Whether to include the orbit of Ceres around the Sun.
pub fn ceres(include_orbit: bool) -> Body {
    let orbit = if include_orbit {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            4.46e11,
            3.81e11,
            10.6_f64.to_radians(),
            73.6_f64.to_radians(),
            80.3_f64.to_radians(),
            291.4_f64.to_radians(),
            MASS_SUN,
        ))
    } else {
        None
    };

    Body::new("Ceres".to_string(), 9.3839e20, 939.4 / 2.0, orbit)
}

/// Returns 50000 Quaoar, a dwarf planet in the Kuiper belt.
///
/// `include_orbit`: Whether to include the orbit of Quaoar around the Sun.
pub fn quaoar(include_orbit: bool) -> Body {
    let orbit = if include_orbit {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            6.805e12,
            6.268e12,
            7.9895_f64.to_radians(),
            147.48_f64.to_radians(),
            188.927_f64.to_radians(),
            301.104_f64.to_radians(),
            MASS_SUN,
        ))
    } else {
        None
    };

    Body::new("Quaoar".to_string(), MASS_QUAOAR, 5.45e5, orbit)
}

/// Returns 90377 Sedna, a dwarf planet, sednoid, and extreme trans-Neptunian object.
///
/// `include_orbit`: Whether to include the orbit of Sedna around the Sun.
pub fn sedna(include_orbit: bool) -> Body {
    let orbit = if include_orbit {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            1.4e14,
            1.14e13,
            11.9307_f64.to_radians(),
            311.352_f64.to_radians(),
            144.248_f64.to_radians(),
            358.117_f64.to_radians(),
            MASS_SUN,
        ))
    } else {
        None
    };

    Body::new(
        "Sedna".to_string(),
        // Sedna's mass has not been directly measured.
        // An estimate is used instead.
        2e21,
        5e5,
        orbit,
    )
}

/// Returns 134340 Pluto, a famous dwarf planet in the Kuiper belt.  
///
/// `include_orbit`: Whether to include the orbit of Pluto around the Sun.
pub fn pluto(include_orbit: bool) -> Body {
    let orbit = if include_orbit {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            7.37593e12,
            4.43682e12,
            17.16_f64.to_radians(),
            113.834_f64.to_radians(),
            110.299_f64.to_radians(),
            14.53_f64.to_radians(),
            MASS_SUN,
        ))
    } else {
        None
    };

    Body::new("Pluto".to_string(), MASS_PLUTO, 1.1883e6, orbit)
}

/// Returns 136108 Haumea, a dwarf planet in the Kuiper belt.  
///
/// `include_orbit`: Whether to include the orbit of Haumea around the Sun.
pub fn haumea(include_orbit: bool) -> Body {
    let orbit = if include_orbit {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            7.717e12,
            5.1831e12,
            28.2137_f64.to_radians(),
            239.041_f64.to_radians(),
            122.167_f64.to_radians(),
            218.205_f64.to_radians(),
            MASS_SUN,
        ))
    } else {
        None
    };

    Body::new("Haumea".to_string(), 4e21, 7.8e5, orbit)
}

/// Returns 136199 Eris, a dwarf planet, and a trans-Neptunian and scattered disc object.  
///
/// `include_orbit`: Whether to include the orbit of Eris around the Sun.
pub fn eris(include_orbit: bool) -> Body {
    let orbit = if include_orbit {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            1.4579e13,
            5.725e12,
            44.04_f64.to_radians(),
            151.639_f64.to_radians(),
            35.951_f64.to_radians(),
            205.989_f64.to_radians(),
            MASS_SUN,
        ))
    } else {
        None
    };

    Body::new("Eris".to_string(), MASS_ERIS, 1.163e6, orbit)
}

/// Returns 136472 Makemake, a dwarf planet in the Kuiper belt.  
///
/// `include_orbit`: Whether to include the orbit of Makemake around the Sun.
pub fn makemake(include_orbit: bool) -> Body {
    let orbit = if include_orbit {
        // Source: Wikipedia
        Some(Orbit::with_apoapsis(
            7.8922e12,
            5.7003e12,
            28.9835_f64.to_radians(),
            294.834_f64.to_radians(),
            79.62_f64.to_radians(),
            165.514_f64.to_radians(),
            MASS_SUN,
        ))
    } else {
        None
    };

    Body::new("Makemake".to_string(), 3.1e21, 7.15e5, orbit)
}
