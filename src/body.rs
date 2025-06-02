use crate::{Orbit, OrbitTrait};
use core::f64::consts::TAU;

/// A struct representing a celestial body.
#[derive(Clone, Debug, PartialEq)]
pub struct Body {
    /// The name of the celestial body.
    pub name: String,

    /// The mass of the celestial body, in kilograms.
    pub mass: f64,

    /// The radius of the celestial body, in meters.
    pub radius: f64,

    /// The orbit of the celestial body, if it is orbiting one.
    pub orbit: Option<Orbit>,
}

impl Body {
    /// Creates a new `Body` instance.
    ///
    /// # Arguments
    ///
    /// * `name` - The name of the celestial body.
    /// * `mass` - The mass of the celestial body, in kilograms.
    /// * `radius` - The radius of the celestial body, in meters.
    /// * `orbit` - An optional orbit for the celestial body.
    ///
    /// # Returns
    ///
    /// A new `Body` instance.
    pub fn new(name: String, mass: f64, radius: f64, orbit: Option<Orbit>) -> Body {
        return Body {
            name,
            mass,
            radius,
            orbit,
        };
    }

    /// Creates a default `Body` instance.
    ///
    /// Currently, this function returns the Earth.  
    /// However, do not rely on this behavior, as it may change in the future.
    pub fn new_default() -> Body {
        return Body {
            name: "Earth".to_string(),
            mass: 5.972e24,
            radius: 6.371e6,
            orbit: None,
        };
    }

    /// Releases the body from its orbit.
    pub fn release_from_orbit(&mut self) {
        self.orbit = None;
    }

    // TODO: This should be part of the OrbitTrait
    /// Get the amount of time it takes for the body to complete one orbit,
    /// given a gravitational constant.
    pub fn get_orbital_period(&self, g: f64) -> Option<f64> {
        let orbit = self.orbit.as_ref()?;
        let mu = g * self.mass;

        if orbit.get_eccentricity() >= 1.0 {
            return Some(core::f64::INFINITY);
        }

        let semi_major_axis = orbit.get_semi_major_axis();

        return Some(TAU * (semi_major_axis / mu).sqrt());
    }
}
