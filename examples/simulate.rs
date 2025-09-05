use keplerian_sim::{MuSetterMode, Orbit, OrbitTrait};
use std::io::Write;

const SIMULATION_TICKS: u128 = 1_000_000;
fn main() {
    let mut universe = generate_solar_system();
    describe_universe(&universe);
    let mut lock = io::stdout().lock();
    eprintln!("Simulating {SIMULATION_TICKS} ticks...");
    for t in 0..SIMULATION_TICKS {
        universe.tick();
        writeln!(&mut lock, "=== Tick {t} ===").unwrap();
        print_all_body_positions(&mut lock, &universe);
    }
}

fn generate_solar_system<'a>() -> Universe {
    let mut universe = Universe::default();

    let sun = the_sun(None);
    let sun_mu = sun.mass * universe.g;
    let sun_idx = universe.add_body(sun, None).unwrap();

    let mercury = mercury(Some(sun_mu));
    universe.add_body(mercury, Some(sun_idx)).unwrap();

    let venus = venus(Some(sun_mu));
    universe.add_body(venus, Some(sun_idx)).unwrap();

    let earth = earth(Some(sun_mu));
    let earth_mu = earth.mass * universe.g;
    let earth_idx = universe.add_body(earth, Some(sun_idx)).unwrap();

    let moon = the_moon(Some(earth_mu));
    universe.add_body(moon, Some(earth_idx)).unwrap();

    let mars = mars(Some(sun_mu));
    universe.add_body(mars, Some(sun_idx)).unwrap();

    let ceres = ceres(Some(sun_mu));
    universe.add_body(ceres, Some(sun_idx)).unwrap();

    let jupiter = jupiter(Some(sun_mu));
    universe.add_body(jupiter, Some(sun_idx)).unwrap();

    let saturn = saturn(Some(sun_mu));
    universe.add_body(saturn, Some(sun_idx)).unwrap();

    let uranus = uranus(Some(sun_mu));
    universe.add_body(uranus, Some(sun_idx)).unwrap();

    let neptune = neptune(Some(sun_mu));
    universe.add_body(neptune, Some(sun_idx)).unwrap();

    let pluto = pluto(Some(sun_mu));
    let pluto_mu = pluto.mass * universe.g;
    let pluto_idx = universe.add_body(pluto, Some(sun_idx)).unwrap();

    let makemake = makemake(Some(sun_mu));
    universe.add_body(makemake, Some(sun_idx)).unwrap();

    let eris = eris(Some(sun_mu));
    let eris_mu = eris.mass * universe.g;
    let eris_idx = universe.add_body(eris, Some(sun_idx)).unwrap();

    let sedna = sedna(Some(sun_mu));
    universe.add_body(sedna, Some(sun_idx)).unwrap();

    let haumea = haumea(Some(sun_mu));
    universe.add_body(haumea, Some(sun_idx)).unwrap();

    let quaoar = quaoar(Some(sun_mu));
    let quaoar_mu = quaoar.mass * universe.g;
    let quaoar_idx = universe.add_body(quaoar, Some(sun_idx)).unwrap();

    let weywot = weywot(Some(quaoar_mu));
    universe.add_body(weywot, Some(quaoar_idx)).unwrap();

    let charon = charon(Some(pluto_mu));
    universe.add_body(charon, Some(pluto_idx)).unwrap();

    let dysnomia = dysnomia(Some(eris_mu));
    universe.add_body(dysnomia, Some(eris_idx)).unwrap();

    return universe;
}

fn describe_universe(universe: &Universe) {
    println!(
        "Simulation universe with {} bodies",
        universe.get_bodies().len()
    );
    for (&i, wrapper) in universe.get_bodies().iter() {
        let body = &wrapper.body;
        println!("    {}: {:?}", i, body.name);
        println!("      Mass: {}", body.mass);
        println!("      Radius: {}", body.radius);
        if let Some(orbit) = &body.orbit {
            let location = universe.get_body_position(i);
            println!("      Orbit: {:?}", location);
            println!("        Semi-major axis: {}", orbit.get_semi_major_axis());
            println!("        Eccentricity: {}", orbit.get_eccentricity());
            println!("        Inclination: {}", orbit.get_inclination());
            println!("        Argument of periapsis: {}", orbit.get_arg_pe());
            println!(
                "        Longitude of ascending node: {}",
                orbit.get_long_asc_node()
            );
            println!(
                "        Mean anomaly at epoch: {}",
                orbit.get_mean_anomaly_at_epoch()
            );
        }
    }
}

fn print_all_body_positions(lock: &mut StdoutLock, universe: &Universe) {
    for (&i, w) in universe.get_bodies().iter() {
        let location = universe.get_body_position(i);
        writeln!(lock, "{}: {:?}", w.body.name, location).unwrap();
    }
}

use std::{
    fmt,
    io::{self, StdoutLock},
};

use glam::DVec3;
use std::{collections::HashMap, error::Error};

type Id = u64;

const GRAVITATIONAL_CONSTANT: f64 = 6.6743e-11;

/// Struct that represents the simulation of the universe.
#[derive(Clone, Debug, PartialEq)]
struct Universe {
    /// The celestial bodies in the universe and their relations.
    bodies: HashMap<Id, BodyWrapper>,

    /// The next ID to assign to a body.
    next_id: Id,

    /// The time elapsed in the universe, in seconds.
    time: f64,

    /// The time step of the simulation, in seconds.
    time_step: f64,

    /// The gravitational constant, in m^3 kg^-1 s^-2.
    g: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]

struct BodyRelation {
    parent: Option<Id>,
    satellites: Vec<Id>,
}

#[derive(Clone, Debug, PartialEq)]

struct BodyWrapper {
    body: Body,
    relations: BodyRelation,
}

#[derive(Clone, Debug)]
struct BodyAddError {
    cause: BodyAddErrorCause,
    body: Box<Body>,
}

#[non_exhaustive]
#[derive(Clone, Debug)]
enum BodyAddErrorCause {
    ParentNotFound { parent_id: Id },
}

impl fmt::Display for BodyAddErrorCause {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BodyAddErrorCause::ParentNotFound { parent_id } => write!(
                f,
                "There was no body at the specified parent index of {parent_id}"
            ),
        }
    }
}

impl Error for BodyAddErrorCause {}

impl fmt::Display for BodyAddError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Failed to add body {:?} to the universe: {}",
            *self.body, self.cause
        )
    }
}

impl Error for BodyAddError {}

impl Universe {
    /// Adds a body to the universe.
    ///
    /// `body`: The body to add into the universe.  
    /// `parent_id`: The index of the body that this body is orbiting.  
    /// Returns: The index of the newly-added body.  
    fn add_body(&mut self, mut body: Body, parent_id: Option<Id>) -> Result<Id, BodyAddError> {
        if let Some(parent_id) = parent_id {
            let parent = match self.bodies.get(&parent_id) {
                Some(b) => b,
                None => {
                    return Err(BodyAddError {
                        cause: BodyAddErrorCause::ParentNotFound { parent_id },
                        body: Box::new(body),
                    })
                }
            };

            if let Some(ref mut o) = body.orbit {
                o.set_gravitational_parameter(
                    self.g * parent.body.mass,
                    MuSetterMode::KeepPositionAtTime(self.time),
                );
            }
        }

        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);

        self.bodies.insert(
            id,
            BodyWrapper {
                body,
                relations: BodyRelation {
                    parent: parent_id,
                    satellites: Vec::new(),
                },
            },
        );
        if let Some(parent_index) = parent_id {
            if let Some(wrapper) = self.bodies.get_mut(&parent_index) {
                wrapper.relations.satellites.push(id);
            }
        }

        Ok(id)
    }

    /// Gets a reference to a HashMap of all bodies in the universe and their relations.
    fn get_bodies(&self) -> &HashMap<Id, BodyWrapper> {
        &self.bodies
    }

    /// Advances the simulation by a tick.
    fn tick(&mut self) {
        self.time += self.time_step;
    }

    /// Gets the absolute position of a body in the universe.
    ///
    /// Each coordinate is in meters.
    ///
    /// `index`: The index of the body to get the position of.
    ///
    /// Returns: The absolute position of the body.  
    /// The top ancestor of the body (i.e, the body with no parent) is at the origin (0, 0, 0).  
    fn get_body_position(&self, index: Id) -> Option<DVec3> {
        let wrapper = self.bodies.get(&index)?;
        let (orbit, parent) = (&wrapper.body.orbit, wrapper.relations.parent);

        let mut position = match orbit {
            Some(orbit) => orbit.get_position_at_time(self.time),
            None => DVec3::ZERO, // If the body is not in orbit, its position is the origin
        };

        if let Some(parent) = parent {
            if let Some(parent_position) = self.get_body_position(parent) {
                position += parent_position;
            }
        }

        Some(position)
    }
}

impl Default for Universe {
    /// Creates an empty universe with default parameters.
    fn default() -> Universe {
        Universe {
            bodies: HashMap::new(),
            time: 0.0,
            time_step: 3.6e3,
            g: GRAVITATIONAL_CONSTANT,
            next_id: 0,
        }
    }
}

impl fmt::Display for Universe {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "Universe with {} bodies, t={}",
            self.bodies.len(),
            self.time
        )
    }
}

/// A struct representing a celestial body.
#[derive(Clone, Debug, PartialEq)]
struct Body {
    /// The name of the celestial body.
    name: String,

    /// The mass of the celestial body, in kilograms.
    mass: f64,

    /// The radius of the celestial body, in meters.
    radius: f64,

    /// The orbit of the celestial body, if it is orbiting one.
    orbit: Option<Orbit>,
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
    fn new(name: String, mass: f64, radius: f64, orbit: Option<Orbit>) -> Self {
        Self {
            name,
            mass,
            radius,
            orbit,
        }
    }
}

impl Default for Body {
    /// Creates a default `Body` instance.
    ///
    /// Currently, this function returns the Earth.  
    /// However, do not rely on this behavior, as it may change in the future.
    fn default() -> Self {
        Self {
            name: "Earth".to_string(),
            mass: 5.972e24,
            radius: 6.371e6,
            orbit: None,
        }
    }
}

/// Returns the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn the_sun(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            2.36518e20, 2.36518e20,
            // I can't seem to find the orientation of the Sun's orbit
            0.0, 0.0, 0.0, 0.0, mu,
        )
    });

    Body::new("The Sun".to_string(), 1.989e30, 6.9634e5, orbit)
}

/// Returns Mercury, the closest planet to the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn mercury(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            6.982e7,
            4.6e7,
            7.005_f64.to_radians(),
            29.124_f64.to_radians(),
            48.331_f64.to_radians(),
            174.796_f64.to_radians(),
            mu,
        )
    });

    Body::new("Mercury".to_string(), 3.3011e23, 2.4397e6, orbit)
}

/// Returns Venus, the second planet from the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn venus(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            1.0894e8,
            1.0748e8,
            3.39458_f64.to_radians(),
            54.884_f64.to_radians(),
            76.680_f64.to_radians(),
            50.115_f64.to_radians(),
            mu,
        )
    });

    Body::new("Venus".to_string(), 4.8675e24, 6.0518e6, orbit)
}

/// Returns Earth, the third planet from the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn earth(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            1.52097597e11,
            1.47098450e11,
            0.00005_f64.to_radians(),
            114.20783_f64.to_radians(),
            -11.26064_f64.to_radians(),
            358.617_f64.to_radians(),
            mu,
        )
    });

    Body::new("Earth".to_string(), 5.972e24, 6.371e6, orbit)
}

/// Returns Mars, the fourth planet from the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn mars(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            2.49261e11,
            2.0665e11,
            1.850_f64.to_radians(),
            286.5_f64.to_radians(),
            49.57854_f64.to_radians(),
            19.412_f64.to_radians(),
            mu,
        )
    });

    Body::new("Mars".to_string(), 6.4171e23, 3.3895e6, orbit)
}

/// Returns Jupiter, the fifth planet from the Sun.  
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn jupiter(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            8.16363e11,
            7.40595e11,
            1.303_f64.to_radians(),
            273.867_f64.to_radians(),
            100.464_f64.to_radians(),
            20.02_f64.to_radians(),
            mu,
        )
    });

    Body::new("Jupiter".to_string(), 1.8982e27, 6.9911e7, orbit)
}

/// Returns Saturn, the sixth planet from the Sun.  
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn saturn(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            1.5145e12,
            1.35255e12,
            2.485_f64.to_radians(),
            339.392_f64.to_radians(),
            113.665_f64.to_radians(),
            317.020_f64.to_radians(),
            mu,
        )
    });

    Body::new("Saturn".to_string(), 5.6834e26, 58232.0, orbit)
}

/// Returns Uranus, the seventh planet from the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn uranus(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            3.00639e12,
            2.73556e12,
            0.773_f64.to_radians(),
            96.998857_f64.to_radians(),
            74.006_f64.to_radians(),
            142.2386_f64.to_radians(),
            mu,
        )
    });

    Body::new("Uranus".to_string(), 8.681e25, 2.5362e7, orbit)
}

/// Returns Neptune, the eighth planet from the Sun.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn neptune(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            4.54e12,
            4.46e12,
            1.77_f64.to_radians(),
            273.187_f64.to_radians(),
            131.783_f64.to_radians(),
            259.883_f64.to_radians(),
            mu,
        )
    });

    Body::new("Neptune".to_string(), 1.02409e26, 2.4341e7, orbit)
}
/// Returns the Moon, the only natural satellite of Earth.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn luna(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            405400.0,
            362600.0,
            5.145_f64.to_radians(),
            0.0,
            0.0,
            0.0,
            mu,
        )
    });

    Body::new("Luna".to_string(), 7.342e22, 1.7371e6, orbit)
}

use luna as the_moon;

/// Returns (50000) Quaoar I, a.k.a. Weywot, the moon of the dwarf planet Quaoar.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn weywot(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::new(
            0.056,
            1.3289e7,
            // I only found inclination to the ecliptic.
            // I couldn't find one in relation to Quaoar's equator.
            0.0,
            335_f64.to_radians(),
            1_f64.to_radians(),
            // I couldn't find the mean anomaly
            0.0,
            mu,
        )
    });

    Body::new(
        "Weywot".to_string(),
        // Weywot's mass has not been measured.
        // I extrapolated from the mean density of Quaoar:
        // ~1.7 g/cm^3
        // Weywot's radius is about 1e5 meters
        // Therefore its volume is:
        // V = 4/3 * pi * (1e5)^3 in cubic meters
        // V = 4.18879e15 m^3
        // Its mass is then:
        // M = 1.7 g/cm^3 * 4.18879e15 m^3
        // M = 1.7e3 kg/m^3 * 4.18879e15 m^3
        // M = 7.12e18 kg
        // (approximately)
        7.12e18,
        1e5,
        orbit,
    )
}

/// Returns (134340) Pluto I, a.k.a., Charon, the largest moon orbiting Pluto.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn charon(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            1.959892e7,
            1.959261e7,
            0.08_f64.to_radians(),
            // Could not find number for arg. of pe.
            0.0,
            223.046_f64.to_radians(),
            // Could not find number for mean anomaly
            0.0,
            mu,
        )
    });

    Body::new("Charon".to_string(), 1.5897e21, 6.06e5, orbit)
}

/// Returns (136199) Eris I Dysnomia, the moon of the dwarf planet Eris.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn dysnomia(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::new(
            0.0062,
            3.7273e7,
            0.0,
            180.83_f64.to_radians(),
            126.17_f64.to_radians(),
            // Could not find mean anomaly number
            0.0,
            mu,
        )
    });

    Body::new(
        "Dysnomia".to_string(),
        // Apparently not very precise;
        // it's plus or minus 5.7e19 kg
        8.2e19,
        6.15e5 / 2.0,
        orbit,
    )
}
/// Returns 1 Ceres, a dwarf planet in the asteroid belt.  
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn ceres(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            4.46e11,
            3.81e11,
            10.6_f64.to_radians(),
            73.6_f64.to_radians(),
            80.3_f64.to_radians(),
            291.4_f64.to_radians(),
            mu,
        )
    });

    Body::new("Ceres".to_string(), 9.3839e20, 939.4 / 2.0, orbit)
}

/// Returns 50000 Quaoar, a dwarf planet in the Kuiper belt.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn quaoar(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            6.805e12,
            6.268e12,
            7.9895_f64.to_radians(),
            147.48_f64.to_radians(),
            188.927_f64.to_radians(),
            301.104_f64.to_radians(),
            mu,
        )
    });

    Body::new("Quaoar".to_string(), 1.2e21, 5.45e5, orbit)
}

/// Returns 90377 Sedna, a dwarf planet, sednoid, and extreme trans-Neptunian object.
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn sedna(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            1.4e14,
            1.14e13,
            11.9307_f64.to_radians(),
            311.352_f64.to_radians(),
            144.248_f64.to_radians(),
            358.117_f64.to_radians(),
            mu,
        )
    });

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
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn pluto(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            7.37593e12,
            4.43682e12,
            17.16_f64.to_radians(),
            113.834_f64.to_radians(),
            110.299_f64.to_radians(),
            14.53_f64.to_radians(),
            mu,
        )
    });

    Body::new("Pluto".to_string(), 1.3025e22, 1.1883e6, orbit)
}

/// Returns 136108 Haumea, a dwarf planet in the Kuiper belt.  
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn haumea(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            7.717e12,
            5.1831e12,
            28.2137_f64.to_radians(),
            239.041_f64.to_radians(),
            122.167_f64.to_radians(),
            218.205_f64.to_radians(),
            mu,
        )
    });

    Body::new("Haumea".to_string(), 4e21, 7.8e5, orbit)
}

/// Returns 136199 Eris, a dwarf planet, and a trans-Neptunian and scattered disc object.  
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn eris(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            1.4579e13,
            5.725e12,
            44.04_f64.to_radians(),
            151.639_f64.to_radians(),
            35.951_f64.to_radians(),
            205.989_f64.to_radians(),
            mu,
        )
    });

    Body::new("Eris".to_string(), 1.6466e22, 1.163e6, orbit)
}

/// Returns 136472 Makemake, a dwarf planet in the Kuiper belt.  
///
/// `parent_mu`: The gravitational parameter of the parent body, if any.
/// If None, the celestial body will not be placed in an orbit.
fn makemake(parent_mu: Option<f64>) -> Body {
    let orbit = parent_mu.map(|mu| {
        Orbit::with_apoapsis(
            7.8922e12,
            5.7003e12,
            28.9835_f64.to_radians(),
            294.834_f64.to_radians(),
            79.62_f64.to_radians(),
            165.514_f64.to_radians(),
            mu,
        )
    });

    Body::new("Makemake".to_string(), 3.1e21, 7.15e5, orbit)
}
