use core::fmt;

use super::Body;

/// Struct that represents the simulation of the universe.
#[derive(Clone, Debug, PartialEq)]
pub struct Universe {
    /// The celestial bodies in the universe.
    bodies: Vec<Body>,

    /// The relations between the bodies.
    body_relations: Vec<BodyRelation>,

    /// The time elapsed in the universe, in seconds.
    pub time: f64,

    /// The time step of the simulation, in seconds.
    pub time_step: f64,

    /// The gravitational constant, in m^3 kg^-1 s^-2.
    pub g: f64,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyRelation {
    pub parent: Option<usize>,
    pub satellites: Vec<usize>,
}

impl Universe {
    /// Creates an empty universe.
    pub fn new(time_step: Option<f64>, g: Option<f64>) -> Universe {
        let time_step = time_step.unwrap_or(3.6e3);
        let g = g.unwrap_or(6.67430e-11);

        return Universe {
            bodies: Vec::new(),
            body_relations: Vec::new(),
            time: 0.0,
            time_step,
            g,
        };
    }

    /// Creates an empty universe with default parameters.
    pub fn new_default() -> Universe {
        return Universe {
            bodies: Vec::new(),
            body_relations: Vec::new(),
            time: 0.0,
            time_step: 3.6e3,
            g: 6.67430e-11,
        };
    }

    /// Adds a body to the universe.
    /// `body`: The body to add into the universe.
    /// `satellite_of`: The index of the body that this body is orbiting.
    /// Returns: The index of the newly-added body.
    pub fn add_body(&mut self, body: Body, satellite_of: Option<usize>) -> usize {
        self.bodies.push(body);
        if let Some(parent) = satellite_of {
            self.body_relations.push(BodyRelation {
                parent: Some(parent),
                satellites: Vec::new(),
            });
            self.body_relations[parent]
                .satellites
                .push(self.bodies.len() - 1);
        } else {
            self.body_relations.push(BodyRelation {
                parent: None,
                satellites: Vec::new(),
            });
        }
        return self.bodies.len() - 1;
    }

    /// Removes a body from the universe.
    /// `body_index`: The index of the body to remove.
    pub fn remove_body(&mut self, body_index: usize) -> Body {
        let body = self.bodies.remove(body_index);
        let relations = &mut self.body_relations[body_index];

        if let Some(_) = relations.parent {
            relations.satellites.retain(|&x| x != body_index);
        }

        let satellites_to_remove = relations.satellites.clone();

        self.body_relations.remove(body_index);

        for satellite in satellites_to_remove {
            self.remove_body(satellite);
        }

        return body;
    }

    /// Gets a Vec of all bodies in the universe.
    pub fn get_bodies(&self) -> &Vec<Body> {
        return &self.bodies;
    }

    /// Gets a Vec of all body relations in the universe.
    pub fn get_body_relations(&self) -> &Vec<BodyRelation> {
        return &self.body_relations;
    }

    /// Gets a mutable reference to a body in the universe.
    pub fn get_body_mut(&mut self, index: usize) -> &mut Body {
        return &mut self.bodies[index];
    }

    /// Gets an immutable reference to a body in the unvierse.
    pub fn get_body(&self, index: usize) -> &Body {
        return &self.bodies[index];
    }

    /// Gets the index of the first body with a given name.
    pub fn get_first_body_index_with_name(&self, name: String) -> Option<usize> {
        for (i, body) in self.bodies.iter().enumerate() {
            if body.name == name {
                return Some(i);
            }
        }
        return None;
    }

    /// Gets the index of the last body with a given name.
    pub fn get_last_body_index_with_name(&self, name: String) -> Option<usize> {
        for (i, body) in self.bodies.iter().enumerate().rev() {
            if body.name == name {
                return Some(i);
            }
        }
        return None;
    }

    /// Advances the simulation by a tick.
    pub fn tick(&mut self) {
        for body in &mut self.bodies {
            if body.orbit.is_none() {
                continue;
            }
            body.progress_orbit(self.time_step, self.g).unwrap();
        }
    }

    /// Advances the universe by multiple ticks.
    pub fn warp(&mut self, ticks: u128) {
        for _ in 0..ticks {
            self.tick();
        }
    }

    /// Gets the absolute position of a body in the universe.
    ///
    /// Each coordinate is in meters.
    pub fn get_body_position(&self, index: usize) -> (f64, f64, f64) {
        let body = &self.bodies[index];

        let mut position = body.get_relative_position();

        if let Some(parent) = self.body_relations[index].parent {
            let parent_position = self.get_body_position(parent);
            position.0 += parent_position.0;
            position.1 += parent_position.1;
            position.2 += parent_position.2;
        }

        return position;
    }
}

impl Default for Universe {
    fn default() -> Self {
        return Universe::new_default();
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
