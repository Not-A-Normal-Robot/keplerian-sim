use glam::DVec2;

pub mod cached_orbit;
pub mod compact_orbit;

/// A trait that defines the methods that a 2D-constrained Keplerian
/// orbit must implement.
///
/// TODO: Examples
pub trait OrbitTrait2D {
    // TODO
}

/// A struct representing a position and velocity at a point in the orbit.
///
/// The position and velocity vectors are two-dimensional.
///
/// The position vector is in meters, while the velocity vector is in
/// meters per second.
///
/// State vectors can be used to form an orbit using
/// [`to_orbit`][Self::to_orbit].
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct StateVectors2D {
    /// The 2D position at a point in the orbit, in meters.
    pub position: DVec2,
    /// The 2D velocity at a point in the orbit, in meters per second.
    pub velocity: DVec2,
}

/// A mode to describe how the gravitational parameter setter should behave.
///
/// This is used to describe how the setter should behave when setting the
/// gravitational parameter of the parent body.
///
/// # Which mode should I use?
/// The mode you should use depends on what you expect from setting the mu value
/// to a different value.
///
/// If you just want to set the mu value naïvely (without touching the
/// other orbital elements), you can use the `KeepElements` variant.
///
/// If this is part of a simulation and you want to keep the current position
/// (not caring about the velocity), you can use the `KeepPositionAtTime` variant.
///
/// If you want to keep the current position and velocity, you can use either
/// the `KeepKnownStateVectors` or `KeepStateVectorsAtTime` modes, the former
/// being more performant if you already know the state vectors beforehand.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MuSetterMode2D {
    /// Keep all the other orbital parameters the same.
    ///
    /// **This will change the position and velocity of the orbiting body abruptly,
    /// if you use the time-based functions.** It will not, however, change the trajectory
    /// of the orbit.
    ///
    /// # Performance
    /// This mode is the fastest of the mu setter modes as it is simply an
    /// assignment operation.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, MuSetterMode2D};
    ///
    /// let mut orbit = Orbit2D::new(
    ///     0.0, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Inclination
    ///     0.0, // Argument of Periapsis
    ///     0.0, // Longitude of Ascending Node
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter (mu = GM)
    /// );
    ///
    /// orbit.set_gravitational_parameter(3.0, MuSetterMode2D::KeepElements);
    ///
    /// assert_eq!(orbit.get_eccentricity(), 0.0);
    /// assert_eq!(orbit.get_periapsis(), 1.0);
    /// assert_eq!(orbit.get_arg_pe(), 0.0);
    /// assert_eq!(orbit.get_mean_anomaly_at_epoch(), 0.0);
    /// assert_eq!(orbit.get_gravitational_parameter(), 3.0);
    /// ```
    KeepElements,
    /// Keep the overall shape of the orbit, but modify the mean anomaly at epoch
    /// such that the position at the given time is the same.
    ///
    /// **This will change the velocity of the orbiting body abruptly, if you use
    /// the time-based position/velocity getter functions.**
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Performance
    /// This mode is slower than the `KeepElements` mode as it has to compute a new
    /// mean anomaly at epoch. However, this isn't too expensive and only costs a
    /// few squareroot operations.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, MuSetterMode2D};
    ///
    /// let mut orbit = Orbit2D::new(
    ///     0.0, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Inclination
    ///     0.0, // Argument of Periapsis
    ///     0.0, // Longitude of Ascending Node
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter (mu = GM)
    /// );
    ///
    /// orbit.set_gravitational_parameter(
    ///     3.0,
    ///     MuSetterMode2D::KeepPositionAtTime(0.4),
    /// );
    ///
    /// assert_eq!(orbit.get_eccentricity(), 0.0);
    /// assert_eq!(orbit.get_periapsis(), 1.0);
    /// assert_eq!(orbit.get_inclination(), 0.0);
    /// assert_eq!(orbit.get_arg_pe(), 0.0);
    /// assert_eq!(orbit.get_long_asc_node(), 0.0);
    /// assert_eq!(orbit.get_mean_anomaly_at_epoch(), -0.2928203230275509);
    /// assert_eq!(orbit.get_gravitational_parameter(), 3.0);
    /// ```
    KeepPositionAtTime(f64),
    /// Keep the position and velocity of the orbit at a certain time
    /// roughly unchanged, using known [StateVectors] to avoid
    /// duplicate calculations.
    ///
    /// **This will change the orbit's overall trajectory.**
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Unchecked Operation
    /// This mode does not check whether or not the state vectors and time values given
    /// match up. Mismatched values may result in undesired behavior and NaNs.  
    /// Use the [`KeepStateVectorsAtTime`][MuSetterMode2D::KeepStateVectorsAtTime]
    /// mode if you don't want this unchecked operation.
    ///
    /// # Performance
    /// This mode uses some trigonometry, and therefore is not very performant.  
    /// Consider using another mode if performance is an issue.  
    ///
    /// This is, however, significantly more performant than the numerical approach
    /// used in the [`KeepStateVectorsAtTime`][MuSetterMode2D::KeepStateVectorsAtTime]
    /// mode.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, MuSetterMode2D};
    ///
    /// let mut orbit = Orbit2D::new(
    ///     0.0, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Inclination
    ///     0.0, // Argument of Periapsis
    ///     0.0, // Longitude of Ascending Node
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter (mu = GM)
    /// );
    ///
    /// let time = 0.75;
    ///
    /// let state_vectors = orbit.get_state_vectors_at_time(time);
    ///
    /// orbit.set_gravitational_parameter(
    ///     3.0,
    ///     MuSetterMode2D::KeepKnownStateVectors {
    ///         state_vectors,
    ///         time
    ///     }
    /// );
    ///
    /// let new_state_vectors = orbit.get_state_vectors_at_time(time);
    ///
    /// println!("Old state vectors: {state_vectors:?}");
    /// println!("New state vectors: {new_state_vectors:?}");
    /// ```
    KeepKnownStateVectors {
        /// The state vectors describing the point in the orbit where you want the
        /// position and velocity to remain the same.
        ///
        /// Must correspond to the time value, or undesired behavior and NaNs may occur.
        state_vectors: StateVectors2D,

        /// The time value of the point in the orbit
        /// where you want the position and velocity to remain the same.
        ///
        /// The time is measured in seconds.
        ///
        /// Must correspond to the given state vectors, or undesired behavior and NaNs may occur.
        time: f64,
    },
    /// Keep the position and velocity of the orbit at a certain time
    /// roughly unchanged.
    ///
    /// **This will change the orbit's overall trajectory.**
    ///
    /// # Time
    /// The time is measured in seconds.
    ///
    /// # Performance
    /// This mode uses numerical approach methods, and therefore is not performant.  
    /// Consider using another mode if performance is an issue.  
    ///
    /// Alternatively, if you already know the state vectors (position and velocity)
    /// of the point you want to keep, use the
    /// [`KeepKnownStateVectors`][MuSetterMode2D::KeepKnownStateVectors]
    /// mode instead. This skips the numerical method used to obtain the eccentric anomaly
    /// and some more trigonometry.
    ///
    /// If you only know the eccentric anomaly and true anomaly, it's more performant
    /// to derive state vectors from those first and then use the aforementioned
    /// [`KeepKnownStateVectors`][MuSetterMode2D::KeepKnownStateVectors] mode. This can
    /// be done using the [`Orbit2D::get_state_vectors_at_eccentric_anomaly`] function, for
    /// example.
    ///
    /// # Example
    /// ```
    /// use keplerian_sim::{Orbit2D, OrbitTrait2D, MuSetterMode2D};
    ///
    /// let old_orbit = Orbit2D::new(
    ///     0.0, // Eccentricity
    ///     1.0, // Periapsis
    ///     0.0, // Inclination
    ///     0.0, // Argument of Periapsis
    ///     0.0, // Longitude of Ascending Node
    ///     0.0, // Mean anomaly at epoch
    ///     1.0, // Gravitational parameter (mu = GM)
    /// );
    ///
    /// let mut new_orbit = old_orbit.clone();
    ///
    /// const TIME: f64 = 1.5;
    ///
    /// new_orbit.set_gravitational_parameter(
    ///     3.0,
    ///     MuSetterMode2D::KeepStateVectorsAtTime(TIME)
    /// );
    ///
    /// let old_state_vectors = old_orbit.get_state_vectors_at_time(TIME);
    /// let new_state_vectors = new_orbit.get_state_vectors_at_time(TIME);
    ///
    /// println!("Old state vectors: {old_state_vectors:?}");
    /// println!("New state vectors: {new_state_vectors:?}");
    /// ```
    KeepStateVectorsAtTime(f64),
}
