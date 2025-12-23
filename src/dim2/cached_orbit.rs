use glam::DMat2;

// TODO
pub struct Orbit2D {
    eccentricity: f64,
    periapsis: f64,
    arg_pe: f64,
    mean_anomaly: f64,
    cache: OrbitCachedCalculations,
}

struct OrbitCachedCalculations {
    matrix: DMat2,
}
