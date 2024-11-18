use std::f64::consts::{PI, TAU};

use crate::{
    keplers_equation, keplers_equation_derivative, keplers_equation_hyperbolic, keplers_equation_hyperbolic_derivative, keplers_equation_hyperbolic_second_derivative, keplers_equation_second_derivative, ApoapsisSetterError, Matrix3x2, Orbit, OrbitTrait
};

/// A minimal struct representing a Keplerian orbit.
/// 
/// This struct minimizes memory footprint by not caching variables.  
/// Because of this, calculations can be slower than caching those variables.  
/// For this reason, you might consider using the `Orbit` struct instead.
#[derive(Clone, Debug)]
pub struct CompactOrbit {
    /// The eccentricity of the orbit.  
    /// e < 1: ellipse  
    /// e = 1: parabola  
    /// e > 1: hyperbola  
    pub eccentricity: f64,

    /// The periapsis of the orbit, in meters.
    pub periapsis: f64,

    /// The inclination of the orbit, in radians.
    pub inclination: f64,

    /// The argument of periapsis of the orbit, in radians.
    pub arg_pe: f64,

    /// The longitude of ascending node of the orbit, in radians.
    pub long_asc_node: f64,

    /// The mean anomaly at orbit epoch, in radians.
    pub mean_anomaly: f64,
}

// Initialization and cache management
impl CompactOrbit {
    /// Creates a new `CompactOrbit` instance with the given parameters.
    /// 
    /// Note: This function uses eccentricity instead of apoapsis.  
    /// If you want to provide an apoapsis instead, consider using the
    /// [`CompactOrbit::with_apoapsis`] function instead.
    /// 
    /// ### Parameters
    /// - `eccentricity`: The eccentricity of the orbit.
    /// - `periapsis`: The periapsis of the orbit, in meters.
    /// - `inclination`: The inclination of the orbit, in radians.
    /// - `arg_pe`: The argument of periapsis of the orbit, in radians.
    /// - `long_asc_node`: The longitude of ascending node of the orbit, in radians.
    /// - `mean_anomaly`: The mean anomaly of the orbit, in radians.
    pub fn new(
        eccentricity: f64, periapsis: f64,
        inclination: f64, arg_pe: f64, long_asc_node: f64,
        mean_anomaly: f64
    ) -> CompactOrbit {
        return CompactOrbit {
            eccentricity, periapsis,
            inclination, arg_pe, long_asc_node,
            mean_anomaly,
        };
    }

    /// Creates a new `CompactOrbit` instance with the given parameters.
    /// 
    /// Note: This function uses apoapsis instead of eccentricity.  
    /// Because of this, it's not recommended to initialize
    /// parabolic or hyperbolic 'orbits' with this function.  
    /// If you're looking to initialize a parabolic or hyperbolic
    /// trajectory, consider using the [`CompactOrbit::new`] function instead.
    /// 
    /// ### Parameters
    /// - `apoapsis`: The apoapsis of the orbit, in meters.
    /// - `periapsis`: The periapsis of the orbit, in meters.
    /// - `inclination`: The inclination of the orbit, in radians.
    /// - `arg_pe`: The argument of periapsis of the orbit, in radians.
    /// - `long_asc_node`: The longitude of ascending node of the orbit, in radians.
    /// - `mean_anomaly`: The mean anomaly of the orbit, in radians.
    pub fn with_apoapsis(
        apoapsis: f64, periapsis: f64,
        inclination: f64, arg_pe: f64, long_asc_node: f64,
        mean_anomaly: f64
    ) -> CompactOrbit {
        let eccentricity = (apoapsis - periapsis ) / (apoapsis + periapsis);
        return CompactOrbit::new(eccentricity, periapsis, inclination, arg_pe, long_asc_node, mean_anomaly);
    }

    /// Creates a unit orbit.
    /// 
    /// The unit orbit is a perfect circle of radius 1 and no "tilt".
    pub fn new_default() -> CompactOrbit {
        return Self::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0);
    }
}


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
/// https://doi.org/10.1051/0004-6361/202141423
const B: f64 = 0.999999;

/// A constant used for the Laguerre method.
/// 
/// The paper "An improved algorithm due to
/// laguerre for the solution of Kepler's equation."
/// says:
/// 
/// > Similar experimentation has been done with values of n both greater and smaller
/// than n = 5. The speed of convergence seems to be very insensitive to the choice of n.
/// No value of n was found to yield consistently better convergence properties than the
/// choice of n = 5 though specific cases were found where other choices would give
/// faster convergence.
const N_U32: u32 = 5;

/// A constant used for the Laguerre method.
/// 
/// The paper "An improved algorithm due to
/// laguerre for the solution of Kepler's equation."
/// says:
/// 
/// > Similar experimentation has been done with values of n both greater and smaller
/// than n = 5. The speed of convergence seems to be very insensitive to the choice of n.
/// No value of n was found to yield consistently better convergence properties than the
/// choice of n = 5 though specific cases were found where other choices would give
/// faster convergence.
const N_F64: f64 = 5.0;

/// The maximum number of iterations for numerical approach methods.
/// 
/// This is used to prevent infinite loops in case the method fails to converge.
const NUMERIC_MAX_ITERS: u32 = 1000;

const PI_SQUARED: f64 = PI * PI;

impl CompactOrbit {
    // "An improved algorithm due to laguerre for the solution of Kepler's equation."
    // by Bruce A. Conway
    // https://doi.org/10.1007/bf01230852
    fn get_eccentric_anomaly_elliptic(&self, mut mean_anomaly: f64) -> f64 {
        let mut sign = 1.0;
        // Use the symmetry and periodicity of the eccentric anomaly
        // Equation 2 from the paper
        // "Two fast and accurate routines for solving
        // the elliptic Kepler equation for all values
        // of the eccentricity and mean anomaly"
        mean_anomaly %= TAU;
        if mean_anomaly > PI {
            // return self.get_eccentric_anomaly_elliptic(mean_anomaly - TAU);
            mean_anomaly -= TAU;
        }
        if mean_anomaly < 0.0 {
            // return -self.get_eccentric_anomaly_elliptic(-mean_anomaly);
            mean_anomaly = -mean_anomaly;
            sign = -1.0;
        }

        // Starting guess
        // Section 2.1.2, 'The "rational seed"',
        // equation 19, from the paper
        // "Two fast and accurate routines for solving
        // the elliptic Kepler equation for all values
        // of the eccentricity and mean anomaly"
        //
        // E_0 = M + (4beM(pi - M)) / (8eM + 4e(e-pi) + pi^2)
        // where:
        // e = eccentricity
        // M = mean anomaly
        // pi = the constant PI
        // b = the constant B
        let mut eccentric_anomaly =
            mean_anomaly +
            (4.0 * self.eccentricity * B * mean_anomaly * (PI - mean_anomaly)) /
            (
                8.0 * self.eccentricity * mean_anomaly +
                4.0 * self.eccentricity * (self.eccentricity - PI) +
                PI_SQUARED
            );

        // Laguerre's method
        // 
        // i = 2, 3, ..., n
        //
        // D = sqrt((n-1)^2(f'(x_i))^2 - n(n-1)f(x_i)f''(x_i))
        //
        // x_i+1 = x_i - (nf(x_i) / (f'(x_i) +/- D))
        // ...where the "+/-" is chosen to so that abs(denominator) is maximized
        for _ in 2..N_U32 {
            let f = keplers_equation(mean_anomaly, eccentric_anomaly, self.eccentricity);
            let fp = keplers_equation_derivative(eccentric_anomaly, self.eccentricity);
            let fpp = keplers_equation_second_derivative(eccentric_anomaly, self.eccentricity);

            // TODO: apply this new algo to compact orbit
            let n = N_F64;
            let n_minus_1 = n - 1.0;
            let d = ((n_minus_1 * n_minus_1) * fp * fp - n * n_minus_1 * f * fpp).abs().sqrt().copysign(fp);

            let denominator = n * f / (fp + d.max(1e-30));
            eccentric_anomaly -= denominator;

            if denominator.abs() < 1e-30 || !denominator.is_finite() {
                // dangerously close to div-by-zero, break out
                break;
            }
        }

        return eccentric_anomaly * sign;
    }

    fn get_eccentric_anomaly_hyperbolic(&self, mean_anomaly: f64) -> f64 {
        let target_accuracy = 1e-9;
        let mut eccentric_anomaly =
            (2.0 * mean_anomaly.abs() / self.eccentricity).ln().max(0.01) *
            mean_anomaly.signum();

        for _ in 0..NUMERIC_MAX_ITERS {
            // HALLEY'S METHOD
            // https://en.wikipedia.org/wiki/Halley%27s_method
            // x_n+1 = x_n - f(x_n) / (f'(x_n) - f(x_n) * f''(x_n) / (2 * f'(x_n)))

            let f = keplers_equation_hyperbolic(mean_anomaly, eccentric_anomaly, self.eccentricity);
            let fp = keplers_equation_hyperbolic_derivative(eccentric_anomaly, self.eccentricity);
            let fpp = keplers_equation_hyperbolic_second_derivative(eccentric_anomaly, self.eccentricity);

            let denominator = fp - f * fpp / (2.0 * fp);

            if denominator.abs() < 1e-30 || !denominator.is_finite() {
                // dangerously close to div-by-zero, break out
                break;
            }

            let next_value = eccentric_anomaly - f / denominator;

            let diff = (eccentric_anomaly - next_value).abs();
            eccentric_anomaly = next_value;

            if diff < target_accuracy {
                break;
            }
        }

        return eccentric_anomaly;
    }
}

impl OrbitTrait for CompactOrbit {
    fn get_semi_major_axis(&self) -> f64 {
        return self.periapsis / (1.0 - self.eccentricity);
    }
    
    fn get_semi_minor_axis(&self) -> f64 {
        let semi_major_axis = self.get_semi_major_axis();
        let eccentricity_squared = self.eccentricity * self.eccentricity;
        return semi_major_axis * (1.0 - eccentricity_squared).abs().sqrt();
    }
    
    fn get_linear_eccentricity(&self) -> f64 {
        return self.get_semi_major_axis() - self.periapsis;
    }
    
    fn get_apoapsis(&self) -> f64 {
        if self.eccentricity == 1.0 {
            return f64::INFINITY;
        } else {
            return self.get_semi_major_axis() * (1.0 + self.eccentricity);
        }
    }

    fn set_apoapsis(&mut self, apoapsis: f64) -> Result<(), ApoapsisSetterError> {
        if apoapsis < 0.0 {
            return Err(ApoapsisSetterError::ApoapsisNegative);
        } else if apoapsis < self.periapsis {
            return Err(ApoapsisSetterError::ApoapsisLessThanPeriapsis);
        }

        self.eccentricity = (apoapsis - self.periapsis) / (apoapsis + self.periapsis);

        return Ok(());
    }

    fn set_apoapsis_force(&mut self, apoapsis: f64) {
        let mut apoapsis = apoapsis;
        if apoapsis < self.periapsis && apoapsis > 0.0 {
            (apoapsis, self.periapsis) = (self.periapsis, apoapsis);
        }

        self.eccentricity = (apoapsis - self.periapsis) / (apoapsis + self.periapsis);
    }

    fn get_transformation_matrix(&self) -> Matrix3x2<f64> {
        let mut matrix = Matrix3x2::<f64>::filled_with(0.0);
        {
            let (sin_inc, cos_inc) = self.inclination.sin_cos();
            let (sin_arg_pe, cos_arg_pe) = self.arg_pe.sin_cos();
            let (sin_lan, cos_lan) = self.long_asc_node.sin_cos();
    
            // https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf
            matrix.e11 = cos_arg_pe * cos_lan - sin_arg_pe * cos_inc * sin_lan;
            matrix.e12 = -(sin_arg_pe * cos_lan + cos_arg_pe * cos_inc * sin_lan);
            
            matrix.e21 = cos_arg_pe * sin_lan + sin_arg_pe * cos_inc * cos_lan;
            matrix.e22 = cos_arg_pe * cos_inc * cos_lan - sin_arg_pe * sin_lan;
    
            matrix.e31 = sin_arg_pe * sin_inc;
            matrix.e32 = cos_arg_pe * sin_inc;
        }
        return matrix;
    }

    fn get_eccentric_anomaly(&self, mean_anomaly: f64) -> f64 {
        if self.eccentricity < 1.0 {
            return self.get_eccentric_anomaly_elliptic(mean_anomaly);
        } else {
            return self.get_eccentric_anomaly_hyperbolic(mean_anomaly);
        }
    }

    fn get_true_anomaly_at_eccentric_anomaly(&self, eccentric_anomaly: f64) -> f64 {
        if self.eccentricity < 1.0 {
            // https://en.wikipedia.org/wiki/True_anomaly#From_the_eccentric_anomaly
            let eccentricity = self.eccentricity;
            let beta = eccentricity / (1.0 + (1.0 - eccentricity * eccentricity).sqrt());
    
            return eccentric_anomaly + 2.0 * 
                (beta * eccentric_anomaly.sin() / (1.0 - beta * eccentric_anomaly.cos()))
                .atan();
        } else {
            // idk copilot got this
            return 2.0 *
                ((self.eccentricity + 1.0) / (self.eccentricity - 1.0)).sqrt().atan() *
                eccentric_anomaly.tanh();
        }
    }

    fn get_semi_latus_rectum(&self) -> f64 {
        return self.periapsis * (1.0 + self.eccentricity);
    }

    fn get_altitude_at_angle(&self, true_anomaly: f64) -> f64 {
        return
            self.get_semi_latus_rectum() /
            (1.0 + self.eccentricity * true_anomaly.cos());
    }

    fn get_mean_anomaly_at_time(&self, t: f64) -> f64 {
        let time = t.rem_euclid(1.0);
        return time * TAU + self.mean_anomaly;
    }
}

impl From<Orbit> for CompactOrbit {
    fn from(cached: Orbit) -> Self {
        return Self {
            eccentricity: cached.get_eccentricity(),
            periapsis: cached.get_periapsis(),
            inclination: cached.get_inclination(),
            arg_pe: cached.get_arg_pe(),
            long_asc_node: cached.get_long_asc_node(),
            mean_anomaly: cached.get_mean_anomaly()
        };
    }
}

impl Orbit {
    /// Compactify the cached orbit into a compact orbit to increase memory efficiency
    /// while sacrificing calculation speed.
    pub fn compactify(self) -> CompactOrbit {
        CompactOrbit::from(self)
    }
}