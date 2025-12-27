#[cfg(feature = "libm")]
use crate::math::F64Math;
use crate::{
    generated_sinh_approximator::SINH_5, keplers_equation, keplers_equation_derivative,
    keplers_equation_second_derivative, sinhcosh, solve_monotone_cubic, B, NUMERIC_MAX_ITERS,
    N_F64, N_U32,
};
use core::f64::consts::{PI, TAU};

/// Get an initial guess for the hyperbolic eccentric anomaly of an orbit.
///
/// # Performance
/// This function uses plenty of floating-point operations, including
/// divisions, natural logarithms, squareroots, and cuberoots, and thus
/// it is not very performant.
///
/// # Unchecked Operation
/// This function does not check whether or not the orbit is hyperbolic. If
/// this function is called on a non-hyperbolic orbit (i.e., elliptic or parabolic),
/// invalid values may be returned.
///
/// # Approximate Guess
/// This function returns a "good" initial guess for the hyperbolic eccentric anomaly.  
/// There are no constraints on the accuracy of the guess, and users may not
/// rely on this value being very accurate, especially in some edge cases.
///
/// # Source
/// From the paper:  
/// "A new method for solving the hyperbolic Kepler equation"  
/// by Baisheng Wu et al.  
/// Quote:
/// "we divide the hyperbolic eccentric anomaly interval into two parts:
/// a finite interval and an infinite interval. For the finite interval,
/// we apply a piecewise Pade approximation to establish an initial
/// approximate solution of HKE. For the infinite interval, an analytical
/// initial approximate solution is constructed."
pub(crate) fn get_approx_hyperbolic_eccentric_anomaly(eccentricity: f64, mean_anomaly: f64) -> f64 {
    let sign = mean_anomaly.signum();
    let mean_anomaly = mean_anomaly.abs();

    // (Paragraph after Eq. 5 in the aforementioned paper)
    //   The [mean anomaly] interval [0, e_c sinh(5) - 5) can
    //   be separated into fifteen subintervals corresponding to
    //   those intervals of F in [0, 5), see Eq. (4).
    sign * if mean_anomaly < eccentricity * SINH_5 - 5.0 {
        // We use the Pade approximation of sinh of order
        // [3 / 2], in `crate::generated_sinh_approximator`.
        // We can then rearrange the equation to a cubic
        // equation in terms of (F - a) and solve it.
        //
        // To quote the paper:
        //   Replacing sinh(F) in [the hyperbolic Kepler
        //   equation] with its piecewise Pade approximation
        //   defined in Eq. (4) [`crate::generated_sinh_approximator`]
        //   yields:
        //     e_c P(F) - F = M_h                          (6)
        //
        //   Eq. (6) can be written as a cubic equation in u = F - a, as
        //     (e_c p_3 - q_2)u^3 +
        //     (e_c p_2 - (M_h + a)q_2 - q_1) u^2 +
        //     (e_c p_1 - (M_h + a)q_1 - 1)u +
        //     e_c s - M_h - a = 0                         (7)
        //
        //   Solving Eq. (7) and picking the real root F = F_0 in the
        //   corresponding subinterval results in an initial approximate
        //   solution to [the hyperbolic Kepler equation].
        //
        // For context:
        // - `e_c` is eccentricity
        // - `p_*`, `q_*`, `a`, and `s` is derived from the Pade approximation
        //   arguments, which can be retrieved using the
        //   `generated_sinh_approximator::get_sinh_approx_params` function
        // - `M_h` is the mean anomaly
        // - `F` is the eccentric anomaly

        use crate::generated_sinh_approximator::get_sinh_approx_params;
        let params = get_sinh_approx_params(mean_anomaly);

        // We first get the value of each coefficient in the cubic equation:
        // Au^3 + Bu^2 + Cu + D = 0
        let mean_anom_plus_a = mean_anomaly + params.a;
        let coeff_a = eccentricity * params.p_3 - params.q_2;
        let coeff_b = eccentricity * params.p_2 - mean_anom_plus_a * params.q_2 - params.q_1;
        let coeff_c = eccentricity * params.p_1 - mean_anom_plus_a * params.q_1 - 1.0;
        let coeff_d = eccentricity * params.s - mean_anomaly - params.a;

        // Then we solve it to get the value of u = F - a
        let u = solve_monotone_cubic(coeff_a, coeff_b, coeff_c, coeff_d);

        u + params.a
    } else {
        // Equation 13
        // A *very* rough guess, with an error that may exceed 1%.
        let rough_guess = (2.0 * mean_anomaly / eccentricity).ln();

        /*
        A fourth-order Schröder iteration of the second kind
        is performed to create a better guess.
        ...Apparently it's not a well-known thing, but the aforementioned paper
        referenced this other paper about Schröder iterations:
        https://doi.org/10.1016/j.cam.2019.02.035

        To do the Schröder iteration, we need to compute a delta value
        to be added to the rough guess. Part of Equation 15 from the paper is below.

        delta = (
                6 * [e_c^2 / (4 * M_h) + F_a] / (e_c * c_a - 1) +
                3 * [e_c * s_a / (e_c * c_a - 1)]{[e_c^2 / (4 * M_h) + F_a] / (e_c * c_a - 1)}^2
            ) / (
                6 +
                6 * [e_c * s_a / (e_c * c_a - 1)]{[e_c^2 / (4 * M_h) + F_a] / (e_c * c_a - 1)} +
                [e_c * c_a / (e_c * c_a - 1)]{[e_c^2 / (4 * M_h) + F_a] / (e_c * c_a - 1)}^2
            )
        ...where:
        e_c = eccentricity
        F_a = rough guess
        c_a = cosh(F_a) = 0.5 * [2 * M_h / e_c + e_c / (2 * M_h)],
        s_a = sinh(F_a) = 0.5 * [2 * M_h / e_c - e_c / (2 * M_h)]

        Although the equation may look intimidating, there are a lot of repeated values.
        We can simplify the equation by extracting the repeated values.

        Let:
            alpha = e_c^2 / (4 * M_h) + F_a
            beta  = 1 / (e_c * c_a - 1)
            gamma = alpha * beta

        The equation gets simplified into:

        delta = (
                6 * gamma +
                3 * e_c * s_a * beta * gamma^2
            ) / (
                6 +
                6 * e_c * s_a * beta * gamma +
                e_c * c_a * beta * gamma^2
            )

        Then we can refine the rough guess into the initial guess:
        F_0 = F_a + delta
        */

        let (c_a, s_a) = {
            // c_a and s_a has a lot of repeated values, so we can
            // optimize by calculating them together.
            // c_a, s_a = 0.5 * [2 * M_h / e_c +- e_c / (2 * M_h)]
            //
            // define "left"  = 2 * M_h / e_c
            // define "right" = e_c / (2 * M_h)

            let left = 2.0 * mean_anomaly / eccentricity;
            let right = eccentricity / (2.0 * mean_anomaly);

            (0.5 * (left + right), 0.5 * (left - right))
        };

        let alpha = eccentricity * eccentricity / (4.0 * mean_anomaly) + rough_guess;

        let beta = (eccentricity * c_a - 1.0).recip();

        let gamma = alpha * beta;
        let gamma_sq = gamma * gamma;

        let delta = (6.0 * alpha * beta + 3.0 * (eccentricity * s_a * beta) * gamma_sq)
            / (6.0
                + 6.0 * (eccentricity * s_a * beta) * gamma
                + (eccentricity * c_a * beta) * gamma_sq);

        rough_guess + delta
    }
}

/// Gets the hyperbolic eccentric anomaly of the orbit.
///
/// # Unchecked Operation
/// This function does not check whether or not the orbit is actually hyperbolic.  
/// Nonsensical output may be produced if the orbit is not hyperbolic, but rather
/// elliptic or parabolic.
///
/// # Performance
/// This function uses numerical methods to approach the value and therefore
/// is not performant. It is recommended to cache this value if you can.
///
/// # Source
/// From the paper:  
/// "A new method for solving the hyperbolic Kepler equation"  
/// by Baisheng Wu et al.  
pub(crate) fn get_hyperbolic_eccentric_anomaly(eccentricity: f64, mean_anomaly: f64) -> f64 {
    let mut ecc_anom = get_approx_hyperbolic_eccentric_anomaly(eccentricity, mean_anomaly);

    /*
    Do a fourth-order Schröder iteration of the second kind

    Equation 25 of "A new method for solving the hyperbolic Kepler equation"
    by Baisheng Wu et al.
    Slightly restructured:

    F_1^(4) = F_0 - (
        (6h/h' - 3h^2 h'' / h'^3) /
        (6 - 6h h'' / h'^2 + h^2 h'''/h'^3)
    )

    ...where:
    e_c = eccentricity
    F_0 = initial guess
    h   = e_c sinh(F_0) - F_0 - M_h
    h'  = e_c cosh(F_0) - 1
    h'' = e_c sinh(F_0)
        = h + F_0 + M_h
    h'''= h' + 1

    Rearranging for efficiency:
    h'''= e_c cosh(F_0)
    h'  = h''' - 1
    h'' = e_c sinh(F_0)
    h   = h'' - F_0 - M_h

    Factoring out 1/h':

    let r = 1 / h'

    F_1^(4) = F_0 - (
        (6hr - 3h^2 h'' r^3) /
        (6 - 6h h'' r^2 + h^2 h''' r^3)
    )

    Since sinh and cosh are very similar algebraically,
    it may be better to calculate them together.

    Paper about Schröder iterations:
    https://doi.org/10.1016/j.cam.2019.02.035
     */

    for _ in 0..NUMERIC_MAX_ITERS {
        let (sinh_eca, cosh_eca) = sinhcosh(ecc_anom);

        let hppp = eccentricity * cosh_eca;
        let hp = hppp - 1.0;
        let hpp = eccentricity * sinh_eca;
        let h = hpp - ecc_anom - mean_anomaly;

        let h_sq = h * h;
        let r = hp.recip();
        let r_sq = r * r;
        let r_cub = r_sq * r;

        let denominator = 6.0 - 6.0 * h * hpp * r_sq + h_sq * hppp * r_cub;

        if denominator.abs() < 1e-30 || !denominator.is_finite() {
            // dangerously close to div-by-zero, break out
            // eprintln!(
            //     "Hyperbolic eccentric anomaly solver: denominator is too small or not finite"
            // );
            break;
        }

        let numerator = 6.0 * h * r - 3.0 * h_sq * hpp * r_cub;
        let delta = numerator / denominator;

        ecc_anom -= delta;

        if delta.abs() < 1e-12 {
            break;
        }
    }

    ecc_anom
}

/// Gets the elliptic eccentric anomaly of the orbit.
///
/// # Unchecked Operation
/// This function does not check whether or not the orbit is actually elliptic (e < 1).  
/// Nonsensical output may be produced if the orbit is not elliptic, but rather
/// hyperbolic or parabolic.
///
/// # Performance
/// This function uses numerical methods to approach the value and therefore
/// is not performant. It is recommended to cache this value if you can.
///
/// # Source
/// From the paper  
/// "An improved algorithm due to laguerre for the solution of Kepler's equation."  
/// by Bruce A. Conway  
/// <https://doi.org/10.1007/bf01230852>
pub(crate) fn get_elliptic_eccentric_anomaly(eccentricity: f64, mut mean_anomaly: f64) -> f64 {
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
    let mut eccentric_anomaly = mean_anomaly
        + (4.0 * eccentricity * B * mean_anomaly * (PI - mean_anomaly))
            / (8.0 * eccentricity * mean_anomaly
                + 4.0 * eccentricity * (eccentricity - PI)
                + (PI * PI));

    // Laguerre's method
    //
    // i = 2, 3, ..., n
    //
    // D = sqrt((n-1)^2(f'(x_i))^2 - n(n-1)f(x_i)f''(x_i))
    //
    // x_i+1 = x_i - (nf(x_i) / (f'(x_i) +/- D))
    // ...where the "+/-" is chosen to so that abs(denominator) is maximized
    for _ in 2..N_U32 {
        let f = keplers_equation(mean_anomaly, eccentric_anomaly, eccentricity);
        let fp = keplers_equation_derivative(eccentric_anomaly, eccentricity);
        let fpp = keplers_equation_second_derivative(eccentric_anomaly, eccentricity);

        let n = N_F64;
        let n_minus_1 = n - 1.0;
        let d = ((n_minus_1 * n_minus_1) * fp * fp - n * n_minus_1 * f * fpp)
            .abs()
            .sqrt()
            .copysign(fp);

        let denominator = n * f / (fp + d.max(1e-30));
        eccentric_anomaly -= denominator;

        if denominator.abs() < 1e-30 || !denominator.is_finite() {
            // dangerously close to div-by-zero, break out
            break;
        }
    }

    eccentric_anomaly * sign
}
