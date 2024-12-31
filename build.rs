use std::fs;

fn main() {
    assert_eq!(PIECEWISE_VALUES.len(), PIECEWISE_CUTOFFS.len());

    let code = generate_sinh_approximator();

    fs::write(SINH_APPROXIMATOR_FILE_PATH, code).expect("Failed to write generated code to file");
}

const SINH_APPROXIMATOR_FILE_PATH: &str = "src/generated_sinh_approximator.rs";

const PIECEWISE_VALUES: [f64; 15] = [
    // The paper says to approximate the sinh function at these values:
    // n/4 for n = 1..8,
    // m/2 for m = 5..10
    // ...which is equivalent to:
    0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 3.50, 4.00, 4.50, 5.00,
];

const PIECEWISE_CUTOFFS: [f64; 15] = [
    // Function number n is used when F is in
    // the interval [cutoffs[n], cutoffs[n + 1]).
    //
    // In other words, it's an if-else chain like:
    // if F < cutoffs[0] {
    //     function_1(F);
    // else if F < cutoffs[1] ...
    //
    // The paper says that the cutoffs are:
    // 29/200,
    // (2n + 1)/8 for n = 1..7,
    // (2m + 1)/4 for m = 4..9
    // 5 (although here we'd use a big number because it doesn't
    // matter and we switch to another algo at 5 anyway)
    0.145,
    0.375,
    0.625,
    0.875,
    1.125,
    1.375,
    1.625,
    1.875,
    2.250,
    2.750,
    3.250,
    3.750,
    4.250,
    4.750,
    f64::MAX,
];

/// Generates a piecewise function for the Pade approximations
/// for the sinh function for the interval [0, 5).
///
/// Returns a string containing Rust code.
///
/// From the paper "A new method for solving the hyperbolic Kepler equation"
/// (https://doi.org/10.1016/j.apm.2023.12.017)
/// Section: "Initial approximation for the F-interval [0, 5)"
/// Equation 4
fn generate_sinh_approximator() -> String {
    let mut code = format!(
        r"
        // Generated by build.rs:generate_sinh_approximator
        #![allow(clippy::all)]
        #![allow(dead_code)]

        pub struct SinhApprParams {{
            pub a: f64,
            pub s: f64,
            pub p_1: f64,
            pub p_2: f64,
            pub p_3: f64,
            pub q_1: f64,
            pub q_2: f64,
        }}

        pub fn sinh_approx_lt5_inner(
            f: f64, args: SinhApprParams
        ) -> f64 {{
            let f_min_a = f - args.a;
            let f_min_a_sq = f_min_a * f_min_a;
            let f_min_a_cu = f_min_a * f_min_a_sq;

            let numerator = 
                args.s + 
                args.p_1 * f_min_a + 
                args.p_2 * f_min_a_sq + 
                args.p_3 * f_min_a_cu;
            
            let denominator =
                1.0 + 
                args.q_1 * f_min_a + 
                args.q_2 * f_min_a_sq;

            return numerator / denominator;
        }}    

        /// Approximates the sinh(F) function for F in the interval [0, 5).
        pub fn sinh_approx_lt5(f: f64) -> f64 {{
            sinh_approx_lt5_inner(f, get_sinh_approx_params(f))
        }}

        /// Returns the parameters for the sinh approximation for the given point.
        pub fn get_sinh_approx_params(f: f64) -> SinhApprParams {{
        ",
    );

    for i in 0..PIECEWISE_VALUES.len() {
        // sinh(F) ~= P_a(F) ===
        //   (s + p_1(F - a) + p_2(F - a)^2 + p_3(F - a)^3) /
        //   (1 + q_1(F - a) + q_2(F - a)^2)
        //
        //...where:
        // a = the point to approximate near
        // s = sinh(a)
        // c = cosh(a)
        // d_1 = c^2 + 3
        // d_2 = s^2 + 4
        // p_1 = c(3c^2 + 17) / (5d_1)
        // p_2 = s(3s^2 + 28) / (20d_2)
        // p_3 = c(c^2 + 27) / (60d_1)
        // q_1 = -2cs / (5d_1)
        // q_2 = (s^2 - 4) / (20d_2)

        let a = PIECEWISE_VALUES[i];
        let s = a.sinh();
        let c = a.cosh();
        let d_1 = c * c + 3.0;
        let d_2 = s * s + 4.0;
        let p_1 = c * (3.0 * c * c + 17.0) / (5.0 * d_1);
        let p_2 = s * (3.0 * s * s + 28.0) / (20.0 * d_2);
        let p_3 = c * (c * c + 27.0) / (60.0 * d_1);
        let q_1 = -2.0 * c * s / (5.0 * d_1);
        let q_2 = (s * s - 4.0) / (20.0 * d_2);

        let cutoff = PIECEWISE_CUTOFFS[i];

        code += format!(
            r"if f < {cutoff:.3} {{
                return SinhApprParams {{
                    a: {a:.80}, s: {s:.80},
                    p_1: {p_1:.80}, p_2: {p_2:.80}, p_3: {p_3:.80},
                    q_1: {q_1:.80}, q_2: {q_2:.80}
                }};
            }} else "
        )
        .as_str();
    }

    code += "{ unreachable!(); } }";
    code += format!("pub const SINH_5: f64 = {};", 5.0_f64.sinh()).as_str();

    return code;
}
