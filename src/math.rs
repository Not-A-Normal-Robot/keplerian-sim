#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("Either std or libm must be used for math operations");

#[cfg(feature = "std")]
extern crate std;

macro_rules! libm_or_std {
    ( $( $fname:ident $( / $lname:ident )? $( : $arity:tt )? $( -> $ret:tt )? ),* $(,)? ) => {
        $(
            libm_or_std!(@make fn $fname $( / $lname )? $( : $arity )? $( -> $ret )? );
        )*
    };

    (@make fn $fname:ident) => {
        pub fn $fname(x: f64) -> f64 {
            #[cfg(feature = "std")]
            { x.$fname() }
            #[cfg(feature = "libm")]
            { libm::$fname(x) }
        }
    };

    (@make fn $fname:ident : 2) => {
        pub fn $fname(x: f64, y: f64) -> f64 {
            #[cfg(feature = "std")]
            { x.$fname(y) }
            #[cfg(feature = "libm")]
            { libm::$fname(x, y) }
        }
    };

    (@make fn $fname:ident / $lname:ident) => {
        pub fn $fname(x: f64) -> f64 {
            #[cfg(feature = "std")]
            { x.$fname() }
            #[cfg(feature = "libm")]
            { libm::$lname(x) }
        }
    };

    (@make fn $fname:ident / $lname:ident -> 2) => {
        pub fn $fname(x: f64) -> (f64, f64) {
            #[cfg(feature = "std")]
            { x.$fname() }
            #[cfg(feature = "libm")]
            { libm::$lname(x) }
        }
    };
}

libm_or_std!(
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    sqrt,
    cbrt,
    exp,
    ln/log,
    atan2: 2,
    sin_cos/sincos -> 2,
);
