#[cfg(not(any(feature = "std", feature = "libm")))]
compile_error!("Either std or libm must be used for math operations");

#[cfg(feature = "std")]
extern crate std;

macro_rules! libm_or_std {
    ( $( $fname:ident $( : $arity:tt )? ),* $(,)? ) => {
        $(
            libm_or_std!(@make fn $fname $( : $arity )?);
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
}

libm_or_std!(
    sin,
    cos,
    tan,
    sinh,
    cosh,
    tanh,
    atan2: 2,
);
