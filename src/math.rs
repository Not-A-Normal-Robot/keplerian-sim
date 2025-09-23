macro_rules! trait_decl {
    ( $( $fname:ident $( / $_lname:ident )? $( : $arity:tt )? $( -> $ret:tt )? ),* $(,)? ) => {
        $(
            trait_decl!(@make fn $fname $( : $arity )? $( -> $ret )? );
        )*
    };

    (@make fn $fname:ident) => {
        fn $fname(self) -> Self;
    };

    (@make fn $fname:ident : 2) => {
        fn $fname(self, y: Self) -> Self;
    };

    (@make fn $fname:ident -> 2) => {
        fn $fname(self) -> (Self, Self);
    };
}

macro_rules! trait_impl {
    ( $( $fname:ident $( / $lname:ident )? $( : $arity:tt )? $( -> $ret:tt )? ),* $(,)? ) => {
        $(
            trait_impl!(@make fn $fname $( / $lname )? $( : $arity )? $( -> $ret )? );
        )*
    };

    (@make fn $fname:ident) => {
        fn $fname(self) -> Self {
            libm::$fname(self)
        }
    };

    (@make fn $fname:ident : 2) => {
        fn $fname(self, y: Self) -> Self {
            libm::$fname(self, y)
        }
    };

    (@make fn $fname:ident / $lname:ident) => {
        fn $fname(self) -> Self {
            libm::$lname(self)
        }
    };

    (@make fn $fname:ident / $lname:ident -> 2) => {
        fn $fname(self) -> (Self, Self) {
            libm::$lname(self)
        }
    };
}

#[allow(dead_code)]
pub(crate) trait F64Math: Sized {
    trait_decl!(
        abs,
        signum,
        copysign: 2,
        sin,
        cos,
        tan,
        sinh,
        cosh,
        tanh,
        asin,
        acos,
        atan,
        atanh,
        sqrt,
        cbrt,
        exp,
        ln/log,
        atan2: 2,
        sin_cos/sincos -> 2,
        rem_euclid: 2,
    );
    fn powi(self, i: i32) -> Self;
}

impl F64Math for f64 {
    trait_impl!(
        abs/fabs,
        copysign: 2,
        sin,
        cos,
        tan,
        sinh,
        cosh,
        tanh,
        asin,
        acos,
        atan,
        atanh,
        sqrt,
        cbrt,
        exp,
        ln/log,
        atan2: 2,
        sin_cos/sincos -> 2,
    );

    #[inline(always)]
    fn powi(self, i: i32) -> Self {
        if i == 2 {
            self * self
        } else if i == 3 {
            self * self * self
        } else if cfg!(debug_assertions) {
            panic!("powi only supports up to 3 in no_std environments");
        } else {
            0.0
        }
    }

    fn rem_euclid(self, rhs: f64) -> Self {
        // Copied from source of core::f64::math::rem_euclid
        let r = self % rhs;
        if r < 0.0 {
            r + rhs.abs()
        } else {
            r
        }
    }

    fn signum(self) -> Self {
        libm::copysign(1.0, self)
    }
}
