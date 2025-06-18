# Keplerian-sim
A Rust library for simulating Keplerian orbits.

**[Documentation](https://docs.rs/keplerian-sim/)**  
**[Crate](https://crates.io/crates/keplerian-sim)**

## External Dependencies
You'll need Cargo and Rust to run the example binary. You can get it [at rustup.rs](https://rustup.rs/).

## Minimum Supported Rust Version (MSRV)
<!-- When changing the MSRV, don't forget to change `Cargo.toml`! -->
The current MSRV is **`1.81.0`**.  
MSRV changes is out-of-scope for this crate's SemVer guarantees and may change in minor or major releases.  
If this version number and the one in Cargo.toml do not match, then assume the one in Cargo.toml is the correct one, and create an issue regarding the mismatch.

## Running
This project is a library crate, so you can't really 'run' that, but you can run the example binaries.  
1. Clone the repo: `git clone https://github.com/Not-A-Normal-Robot/keplerian-sim`
2. Check out the available example binaries: `cargo run --example`
3. See the list of example binaries: `cargo run --example`
3. Run an example binary: `cargo run --example <binary_name>`

For the examples that dump a CSV, you can use external tools to chart it.  
One example is https://csvplot.com/.

## Benchmarking
You can run `cargo bench` to run the benchmarks in `/benches`. Note that most of them run the function around 1000 times, so you'll have to divide the times you get by 1000 to get the average time for one function call.

The benchmarks use shortened names, and here's what they mean:
- `ecc poll`: The algorithm for obtaining the eccentric anomaly from the mean anomaly in a certain orbit.
- `pos`: The algorithm for obtaining the position of a body at a certain angle in an orbit.
- `pos time`: The algorithm for obtaining the position of a body at a certain time in an orbit.
- `tilt poll`: The algorithm for tilting a certain point from 2D to 3D based on the orbit's inclination and longitude of ascending node.
- `true poll`: The algorithm for obtaining the true anomaly from the mean anomaly in an orbit.
- `hyp`: Specifies that the orbit/trajectory is hyperbolic. If this is not present, then the orbit is elliptic.
- `cached`: Specifies that the orbit struct benchmarked is the regular `Orbit` struct.
- `compact`: Specifies that the orbit struct benchmarked is the `CompactOrbit` struct instead of the regular `Orbit` struct.

## Testing
You can run `cargo test` to run the tests.

## Resources
I did not come up with the algorithms myself. For more information and useful resources to learn about the algorithms used, check out the `CREDITS.md` file.