use core::f64::NAN;
use glam::{DVec2, DVec3};
use keplerian_sim::{Orbit, OrbitTrait};
use std::{env::args, fs, process::exit};

const DEFAULT_SIMULATION_TICKS: usize = 10_000;
const CSV_PATH: &str = "out/output-orbit-dump.csv";

fn get_sim_ticks() -> usize {
    let mut ticks = DEFAULT_SIMULATION_TICKS;

    let mut cur_val_is_ticks = false;

    for arg in args() {
        if arg == "--ticks" || arg == "-t" {
            cur_val_is_ticks = true;
            continue;
        }

        if cur_val_is_ticks {
            ticks = match usize::from_str_radix(&arg, 10) {
                Ok(n) => n,
                Err(_) => {
                    println!(
                        "Failed to parse {arg} as valid number\
                        Please input a valid number after the --ticks argument."
                    );
                    exit(1);
                }
            };
            cur_val_is_ticks = false;
        }
    }

    ticks
}

struct OrbitWrapper {
    name: &'static str,
    orbit: Orbit,
    time_mult: f64,
    time_offset: f64,
}

struct OrbitLog<'a> {
    name: &'a str,

    iter: usize,
    time: f64,
    mean_anom: f64,
    ecc_anom: f64,
    angle: f64,

    position: DVec3,
    flat_position: DVec2,

    altitude: f64,

    expected_speed: f64,
    real_speed: f64,
    real_flat_vel: DVec2,
    real_vel: DVec3,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let orbits = [
        OrbitWrapper {
            name: "The Moon",
            orbit: Orbit::with_apoapsis(
                405400.0,
                362600.0,
                5.145_f64.to_radians(),
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            time_mult: 1.0,
            time_offset: 0.0,
        },
        OrbitWrapper {
            name: "Eccentricity 0.85",
            orbit: Orbit::new(0.85, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 1.0,
            time_offset: 0.0,
        },
        OrbitWrapper {
            name: "Eccentricity 0.99",
            orbit: Orbit::new(0.99, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 1.0,
            time_offset: 0.0,
        },
        OrbitWrapper {
            name: "Eccentricity 1",
            orbit: Orbit::new(1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 10.0,
            time_offset: -5.0,
        },
        OrbitWrapper {
            name: "Eccentricity 1.001",
            orbit: Orbit::new(1.001, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 10.0,
            time_offset: -5.0,
        },
        OrbitWrapper {
            name: "Eccentricity 1.002",
            orbit: Orbit::new(1.002, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 10.0,
            time_offset: -5.0,
        },
        OrbitWrapper {
            name: "Eccentricity 1.005",
            orbit: Orbit::new(1.005, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 10.0,
            time_offset: -5.0,
        },
        OrbitWrapper {
            name: "Eccentricity 1.01",
            orbit: Orbit::new(1.01, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 10.0,
            time_offset: -5.0,
        },
        OrbitWrapper {
            name: "Eccentricity 1.02",
            orbit: Orbit::new(1.02, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 10.0,
            time_offset: -5.0,
        },
        OrbitWrapper {
            name: "Eccentricity 1.1",
            orbit: Orbit::new(1.1, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 10.0,
            time_offset: -5.0,
        },
        OrbitWrapper {
            name: "Eccentricity 1.5",
            orbit: Orbit::new(1.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 10.0,
            time_offset: -5.0,
        },
        OrbitWrapper {
            name: "Eccentricity 2.5",
            orbit: Orbit::new(2.5, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 10.0,
            time_offset: -5.0,
        },
        OrbitWrapper {
            name: "Eccentricity 12",
            orbit: Orbit::new(12.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 50.0,
            time_offset: -25.0,
        },
        OrbitWrapper {
            name: "Eccentricity 1000",
            orbit: Orbit::new(1000.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 1000.0,
            time_offset: -500.0,
        },
        OrbitWrapper {
            name: "Escape trajectory",
            orbit: Orbit::new(2.0, 1.496e11, 0.0, 0.0, 0.0, 0.0, 1.0),
            time_mult: 100.0,
            time_offset: -5.0,
        },
    ];

    let simulation_ticks = get_sim_ticks();

    let mut logs: Vec<OrbitLog> = Vec::with_capacity(orbits.len() * simulation_ticks);

    println!(">> Begin simulations.");

    for named_orbit in orbits {
        let (name, orbit) = (named_orbit.name, named_orbit.orbit);

        println!("Simulating orbit {name} for {simulation_ticks} ticks");

        for iter in 0..simulation_ticks {
            let time = iter as f64 / simulation_ticks as f64;
            let time = named_orbit.time_mult * time + named_orbit.time_offset;

            let mean_anom = orbit.get_mean_anomaly_at_time(time);
            let ecc_anom = orbit.get_eccentric_anomaly_at_time(time);
            let angle = orbit.get_true_anomaly_at_eccentric_anomaly(ecc_anom);

            let altitude = orbit.get_altitude_at_true_anomaly(angle);
            let flat_position = orbit.get_pqw_position_at_true_anomaly(angle);
            let position = orbit.get_position_at_true_anomaly(angle);

            let expected_speed = (2.0 / altitude - 1.0 / orbit.get_semi_major_axis()).sqrt();
            let real_speed = orbit.get_speed_at_true_anomaly(angle);
            let real_vel = orbit.get_velocity_at_eccentric_anomaly(ecc_anom);
            let real_flat_vel = orbit.get_pqw_velocity_at_eccentric_anomaly(ecc_anom);

            logs.push(OrbitLog {
                name,
                iter,
                time,
                mean_anom,
                ecc_anom,
                angle,
                position,
                flat_position,
                altitude,
                expected_speed,
                real_speed,
                real_flat_vel,
                real_vel,
            });
        }
    }

    println!(">> Creating CSV file.");

    let csv = create_csv(&logs);

    println!("CSV file generated, length: {} KiB", csv.len() / 1024);

    if args().all(|a| a != "--no-write") {
        println!(">> Writing CSV file to '{CSV_PATH}'");

        let result = fs::write(CSV_PATH, csv);

        if let Err(e) = result {
            eprintln!("Failed to write CSV file: {e}");
        }
    }

    println!(">> All done!");

    return Ok(());
}

fn create_csv(logs: &Vec<OrbitLog>) -> String {
    let mut string = String::new();
    string += "orbit,iter,time,mean-anom,ecc-anom,angle,\
        flat-x,flat-y,naive-dfx,naive-dfy,x,y,z,\
        naive-dx,naive-dy,naive-dz,altitude,d-altitude,\
        naive-speed,accel,expected-speed,real-speed,real-dfx,real-dfy,\
        real-dx,real-dy,real-dz\n";

    let mut prev_orbit_type: &str = "";
    let mut prev_flat_pos: Option<DVec2> = None;
    let mut prev_pos: Option<DVec3> = None;
    let mut prev_altitude: Option<f64> = None;
    let mut prev_speed: Option<f64> = None;

    for log in logs {
        if prev_orbit_type != log.name {
            prev_orbit_type = log.name;
            prev_flat_pos = None;
            prev_pos = None;
            prev_altitude = None;
            prev_speed = None;
        }

        let altitude = log.altitude;

        let prev_flat_position = prev_flat_pos.unwrap_or(DVec2::NAN);
        let prev_position = prev_pos.unwrap_or(DVec3::NAN);
        let prev_altitude_unwrapped = prev_altitude.unwrap_or(NAN);

        let d = log.position - prev_position;

        let df = log.flat_position - prev_flat_position;

        let d_altitude = altitude - prev_altitude_unwrapped;

        let speed = df.length();
        let accel = speed - prev_speed.unwrap_or(NAN);

        string += &format!(
            "{orbit},{iter},{time},{mean_anom},{ecc_anom},{angle},{flat_x},{flat_y},{dfx},{dfy},{x},{y},{z},{dx},{dy},{dz},{altitude},{d_altitude},{speed},{accel},{expected_speed},{real_speed},{real_dfx},{real_dfy},{real_dx},{real_dy},{real_dz}\n",
            orbit = log.name,
            iter = log.iter,
            time = log.time,
            mean_anom = log.mean_anom,
            ecc_anom = log.ecc_anom,
            angle = log.angle,
            flat_x = log.flat_position.x,
            flat_y = log.flat_position.y,
            dfx = df.x,
            dfy = df.y,
            x = log.position.x,
            y = log.position.y,
            z = log.position.z,
            dx = d.x,
            dy = d.y,
            dz = d.z,
            expected_speed = log.expected_speed,
            real_speed = log.real_speed,
            real_dfx = log.real_flat_vel.x,
            real_dfy = log.real_flat_vel.y,
            real_dx = log.real_vel.x,
            real_dy = log.real_vel.y,
            real_dz = log.real_vel.z,
        );

        prev_flat_pos = Some(prev_flat_position);
        prev_pos = Some(prev_position);
        prev_altitude = Some(altitude);
        prev_speed = Some(speed);
    }

    string
}
