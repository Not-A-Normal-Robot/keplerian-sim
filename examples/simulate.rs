use keplerian_sim::{body_presets, OrbitTrait, Universe};

const SIMULATION_TICKS: u128 = 1_000_000;
fn main() {
    let mut universe = generate_solar_system();
    describe_universe(&universe);
    print!("Simulating {SIMULATION_TICKS} ticks...");
    universe.warp(SIMULATION_TICKS);
    println!(" done");
    print_all_body_positions(&universe);
}

fn generate_solar_system<'a>() -> Universe {
    let mut universe = Universe::new_default();

    let sun = body_presets::stars::the_sun(None);
    let sun_mu = sun.mass * universe.g;
    let sun_idx = universe.add_body(sun, None).unwrap();

    let mercury = body_presets::planets::mercury(Some(sun_mu));
    universe.add_body(mercury, Some(sun_idx)).unwrap();

    let venus = body_presets::planets::venus(Some(sun_mu));
    universe.add_body(venus, Some(sun_idx)).unwrap();

    let earth = body_presets::planets::earth(Some(sun_mu));
    let earth_mu = earth.mass * universe.g;
    let earth_idx = universe.add_body(earth, Some(sun_idx)).unwrap();

    let moon = body_presets::moons::the_moon(Some(earth_mu));
    universe.add_body(moon, Some(earth_idx)).unwrap();

    let mars = body_presets::planets::mars(Some(sun_mu));
    universe.add_body(mars, Some(sun_idx)).unwrap();

    let ceres = body_presets::dwarf_planets::ceres(Some(sun_mu));
    universe.add_body(ceres, Some(sun_idx)).unwrap();

    let jupiter = body_presets::planets::jupiter(Some(sun_mu));
    universe.add_body(jupiter, Some(sun_idx)).unwrap();

    let saturn = body_presets::planets::saturn(Some(sun_mu));
    universe.add_body(saturn, Some(sun_idx)).unwrap();

    let uranus = body_presets::planets::uranus(Some(sun_mu));
    universe.add_body(uranus, Some(sun_idx)).unwrap();

    let neptune = body_presets::planets::neptune(Some(sun_mu));
    universe.add_body(neptune, Some(sun_idx)).unwrap();

    let pluto = body_presets::dwarf_planets::pluto(Some(sun_mu));
    let pluto_mu = pluto.mass * universe.g;
    let pluto_idx = universe.add_body(pluto, Some(sun_idx)).unwrap();

    let makemake = body_presets::dwarf_planets::makemake(Some(sun_mu));
    universe.add_body(makemake, Some(sun_idx)).unwrap();

    let eris = body_presets::dwarf_planets::eris(Some(sun_mu));
    let eris_mu = eris.mass * universe.g;
    let eris_idx = universe.add_body(eris, Some(sun_idx)).unwrap();

    let sedna = body_presets::dwarf_planets::sedna(Some(sun_mu));
    universe.add_body(sedna, Some(sun_idx)).unwrap();

    let haumea = body_presets::dwarf_planets::haumea(Some(sun_mu));
    universe.add_body(haumea, Some(sun_idx)).unwrap();

    let quaoar = body_presets::dwarf_planets::quaoar(Some(sun_mu));
    let quaoar_mu = quaoar.mass * universe.g;
    let quaoar_idx = universe.add_body(quaoar, Some(sun_idx)).unwrap();

    let weywot = body_presets::moons::weywot(Some(quaoar_mu));
    universe.add_body(weywot, Some(quaoar_idx)).unwrap();

    let charon = body_presets::moons::charon(Some(pluto_mu));
    universe.add_body(charon, Some(pluto_idx)).unwrap();

    let dysnomia = body_presets::moons::dysnomia(Some(eris_mu));
    universe.add_body(dysnomia, Some(eris_idx)).unwrap();

    return universe;
}

fn describe_universe(universe: &Universe) {
    println!(
        "Simulation universe with {} bodies",
        universe.get_bodies().len()
    );
    for (&i, wrapper) in universe.get_bodies().iter() {
        let body = &wrapper.body;
        println!("    {}: {:?}", i, body.name);
        println!("      Mass: {}", body.mass);
        println!("      Radius: {}", body.radius);
        if let Some(orbit) = &body.orbit {
            let location = universe.get_body_position(i);
            println!("      Orbit: {:?}", location);
            println!("        Semi-major axis: {}", orbit.get_semi_major_axis());
            println!("        Eccentricity: {}", orbit.get_eccentricity());
            println!("        Inclination: {}", orbit.get_inclination());
            println!("        Argument of periapsis: {}", orbit.get_arg_pe());
            println!(
                "        Longitude of ascending node: {}",
                orbit.get_long_asc_node()
            );
            println!(
                "        Mean anomaly at epoch: {}",
                orbit.get_mean_anomaly_at_epoch()
            );
        }
    }
}

fn print_all_body_positions(universe: &Universe) {
    for (&i, w) in universe.get_bodies().iter() {
        let location = universe.get_body_position(i);
        println!("{}: {:?}", w.body.name, location);
    }
}
