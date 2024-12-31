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

    let sun = body_presets::stars::the_sun(false);
    let sun_idx = universe.add_body(sun, None);

    let mercury = body_presets::planets::mercury(true);
    universe.add_body(mercury, Some(sun_idx));

    let venus = body_presets::planets::venus(true);
    universe.add_body(venus, Some(sun_idx));

    let earth = body_presets::planets::earth(true);
    let earth_idx = universe.add_body(earth, Some(sun_idx));

    let moon = body_presets::moons::the_moon(true);
    universe.add_body(moon, Some(earth_idx));

    let mars = body_presets::planets::mars(true);
    universe.add_body(mars, Some(sun_idx));

    let ceres = body_presets::dwarf_planets::ceres(true);
    universe.add_body(ceres, Some(sun_idx));

    let jupiter = body_presets::planets::jupiter(true);
    universe.add_body(jupiter, Some(sun_idx));

    let saturn = body_presets::planets::saturn(true);
    universe.add_body(saturn, Some(sun_idx));

    let uranus = body_presets::planets::uranus(true);
    universe.add_body(uranus, Some(sun_idx));

    let neptune = body_presets::planets::neptune(true);
    universe.add_body(neptune, Some(sun_idx));

    let pluto = body_presets::dwarf_planets::pluto(true);
    let pluto_idx = universe.add_body(pluto, Some(sun_idx));

    let makemake = body_presets::dwarf_planets::makemake(true);
    universe.add_body(makemake, Some(sun_idx));

    let eris = body_presets::dwarf_planets::eris(true);
    let eris_idx = universe.add_body(eris, Some(sun_idx));

    let sedna = body_presets::dwarf_planets::sedna(true);
    universe.add_body(sedna, Some(sun_idx));

    let haumea = body_presets::dwarf_planets::haumea(true);
    universe.add_body(haumea, Some(sun_idx));

    let quaoar = body_presets::dwarf_planets::quaoar(true);
    let quaoar_idx = universe.add_body(quaoar, Some(sun_idx));

    let weywot = body_presets::moons::weywot(true);
    universe.add_body(weywot, Some(quaoar_idx));

    let charon = body_presets::moons::charon(true);
    universe.add_body(charon, Some(pluto_idx));

    let dysnomia = body_presets::moons::dysnomia(true);
    universe.add_body(dysnomia, Some(eris_idx));

    return universe;
}

fn describe_universe(universe: &Universe) {
    println!(
        "Simulation universe with {} bodies",
        universe.get_bodies().len()
    );
    for (i, body) in universe.get_bodies().iter().enumerate() {
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
    for (i, body) in universe.get_bodies().iter().enumerate() {
        let location = universe.get_body_position(i);
        println!("{}: {:?}", body.name, location);
    }
}
