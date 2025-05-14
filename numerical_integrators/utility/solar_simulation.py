def run_simulation(integrator_class, years=200, n_steps=50000, system_class=None):
    """
    Run a simulation with the given integrator class.

    Args:
        integrator_class: Class of the integrator to use
        years: Number of years to simulate
        n_steps: Number of steps for the integrator
        system_class: System class to instantiate (defaults to SolarSystem)

    Returns:
        Results dictionary, system object
    """
    # Default to SolarSystem if no system class provided
    if system_class is None:
        from solar_system import SolarSystem
        system_class = SolarSystem

    # Create the system
    system = system_class()

    # Create the integrator
    integrator = integrator_class()

    print(f"Running simulation with {integrator.name} for {years} years...")

    # Define callback to collect energy
    def collect_energy(system, t, data):
        if 'energy' not in data:
            data['energy'] = []
        data['energy'].append(system.energy())

    # Run the simulation
    results = integrator.integrate(system, [0, years], n_steps=n_steps, callback=collect_energy)

    # Add some metadata to results
    results['integrator_name'] = integrator.name
    results['years'] = years
    results['system_name'] = system.__class__.__name__  # Add system name

    print(f"Simulation complete. dt = {results['dt']}")

    return results, system