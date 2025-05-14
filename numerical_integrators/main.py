import matplotlib.pyplot as plt
# Import util functions
from solar_simulation import run_simulation
from plot_utils import save_trajectory_plots, save_individual_energy_plot
# Import system and integrator
from pythagorean_system import PythagoreanSystem
from ias15_integrator import IAS15
from euler_integrator import Euler

def main():
    # List of integrators to run
    integrators = [IAS15]

    # Define systems to simulate
    systems = [
        {"class": PythagoreanSystem, "name": "Pythagorean", "years": 10}
    ]

    # Run simulations for each system and integrator combination
    for system_info in systems:
        system_class = system_info["class"]
        years = system_info["years"]

        print(f"\n=== Starting {system_info['name']} Simulations ===\n")

        for integrator_class in integrators:
            # Adjust steps based on integrator
            if integrator_class == Euler:
                n_steps = 200000  # Need more steps for Euler stability
            else:
                n_steps = 100000

            # Run simulation with the specified system class
            results, system = run_simulation(
                integrator_class,
                years=years,
                n_steps=n_steps,
                system_class=system_class
            )

            # Add filename suffix to differentiate between systems
            system_suffix = system_info['name'].replace(' ', '_').lower()

            # Save trajectory plots
            save_trajectory_plots(results, system, filename_suffix=system_suffix)

            # Save individual energy plot
            save_individual_energy_plot(results, filename_suffix=system_suffix)

            # Display plots
            plt.show()


if __name__ == "__main__":
    main()