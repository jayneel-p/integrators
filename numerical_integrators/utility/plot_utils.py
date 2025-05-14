import matplotlib.pyplot as plt
from utils import ensure_directory_exists
from trajectoryPlot_visualization import plot_all_planets, plot_energy_error


def save_trajectory_plots(results, system, filename_suffix=""):
    """
    Save trajectory plots for a given simulation.

    Args:
        results: Results dictionary from integration
        system: Solar system object
        filename_suffix: Optional suffix to add to filename
    """
    ensure_directory_exists()

    # Construct filename
    if filename_suffix:
        filename = f"plots/{results['integrator_name']}_{filename_suffix}.png"
    else:
        filename = f"plots/{results['integrator_name']}.png"

    # Plot using existing function
    fig1 = plot_all_planets(results, system)
    fig1.savefig(filename)
    plt.close(fig1)

    print(f"Saved trajectory plots for {results['integrator_name']} - {results.get('system_name', '')}")


def save_individual_energy_plot(results, filename_suffix=""):
    """
    Save an energy error plot for a single integration method.

    Args:
        results: Results dictionary from integration
        filename_suffix: Optional suffix to add to filename
    """
    ensure_directory_exists()

    # Construct filename
    if filename_suffix:
        filename = f"plots/energy_{results['integrator_name']}_{filename_suffix}.png"
    else:
        filename = f"plots/energy_{results['integrator_name']}.png"

    # Use existing function
    fig = plot_energy_error(results)

    if fig is not None:
        fig.savefig(filename)
        plt.close(fig)
        print(f"Saved energy plot for {results['integrator_name']} - {results.get('system_name', '')}")
    else:
        print(f"No energy data found for {results['integrator_name']} - {results.get('system_name', '')}")