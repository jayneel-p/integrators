import matplotlib.pyplot as plt
import numpy as np


def plot_all_planets(results, system, title=None):
    """
    Plot the trajectories of all planets in the solar system.

    Args:
        results: Results dictionary from integration
        system: Solar system object
        title: Optional title for the plot

    Returns:
        Matplotlib figure
    """
    # Create figure
    fig = plt.figure(figsize=(15, 15))

    if title is None:
        plt.title(
            f"Solar System - {results.get('integrator_name', 'Unknown Integrator')} - {results.get('years', 0)} years")
    else:
        plt.title(title)

    # Colors for the planets
    colors = ['gold', 'gray', 'blue', 'orange', 'red',
              'brown', 'goldenrod', 'cyan', 'navy','black']

    # Plot each planet
    for i, body in enumerate(system.bodies):


        # Get position data
        x = results['states'][:, i * 6]
        y = results['states'][:, i * 6 + 1]

        # Plot orbit
        plt.plot(x, y, label=body['name'], color=colors[i])

        # Mark final position
        plt.scatter(x[-1], y[-1], s=30, color=colors[i])

    # Make it pretty
    plt.axis('equal')
    plt.grid(True)
    plt.legend()

    return fig

def plot_energy_error(results, title=None):
    """
    Plot the relative energy error over time.

    Args:
        results: Results dictionary from integration
        title: Optional title for the plot

    Returns:
        Matplotlib figure
    """
    if 'energy' not in results.get('data', {}):
        return None

    # Create figure
    fig = plt.figure(figsize=(10, 6))

    if title is None:
        plt.title(f"Energy Error - {results.get('integrator_name', 'Unknown Integrator')}")
    else:
        plt.title(title)

    # Get data
    times = results['times']
    energy = np.array(results['data']['energy'])
    initial_energy = energy[0]
    energy_error = (energy - initial_energy) / abs(initial_energy)
    
    # Plot
    plt.plot(times, energy_error)
    plt.xlabel("Time (years)")
    plt.ylabel("Relative Energy Error")
    plt.grid(True)

    return fig