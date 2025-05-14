import numpy as np


class Integrator:
    """
    Base class for all numerical integrators.

    This class defines the interface that all integrators must implement.
    The timestep dt controls only the accuracy of the simulation, not its duration.
    """

    def __init__(self, name):
        """
        Initialize the integrator.

        Args:
            name: A string identifier for the integrator
        """
        self.name = name

    def step(self, system, t, dt):
        """
        Take a single step of size dt from time t.

        This is the core method that specific integrators must implement.

        Args:
            system: System object with state and derivatives methods
            t: Current time
            dt: Time step size (controls accuracy)

        Returns:
            Updated state vector
        """
        raise NotImplementedError("Subclasses must implement step method")

    def integrate(self, system, t_span, n_steps=1000, callback=None):
        """
        Integrate the system over the given time span.

        The timestep dt is calculated based on the t_span and n_steps,
        so changing n_steps affects the accuracy without changing the
        simulation duration.

        Args:
            system: System object with state and derivatives methods
            t_span: [t_start, t_end] time span to integrate over
            n_steps: Number of steps to take (controls accuracy)
            callback: Optional function called after each step with (system, t, data_dict)

        Returns:
            Dictionary containing:
                - times: Array of time points
                - states: Array of states at each time point
                - additional data collected by callback
        """
        t_start, t_end = t_span
        dt = (t_end - t_start) / n_steps
        times = np.linspace(t_start, t_end, n_steps + 1)

        # Initialize data storage
        states = np.zeros((n_steps + 1, len(system.state)))
        states[0] = system.state.copy()

        # Data collection from callback
        callback_data = {}
        if callback:
            callback(system, times[0], callback_data)

        # Integration loop
        for i in range(1, n_steps + 1):
            new_state = self.step(system, times[i - 1], dt)
            system.state = new_state

            # Check if step was rejected
            if hasattr(self, 'rejected_step') and self.rejected_step:
                # If step was rejected, adjust dt and retry the step
                dt = self.suggested_dt
                times = np.linspace(t_start, t_end, n_steps + 1)  # Recalculate times
                i -= 1  # Retry this step
                continue

            states[i] = system.state.copy()

            if callback:
                callback(system, times[i], callback_data)

        return {
            'times': times,
            'states': states,
            'data': callback_data,
            'dt': dt
        }
