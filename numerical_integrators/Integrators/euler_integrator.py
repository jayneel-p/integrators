from base_integrator import Integrator


class Euler(Integrator):
    """
    First-order Euler integrator (RK1).
    Suitable for simple simulations.
        Low accuracy over high time scales.

    """

    def __init__(self):
        """Initialize the Euler integrator."""
        super().__init__("Euler")

    def step(self, system, t, dt):
        """
        Take a single Euler step.

        Args:
            system: System object with state and derivatives methods
            t: Current time
            dt: Time step size

        Returns:
            Updated state vector
        """
        # Get current state
        state = system.state.copy()

        # Compute derivatives (acceleration for orbits)
        derivatives = system.derivatives(state, t)

        # Update state using Euler's method
        new_state = state + dt * derivatives

        return new_state
