import numpy as np
from base_integrator import Integrator

class RK4(Integrator):
    """
    Fourth-order Runge-Kutta integrator.

    This is a classic fourth-order integrator that provides good accuracy
    for most problems, though it is not symplectic.
    """

    def __init__(self):
        """Initialize the RK4 integrator."""
        super().__init__("RK4")

    def step(self, system, t, dt):
        """
        Take a single RK4 step.

        Args:
            system: System object with state and derivatives methods
            t: Current time
            dt: Time step size

        Returns:
            Updated state vector
        """
        # Get current state
        state = system.state.copy()

        # Calculate the four RK4 stages
        k1 = system.derivatives(state, t)
        k2 = system.derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = system.derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = system.derivatives(state + dt * k3, t + dt)

        # Update state using weighted average of slopes
        new_state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return new_state
