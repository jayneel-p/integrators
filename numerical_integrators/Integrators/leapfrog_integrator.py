import numpy as np
from base_integrator import Integrator


class Leapfrog(Integrator):
    """
    Leapfrog (Störmer-Verlet) symplectic integrator.

    This integrator is particularly well-suited for orbital mechanics and
    gravitational simulations because it conserves energy very well over
    long time periods, even with fixed time steps.
    """

    def __init__(self):
        super().__init__("Leapfrog")

    def step(self, system, t, dt):
        """
        Take a single Leapfrog step using the Störmer-Verlet algorithm.

        The Leapfrog/Verlet method proceeds in three distinct phases:
        1. Half-step velocity update (kick)
        2. Full-step position update (drift)
        3. Half-step velocity update (kick)

        This "kick-drift-kick" sequence gives second-order accuracy while
        preserving the symplectic (energy-conserving) property.

        Args:
            system: System object containing state and body information
            t: Current time
            dt: Time step size

        Returns:
            Updated state vector
        """
        # Create a copy of the current state to avoid modifying the original
        state = system.state.copy()
        n_bodies = len(system.bodies)

        # First half-step velocity update (kick)

        # Calculate acceleration for all bodies
        accelerations = system.derivatives(state, t)

        # Update all velocities by half a time step
        for body_index in range(n_bodies):
            # Get indices for this body's velocity components
            vel_start_idx = body_index * 6 + 3
            vel_end_idx = vel_start_idx + 3

            # Get acceleration components for this body
            body_acceleration = accelerations[vel_start_idx:vel_end_idx]

            # Apply half-step acceleration (v += 0.5 * a * dt)
            state[vel_start_idx:vel_end_idx] += 0.5 * dt * body_acceleration

        # Full-step position update (drift)


        for body_index in range(n_bodies):
            # Get indices for position and velocity
            pos_start_idx = body_index * 6
            pos_end_idx = pos_start_idx + 3
            vel_start_idx = pos_end_idx
            vel_end_idx = vel_start_idx + 3

            # Get velocity for this body
            body_velocity = state[vel_start_idx:vel_end_idx]

            # Update position with full time step (x += v * dt)
            state[pos_start_idx:pos_end_idx] += dt * body_velocity

        # Second half-step velocity update (kick)

        # Calculate new accelerations at the updated positions and time
        accelerations = system.derivatives(state, t + dt)

        # Apply second half-step acceleration update
        for body_index in range(n_bodies):
            # Get indices for this body's velocity components
            vel_start_idx = body_index * 6 + 3
            vel_end_idx = vel_start_idx + 3

            # Get acceleration components for this body
            body_acceleration = accelerations[vel_start_idx:vel_end_idx]

            # Apply half-step acceleration (v += 0.5 * a * dt)
            state[vel_start_idx:vel_end_idx] += 0.5 * dt * body_acceleration

        return stateias15