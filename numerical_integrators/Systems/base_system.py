import numpy as np

class System:
    """
    Base class for all dynamical systems.
    """
    def __init__(self, name):
        """
        Initialize the system.

        Args:
            name: A string identifier for the system
        """
        self.name = name
        self.state = None  # Will be initialized by subclasses

    def derivatives(self, state, t):
        """
        Compute the derivatives of the state variables.

        For a mechanical system, this would compute velocities and accelerations.

        Args:
            state: Current state vector
            t: Current time

        Returns:
            Array of derivatives [dx1/dt, dx2/dt, ...]
        """
        raise NotImplementedError("Subclasses must implement derivatives method")

    def energy(self):
        """
        Calculate the total energy of the system.

        Returns:
            Float representing the total energy
        """
        raise NotImplementedError("Subclasses must implement energy method")

    def angular_momentum(self):
        """
        Calculate the total angular momentum of the system.

        Returns:
            Array representing the angular momentum vector
        """
        raise NotImplementedError("Subclasses must implement angular_momentum method")

    def linear_momentum(self):
        """
        Calculate the total linear momentum of the system.

        Returns:
            Array representing the momentum vector
        """
        raise NotImplementedError("Subclasses must implement linear_momentum method")