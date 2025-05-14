import numpy as np

class ForestRuthIntegrator:
    def __init__(self, G, softening=1e-2):
        '''
        Initializing the forest ruth integrator
        '''

        self.G = G
        self.softening = softening  # Prevents division by zero

        # Forest-Ruth integrator coefficients
        self.theta = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
        self.c1 = self.c4 = self.theta / 2.0
        self.c2 = self.c3 = (1.0 - self.theta) / 2.0
        self.d1 = self.d3 = self.theta
        self.d2 = 1.0 - 2.0 * self.theta

        # Verify coefficients sum to expected values
        assert abs((self.c1 + self.c2 + self.c3 + self.c4) - 1.0) < 1e-12, "Drift coefficients don't sum to 1"
        assert abs((self.d1 + self.d2 + self.d3) - 1.0) < 1e-12, "Kick coefficients don't sum to 1"

    def compute_accelerations(self, bodies):
        '''
        Compute accelerations for all bodies.
        params:
            bodies:
        '''

        n = len(bodies)
        accelerations = [np.zeros(3) for j in range(n)]  # Fixed missing variable name

        # Compute all pairwise interactions once
        for i in range(n):
            for j in range(i + 1, n):  # Start from i+1 to avoid duplicate calculations
                r_vec = bodies[j].position - bodies[i].position  # Consistent variable naming
                distance_squared = np.sum(r_vec ** 2) + self.softening ** 2
                distance_cubed = distance_squared * np.sqrt(distance_squared)

                # Calculate force (acceleration) using Newton's law of gravitation
                force = self.G * r_vec / distance_cubed

                # Apply force to both bodies (Newton's third law)
                accelerations[i] += bodies[j].mass * force
                accelerations[j] -= bodies[i].mass * force  # Equal and opposite

        return accelerations

    def drift(self, bodies, coeff, dt):
        """
        Update positions based on current velocities.

        Parameters:
            bodies (list): List of body objects with position and velocity attributes
            coeff (float): Coefficient for the drift step
            dt (float): Time step size
        """

        dt_coeff = dt * coeff
        for body in bodies:
            body.position += body.velocity * dt_coeff

    def kick(self, bodies, accelerations, coeff, dt):
        """
        Update velocities based on calculated accelerations.

        Parameters:
            bodies (list): List of body objects with velocity attribute
            accelerations (list): List of acceleration vectors for each body
            coeff (float): Coefficient for the kick step
            dt (float): Time step size
        """
        dt_coeff = dt * coeff
        for i, body in enumerate(bodies):
            body.velocity += accelerations[i] * dt_coeff

    def step(self, bodies, dt):
        """
        Perform one complete integration step using the Forest-Ruth scheme.

        Parameters:
            bodies (list): List of body objects
            dt (float): Time step size

        Returns:
            list: Updated list of body objects
        """
        # First drift
        self.drift(bodies, self.c1, dt)

        # First kick
        accelerations = self.compute_accelerations(bodies)
        self.kick(bodies, accelerations, self.d1, dt)

        # Second drift
        self.drift(bodies, self.c2, dt)

        # Second kick
        accelerations = self.compute_accelerations(bodies)
        self.kick(bodies, accelerations, self.d2, dt)

        # Third drift
        self.drift(bodies, self.c3, dt)

        # Third kick
        accelerations = self.compute_accelerations(bodies)
        self.kick(bodies, accelerations, self.d3, dt)

        # Fourth drift
        self.drift(bodies, self.c4, dt)

        

        return bodies

