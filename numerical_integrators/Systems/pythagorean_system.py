import numpy as np
from base_system import System


class PythagoreanSystem(System):
    """
    Three-body problem simulation with Pythagorean initial conditions.

    This implements a three-body system where masses are placed at the
    corners of a right triangle with side lengths 3, 4, 5 (a Pythagorean triple).
    """

    def __init__(self):
        super().__init__("Pythagorean Three-Body")

        # Gravitational constant
        self.G = 1.0

        self.bodies = self._create_bodies()
        self._initialize_state()

    def _create_bodies(self):
        bodies = []

        # Pythagorean bodies at corners of a 3-4-5 right triangle
        bodies.append({
            'name': 'Body 1',
            'mass': 3.0,
            'position': np.array([1.0, 1.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0])  # Will be adjusted for momentum conservation
        })

        bodies.append({
            'name': 'Body 2',
            'mass': 4.0,
            'position': np.array([4.0, 1.0, 0.0]),  # 3 units to the right of Body 1
            'velocity': np.array([0.0, 0.0, 0.0])  # Initial velocity perpendicular to side
        })

        bodies.append({
            'name': 'Body 3',
            'mass': 5.0,
            'position': np.array([1.0, 5.0, 0.0]),  # 4 units above Body 1
            'velocity': np.array([0, 0.0, 0.0])  # Initial velocity perpendicular to side
        })

        # Calculate velocity for Body 1 to ensure zero total momentum
        m1 = bodies[0]['mass']
        m2 = bodies[1]['mass']
        m3 = bodies[2]['mass']
        v2 = bodies[1]['velocity']
        v3 = bodies[2]['velocity']

        bodies[0]['velocity'] = -(m2 * v2 + m3 * v3) / m1

        return bodies

    def _initialize_state(self):
        """Initialize state vector from body data."""
        n_bodies = len(self.bodies)
        self.state = np.zeros(n_bodies * 6)

        for i, body in enumerate(self.bodies):
            # Position
            self.state[i * 6:i * 6 + 3] = body['position']
            # Velocity
            self.state[i * 6 + 3:i * 6 + 6] = body['velocity']

    def derivatives(self, state, t):
        """Compute derivatives."""
        n_bodies = len(self.bodies)
        derivatives = np.zeros_like(state)

        # Position derivatives = velocities
        for i in range(n_bodies):
            derivatives[i * 6:i * 6 + 3] = state[i * 6 + 3:i * 6 + 6]

        # Velocity derivatives = accelerations
        for i in range(n_bodies):
            pos_i = state[i * 6:i * 6 + 3]

            # Sum forces from all other bodies
            acceleration = np.zeros(3)
            for j in range(n_bodies):
                if i != j:  # Skip self-interaction
                    pos_j = state[j * 6:j * 6 + 3]
                    mass_j = self.bodies[j]['mass']

                    # Vector from body i to body j
                    r_vec = pos_j - pos_i
                    r_mag = np.linalg.norm(r_vec)

                    # Small softening parameter to prevent numerical instability
                    epsilon = 1e-6

                    # Newton's law of gravitation with softening
                    acceleration += self.G * mass_j * r_vec / ((r_mag ** 2 + epsilon ** 2) ** 1.5)

            derivatives[i * 6 + 3:i * 6 + 6] = acceleration

        return derivatives

    def energy(self):
        """Calculate the total energy of the system."""
        n_bodies = len(self.bodies)

        # Update body data
        for i, body in enumerate(self.bodies):
            body['position'] = self.state[i * 6:i * 6 + 3]
            body['velocity'] = self.state[i * 6 + 3:i * 6 + 6]

        # Kinetic energy: Σ½mv²
        kinetic = 0.0
        for body in self.bodies:
            v_squared = np.sum(body['velocity'] ** 2)
            kinetic += 0.5 * body['mass'] * v_squared

        # Potential energy: Σ-G*m₁*m₂/r
        potential = 0.0
        for i in range(n_bodies):
            for j in range(i + 1, n_bodies):
                pos_i = self.bodies[i]['position']
                pos_j = self.bodies[j]['position']
                mass_i = self.bodies[i]['mass']
                mass_j = self.bodies[j]['mass']

                r = np.linalg.norm(pos_j - pos_i)
                potential -= self.G * mass_i * mass_j / r

        return kinetic + potential

    def angular_momentum(self):
        """Calculate the total angular momentum of the system."""
        L = np.zeros(3)

        for i, body in enumerate(self.bodies):
            body['position'] = self.state[i * 6:i * 6 + 3]
            body['velocity'] = self.state[i * 6 + 3:i * 6 + 6]

            L += body['mass'] * np.cross(body['position'], body['velocity'])

        return L

    def linear_momentum(self):
        """Calculate the total linear momentum of the system."""
        P = np.zeros(3)

        for i, body in enumerate(self.bodies):
            body['position'] = self.state[i * 6:i * 6 + 3]
            body['velocity'] = self.state[i * 6 + 3:i * 6 + 6]

            P += body['mass'] * body['velocity']

        return P