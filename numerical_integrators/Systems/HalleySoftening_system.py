'''
Similar to Halleys commet but with softening term so the commet doesnt fling off
into space in RK4 integrator. Modified to start Halley at aphelion.
'''

import numpy as np
from base_system import System


class Softening(System):
    """
    Solar system model with the Sun and planets up to Neptune.
    """

    def __init__(self):
        super().__init__("Solar System")

        # G=4π² in units where:
            # - 1 AU is the unit of length
            # - 1 solar mass is the unit of mass
            # - 1 year is the unit of time

        self.G = 4 * np.pi ** 2
        self.epsilon = 0.2

        self.bodies = self._create_planets()
        self._initialize_state()

    def _create_planets(self):
        bodies = []

        # Sun at the origin
        bodies.append({
            'name': 'Sun',
            'mass': 1.0,
            'position': np.array([0.0, 0.0, 0.0]),
            'velocity': np.array([0.0, 0.0, 0.0])
        })

        # Planet data: [name, mass/solar mass, semi-major axis/AU, eccentricity]
        planet_data = [
            ['Mercury', 1.6601e-7, 0.387, 0.2056],
            ['Venus', 2.4478e-6, 0.723, 0.0068],
            ['Earth', 3.0034e-6, 1.0, 0.0167],
            ['Mars', 3.2271e-7, 1.524, 0.0934],
            ['Jupiter', 9.5479e-4, 5.203, 0.0484],
            ['Saturn', 2.8577e-4, 9.537, 0.0539],
            ['Uranus', 4.3657e-5, 19.191, 0.0473],
            ['Neptune', 5.1503e-5, 30.069, 0.0086],
            ['Halley', 1.1e-16,    17.8,   0.967]  # Halley's Comet
        ]

        # Add each planet
        for name, mass, a, e in planet_data:
            if name == 'Halley':
                # For Halley's Comet, start at aphelion (furthest from Sun)
                r_apo = a * (1 + e)  # Distance at aphelion

                # Velocity at aphelion is slower than at perihelion
                # v_apo = sqrt(μ*(1-e)/(a*(1+e)))
                v_apo = np.sqrt(self.G * (1 - e) / (a * (1 + e)))

                bodies.append({
                    'name': name,
                    'mass': mass,
                    'position': np.array([-r_apo, 0.0, 0.0]),  # Place it on negative x-axis
                    'velocity': np.array([0.0, -v_apo, 0.0])   # Velocity perpendicular to position
                })
            else:
                # Other planets at perihelion as before
                r_peri = a * (1 - e)
                v_peri = np.sqrt(self.G * (1 + e) / (a * (1 - e)))

                bodies.append({
                    'name': name,
                    'mass': mass,
                    'position': np.array([r_peri, 0.0, 0.0]),
                    'velocity': np.array([0.0, v_peri, 0.0])
                })

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

                    # Newton's law of gravitation with softening
                    acceleration += self.G * mass_j * r_vec / ((r_mag**2 + self.epsilon**2)**1.5)

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