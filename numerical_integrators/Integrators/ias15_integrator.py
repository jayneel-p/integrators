import numpy as np
from base_integrator import Integrator


class IAS15(Integrator):
    """
    IAS15 Integrator - 15th order implicit integrator.
    Based on Rein & Spiegel (2015) - "IAS15: A fast, adaptive, high-order integrator
    for gravitational dynamics, accurate to machine precision over a billion orbits"
    """

    def __init__(self):
        """Initialize the IAS15 integrator."""
        super().__init__("IAS15")

        # Add these flags for step rejection handling
        self.rejected_step = False
        self.suggested_dt = None

        # Precision parameter controlling adaptive timestepping
        self.epsilon = 1e-9  # Default value, balance between speed and accuracy
        self.min_dt = 1e-12  # Minimum allowed timestep
        self.safety_factor = 0.25  # Maximum increase/decrease of consecutive timesteps

        # For tracking iterations and statistics
        self.max_iterations = 12
        self.iterations_max_exceeded = 0

        # Last completed timestep (for prediction)
        self.dt_last_done = 0

        # Initialize Gauss-Radau spacings
        self.h = np.array([
            0.0, 0.0562625605369221464656521910318, 0.180240691736892364987579942780,
            0.352624717113169637373907769648, 0.547153626330555383001448554766,
            0.734210177215410531523210605558, 0.885320946839095768090359771030,
            0.977520613561287501891174488626
        ])

        # Precomputed constants for the algorithm
        self.rr = np.array([
            0.0562625605369221464656522, 0.1802406917368923649875799, 0.1239781311999702185219278,
            0.3526247171131696373739078, 0.2963621565762474909082556, 0.1723840253762772723863278,
            0.5471536263305553830014486, 0.4908910657936332365357964, 0.3669129345936630180138686,
            0.1945289092173857456275408, 0.7342101772154105315232106, 0.6779476166784883850575584,
            0.5539694854785181665356307, 0.3815854601022408941493028, 0.1870565508848551485217621,
            0.8853209468390957680903598, 0.8290583863021736216247076, 0.7050802551022034031027798,
            0.5326962297259261307164520, 0.3381673205085403850889112, 0.1511107696236852365671492,
            0.9775206135612875018911745, 0.9212580530243653554255223, 0.7972799218243951369035945,
            0.6248958964481178645172667, 0.4303669872307321188897259, 0.2433104363458769703679639,
            0.0921996667221917338008147
        ])

        self.c = np.array([
            -0.0562625605369221464656522, 0.0101408028300636299864818, -0.2365032522738145114532321,
            -0.0035758977292516175949345, 0.0935376952594620658957485, -0.5891279693869841488271399,
            0.0019565654099472210769006, -0.0547553868890686864408084, 0.4158812000823068616886219,
            -1.1362815957175395318285885, -0.0014365302363708915424460, 0.0421585277212687077072973,
            -0.3600995965020568122897665, 1.2501507118406910258505441, -1.8704917729329500633517991,
            0.0012717903090268677492943, -0.0387603579159067703699046, 0.3609622434528459832253398,
            -1.4668842084004269643701553, 2.9061362593084293014237913, -2.7558127197720458314421588
        ])

        self.d = np.array([
            0.0562625605369221464656522, 0.0031654757181708292499905, 0.2365032522738145114532321,
            0.0000178097769221743388112, 0.0457929855060279188954539, 0.5891279693869841488271399,
            0.0000100202365223291272096, 0.0084318571535257015445000, 0.2535340690545692665214616,
            1.1362815957175395318285885, 0.0000005637641639318207610, 0.0015297840025004658189490,
            0.0978342365324440053653648, 0.8752546646840910912297246, 1.8704917729329500633517991,
            0.0000000317188154017613665, 0.0002762930909826476593130, 0.0360285539837364596003871,
            0.5767330002770787313544596, 2.2485887607691597933926895, 2.7558127197720458314421588
        ])

        # Initialize coefficient arrays
        self.g = None  # Intermediate coefficients
        self.b = None  # Coefficients for final state
        self.e = None  # Predicted coefficients for next step
        self.csx = None  # Compensated summation for position
        self.csv = None  # Compensated summation for velocity
        self.csb = None  # Compensated summation for b coefficients

    def add_compensated(self, value, compensation, term):
        """Add with compensated summation to reduce floating-point errors."""
        y = term - compensation
        t = value + y
        compensation = (t - value) - y
        return t, compensation

    def step(self, system, t, dt):
        """
        Take a single IAS15 step.

        Args:
            system: System object with state and derivatives methods
            t: Current time
            dt: Time step size

        Returns:
            Updated state vector
        """
        # Get current state and initialize arrays
        current_state = system.state.copy()
        n_bodies = len(system.bodies)
        n_state = len(current_state)

        # Initialize arrays if needed or if they need to be resized
        if self.g is None or len(self.g) != 7 or self.g[0].shape[0] != n_state:
            self.g = [np.zeros(n_state) for _ in range(7)]
            self.b = [np.zeros(n_state) for _ in range(7)]
            self.e = [np.zeros(n_state) for _ in range(7)]
            self.csb = [np.zeros(n_state) for _ in range(7)]
            self.csx = np.zeros(n_state)
            self.csv = np.zeros(n_state)

        # Extract positions, velocities, and initial accelerations
        x0 = np.zeros(n_state)
        v0 = np.zeros(n_state)
        a0 = np.zeros(n_state)

        for i in range(n_bodies):
            # Position
            x0[i * 6:i * 6 + 3] = current_state[i * 6:i * 6 + 3]
            # Velocity
            v0[i * 6 + 3:i * 6 + 6] = current_state[i * 6 + 3:i * 6 + 6]

        # Get initial accelerations
        derivatives = system.derivatives(current_state, t)
        for i in range(n_bodies):
            a0[i * 6 + 3:i * 6 + 6] = derivatives[i * 6 + 3:i * 6 + 6]

        # Calculate initial g values from b coefficients (if they exist from previous step)
        for k in range(n_state):
            self.g[0][k] = self.b[6][k] * self.d[15] + self.b[5][k] * self.d[10] + self.b[4][k] * self.d[6] + \
                           self.b[3][k] * self.d[3] + self.b[2][k] * self.d[1] + self.b[1][k] * self.d[0] + self.b[0][k]
            self.g[1][k] = self.b[6][k] * self.d[16] + self.b[5][k] * self.d[11] + self.b[4][k] * self.d[7] + \
                           self.b[3][k] * self.d[4] + self.b[2][k] * self.d[2] + self.b[1][k]
            self.g[2][k] = self.b[6][k] * self.d[17] + self.b[5][k] * self.d[12] + self.b[4][k] * self.d[8] + \
                           self.b[3][k] * self.d[5] + self.b[2][k]
            self.g[3][k] = self.b[6][k] * self.d[18] + self.b[5][k] * self.d[13] + self.b[4][k] * self.d[9] + self.b[3][
                k]
            self.g[4][k] = self.b[6][k] * self.d[19] + self.b[5][k] * self.d[14] + self.b[4][k]
            self.g[5][k] = self.b[6][k] * self.d[20] + self.b[5][k]
            self.g[6][k] = self.b[6][k]

        # Predictor corrector loop
        predictor_corrector_error = 1e300
        predictor_corrector_error_last = 2
        iterations = 0

        while True:
            if predictor_corrector_error < 1e-16:
                break
            if iterations > 2 and predictor_corrector_error_last <= predictor_corrector_error:
                break
            if iterations >= self.max_iterations:
                self.iterations_max_exceeded += 1
                if self.iterations_max_exceeded == 10:
                    print(
                        f"Warning: IAS15 predictor corrector loop did not converge. This is typically an indication of the timestep being too large.")
                break

            predictor_corrector_error_last = predictor_corrector_error
            predictor_corrector_error = 0
            iterations += 1

            # Loop over interval using Gauss-Radau spacings
            for n in range(1, 8):
                current_t = t + dt * self.h[n]

                # Prepare state for force calculation at substep n
                substep_state = current_state.copy()

                # Predict positions at interval n using b values
                for i in range(n_bodies):
                    for k in range(3):  # x, y, z coordinates
                        pos_idx = i * 6 + k
                        vel_idx = i * 6 + k + 3

                        # Calculate position increment with compensated summation
                        pos_increment = dt * self.h[n] * v0[vel_idx]  # First-order term

                        # Add acceleration term
                        pos_increment += 0.5 * dt ** 2 * self.h[n] ** 2 * a0[vel_idx]

                        # Add higher-order terms from b coefficients
                        for j in range(7):
                            term = self.b[j][vel_idx] * self.h[n] ** (j + 3) * dt ** 2 / ((j + 1) * (j + 2))
                            pos_increment += term

                        # Apply position increment with compensated summation
                        substep_state[pos_idx], self.csx[pos_idx] = self.add_compensated(
                            substep_state[pos_idx], self.csx[pos_idx], pos_increment
                        )

                    # Predict velocities at interval n using b values
                    for k in range(3):  # x, y, z coordinates
                        vel_idx = i * 6 + k + 3

                        # Calculate velocity increment
                        vel_increment = dt * self.h[n] * a0[vel_idx]  # First-order term

                        # Add higher-order terms
                        for j in range(7):
                            term = self.b[j][vel_idx] * self.h[n] ** (j + 2) * dt / (j + 1)
                            vel_increment += term

                        # Apply velocity increment with compensated summation
                        substep_state[vel_idx], self.csv[vel_idx] = self.add_compensated(
                            substep_state[vel_idx], self.csv[vel_idx], vel_increment
                        )

                # Calculate forces at this substep
                substep_derivatives = system.derivatives(substep_state, current_t)

                # Extract accelerations
                at = np.zeros(n_state)
                for i in range(n_bodies):
                    at[i * 6 + 3:i * 6 + 6] = substep_derivatives[i * 6 + 3:i * 6 + 6]

                # Update g and b coefficients based on the forces
                if n == 1:
                    for k in range(n_state):
                        if k % 6 >= 3:  # Only for velocity components (accelerations)
                            tmp = self.g[0][k]
                            self.g[0][k] = (at[k] - a0[k]) / self.rr[0]
                            self.b[0][k], self.csb[0][k] = self.add_compensated(
                                self.b[0][k], self.csb[0][k], self.g[0][k] - tmp
                            )

                elif n == 2:
                    for k in range(n_state):
                        if k % 6 >= 3:  # Only for velocity components
                            tmp = self.g[1][k]
                            gk = (at[k] - a0[k]) / self.rr[1] - self.g[0][k]
                            self.g[1][k] = gk / self.rr[2]
                            tmp = self.g[1][k] - tmp
                            self.b[0][k], self.csb[0][k] = self.add_compensated(
                                self.b[0][k], self.csb[0][k], tmp * self.c[0]
                            )
                            self.b[1][k], self.csb[1][k] = self.add_compensated(
                                self.b[1][k], self.csb[1][k], tmp
                            )

                elif n == 3:
                    for k in range(n_state):
                        if k % 6 >= 3:  # Only for velocity components
                            tmp = self.g[2][k]
                            gk = ((at[k] - a0[k]) / self.rr[3] - self.g[0][k]) / self.rr[4] - self.g[1][k]
                            self.g[2][k] = gk / self.rr[5]
                            tmp = self.g[2][k] - tmp
                            self.b[0][k], self.csb[0][k] = self.add_compensated(
                                self.b[0][k], self.csb[0][k], tmp * self.c[1]
                            )
                            self.b[1][k], self.csb[1][k] = self.add_compensated(
                                self.b[1][k], self.csb[1][k], tmp * self.c[2]
                            )
                            self.b[2][k], self.csb[2][k] = self.add_compensated(
                                self.b[2][k], self.csb[2][k], tmp
                            )

                elif n == 4:
                    for k in range(n_state):
                        if k % 6 >= 3:  # Only for velocity components
                            tmp = self.g[3][k]
                            gk = (((at[k] - a0[k]) / self.rr[6] - self.g[0][k]) / self.rr[7] - self.g[1][k]) / self.rr[
                                8] - self.g[2][k]
                            self.g[3][k] = gk / self.rr[9]
                            tmp = self.g[3][k] - tmp
                            self.b[0][k], self.csb[0][k] = self.add_compensated(
                                self.b[0][k], self.csb[0][k], tmp * self.c[3]
                            )
                            self.b[1][k], self.csb[1][k] = self.add_compensated(
                                self.b[1][k], self.csb[1][k], tmp * self.c[4]
                            )
                            self.b[2][k], self.csb[2][k] = self.add_compensated(
                                self.b[2][k], self.csb[2][k], tmp * self.c[5]
                            )
                            self.b[3][k], self.csb[3][k] = self.add_compensated(
                                self.b[3][k], self.csb[3][k], tmp
                            )

                elif n == 5:
                    for k in range(n_state):
                        if k % 6 >= 3:  # Only for velocity components
                            tmp = self.g[4][k]
                            gk = ((((at[k] - a0[k]) / self.rr[10] - self.g[0][k]) / self.rr[11] - self.g[1][k]) /
                                  self.rr[12] - self.g[2][k]) / self.rr[13] - self.g[3][k]
                            self.g[4][k] = gk / self.rr[14]
                            tmp = self.g[4][k] - tmp
                            self.b[0][k], self.csb[0][k] = self.add_compensated(
                                self.b[0][k], self.csb[0][k], tmp * self.c[6]
                            )
                            self.b[1][k], self.csb[1][k] = self.add_compensated(
                                self.b[1][k], self.csb[1][k], tmp * self.c[7]
                            )
                            self.b[2][k], self.csb[2][k] = self.add_compensated(
                                self.b[2][k], self.csb[2][k], tmp * self.c[8]
                            )
                            self.b[3][k], self.csb[3][k] = self.add_compensated(
                                self.b[3][k], self.csb[3][k], tmp * self.c[9]
                            )
                            self.b[4][k], self.csb[4][k] = self.add_compensated(
                                self.b[4][k], self.csb[4][k], tmp
                            )

                elif n == 6:
                    for k in range(n_state):
                        if k % 6 >= 3:  # Only for velocity components
                            tmp = self.g[5][k]
                            gk = (((((at[k] - a0[k]) / self.rr[15] - self.g[0][k]) / self.rr[16] - self.g[1][k]) /
                                   self.rr[17] - self.g[2][k]) / self.rr[18] - self.g[3][k]) / self.rr[19] - self.g[4][
                                     k]
                            self.g[5][k] = gk / self.rr[20]
                            tmp = self.g[5][k] - tmp
                            self.b[0][k], self.csb[0][k] = self.add_compensated(
                                self.b[0][k], self.csb[0][k], tmp * self.c[10]
                            )
                            self.b[1][k], self.csb[1][k] = self.add_compensated(
                                self.b[1][k], self.csb[1][k], tmp * self.c[11]
                            )
                            self.b[2][k], self.csb[2][k] = self.add_compensated(
                                self.b[2][k], self.csb[2][k], tmp * self.c[12]
                            )
                            self.b[3][k], self.csb[3][k] = self.add_compensated(
                                self.b[3][k], self.csb[3][k], tmp * self.c[13]
                            )
                            self.b[4][k], self.csb[4][k] = self.add_compensated(
                                self.b[4][k], self.csb[4][k], tmp * self.c[14]
                            )
                            self.b[5][k], self.csb[5][k] = self.add_compensated(
                                self.b[5][k], self.csb[5][k], tmp
                            )

                elif n == 7:
                    max_a = 0.0
                    max_b6_change = 0.0

                    for k in range(n_state):
                        if k % 6 >= 3:  # Only for velocity components
                            tmp = self.g[6][k]
                            gk = ((((((at[k] - a0[k]) / self.rr[21] - self.g[0][k]) / self.rr[22] - self.g[1][k]) /
                                    self.rr[23] - self.g[2][k]) / self.rr[24] - self.g[3][k]) / self.rr[25] - self.g[4][
                                      k]) / self.rr[26] - self.g[5][k]
                            self.g[6][k] = gk / self.rr[27]
                            tmp = self.g[6][k] - tmp

                            self.b[0][k], self.csb[0][k] = self.add_compensated(
                                self.b[0][k], self.csb[0][k], tmp * self.c[15]
                            )
                            self.b[1][k], self.csb[1][k] = self.add_compensated(
                                self.b[1][k], self.csb[1][k], tmp * self.c[16]
                            )
                            self.b[2][k], self.csb[2][k] = self.add_compensated(
                                self.b[2][k], self.csb[2][k], tmp * self.c[17]
                            )
                            self.b[3][k], self.csb[3][k] = self.add_compensated(
                                self.b[3][k], self.csb[3][k], tmp * self.c[18]
                            )
                            self.b[4][k], self.csb[4][k] = self.add_compensated(
                                self.b[4][k], self.csb[4][k], tmp * self.c[19]
                            )
                            self.b[5][k], self.csb[5][k] = self.add_compensated(
                                self.b[5][k], self.csb[5][k], tmp * self.c[20]
                            )
                            self.b[6][k], self.csb[6][k] = self.add_compensated(
                                self.b[6][k], self.csb[6][k], tmp
                            )

                            # Calculate error estimate for predictor-corrector convergence
                            ak = abs(at[k])
                            b6k_change = abs(tmp)

                            if np.isfinite(ak) and ak > max_a:
                                max_a = ak
                            if np.isfinite(b6k_change) and b6k_change > max_b6_change:
                                max_b6_change = b6k_change

                    # Update predictor-corrector error estimate
                    if max_a > 0:
                        predictor_corrector_error = max_b6_change / max_a

        # Determine if we need to adjust the timestep
        dt_new = dt
        if self.epsilon > 0:
            # Estimate error (given by the magnitude of the last term)
            max_a = 0.0
            max_b6 = 0.0

            for k in range(n_state):
                if k % 6 >= 3:  # Only for velocity components
                    ak = abs(a0[k])
                    if np.isfinite(ak) and ak > max_a:
                        max_a = ak
                    b6k = abs(self.b[6][k])
                    if np.isfinite(b6k) and b6k > max_b6:
                        max_b6 = b6k

            integrator_error = max_b6 / max_a if max_a > 0 else 0

            # Use error estimate to calculate new timestep
            if np.isfinite(integrator_error) and integrator_error > 0:
                # The 1/7 power is because the error scales with dt^7 (15th order scheme)
                dt_new = dt * (self.epsilon / integrator_error) ** (1 / 7)
            else:
                # If error estimate isn't usable, increase timestep slightly
                dt_new = dt / self.safety_factor

            # Ensure timestep doesn't change too abruptly
            if dt_new / dt < self.safety_factor:
                # Instead of returning None, set a flag on the integrator
                # and return the original state (unchanged)
                self.rejected_step = True
                self.suggested_dt = dt_new
                return current_state  # Return original state unchanged

            if dt_new / dt > 1.0:
                # New timestep is larger than current one
                if dt_new / dt > 1.0 / self.safety_factor:
                    # Don't increase timestep too much
                    dt_new = dt / self.safety_factor

        # Finally, calculate new positions and velocities
        new_state = current_state.copy()

        for i in range(n_bodies):
            for k in range(3):
                pos_idx = i * 6 + k
                vel_idx = i * 6 + k + 3

                # Update position
                pos_increment = dt * v0[vel_idx] + 0.5 * dt ** 2 * a0[vel_idx]
                for j in range(7):
                    pos_increment += dt ** 2 * self.b[j][vel_idx] / ((j + 1) * (j + 2))

                new_state[pos_idx], self.csx[pos_idx] = self.add_compensated(
                    new_state[pos_idx], self.csx[pos_idx], pos_increment
                )

                # Update velocity
                vel_increment = dt * a0[vel_idx]
                for j in range(7):
                    vel_increment += dt * self.b[j][vel_idx] / (j + 1)

                new_state[vel_idx], self.csv[vel_idx] = self.add_compensated(
                    new_state[vel_idx], self.csv[vel_idx], vel_increment
                )

        # Store information for next step
        self.dt_last_done = dt

        # Predict b coefficients for next step
        if dt_new != dt:
            self.predict_next_step(dt_new / dt)

        self.rejected_step = False

        return new_state

    def predict_next_step(self, ratio):
        """
        Predict b values for next step based on current values and timestep ratio.

        Args:
            ratio: Ratio of new timestep to old timestep
        """
        # Don't predict if timestep increase is very large
        if ratio > 20.0:
            for j in range(7):
                self.e[j][:] = 0.0
                self.b[j][:] = 0.0
            return

        # Calculate powers of the ratio
        q1 = ratio
        q2 = q1 * q1
        q3 = q1 * q2
        q4 = q2 * q2
        q5 = q2 * q3
        q6 = q3 * q3
        q7 = q3 * q4

        # Save current b values
        be = [[0.0 for _ in range(len(self.b[0]))] for _ in range(7)]
        for j in range(7):
            for k in range(len(self.b[0])):
                be[j][k] = self.b[j][k] - self.e[j][k]

        # Predict new values
        for k in range(len(self.b[0])):
            self.e[0][k] = q1 * (self.b[6][k] * 7.0 + self.b[5][k] * 6.0 + self.b[4][k] * 5.0 +
                                 self.b[3][k] * 4.0 + self.b[2][k] * 3.0 + self.b[1][k] * 2.0 + self.b[0][k])
            self.e[1][k] = q2 * (self.b[6][k] * 21.0 + self.b[5][k] * 15.0 + self.b[4][k] * 10.0 +
                                 self.b[3][k] * 6.0 + self.b[2][k] * 3.0 + self.b[1][k])
            self.e[2][k] = q3 * (self.b[6][k] * 35.0 + self.b[5][k] * 20.0 + self.b[4][k] * 10.0 +
                                 self.b[3][k] * 4.0 + self.b[2][k])
            self.e[3][k] = q4 * (self.b[6][k] * 35.0 + self.b[5][k] * 15.0 + self.b[4][k] * 5.0 + self.b[3][k])
            self.e[4][k] = q5 * (self.b[6][k] * 21.0 + self.b[5][k] * 6.0 + self.b[4][k])
            self.e[5][k] = q6 * (self.b[6][k] * 7.0 + self.b[5][k])
            self.e[6][k] = q7 * self.b[6][k]

            # Apply correction
            for j in range(7):
                self.b[j][k] = self.e[j][k] + be[j][k]

    def sqrt7(self, a):
        """
        Machine-independent implementation of pow(a, 1/7).

        Args:
            a: Input value

        Returns:
            Seventh root of a
        """
        if a <= 0:
            return 0.0

        # Scale to avoid numerical issues
        scale = 1.0
        while a < 1e-7 and np.isfinite(a):
            scale *= 0.1
            a *= 1e7
        while a > 1e2 and np.isfinite(a):
            scale *= 10.0
            a *= 1e-7

        # Newton's method to find 7th root
        x = 1.0
        for _ in range(20):
            x6 = x ** 6
            x += (a / x6 - x) / 7.0

        return x * scale