import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math
import random
random.seed(42)


class PolarQuantumCircuit:
    def __init__(self, num_qubits: int, initial_state=None):
        self._num_qubits = num_qubits
        self._state_size = 2**num_qubits

        if initial_state is None:
            self._state = [(1.0 if i == 0 else 0.0, 0.0)
                           for i in range(self._state_size)]
        else:
            if len(initial_state) != self._state_size:
                raise ValueError(
                    "initial_state size does not match nun_qubits!")
            self._state = initial_state

        self._applied_gates = []

    def get_circuit_info(self):
        """
        Returns information about the quantum circuit, including the number of qubits
        and the gates applied so far.
        """
        return {"num_qubits": self._num_qubits, "gates_applied": self._applied_gates}

    def reset(self):
        """
        Resets the quantum circuit to the |0> state.
        """
        self._state = [(1.0 if i == 0 else 0.0, 0.0)
                       for i in range(self._state_size)]

    def get_state(self):
        """
        Returns the current state of the quantum circuit.
        """
        return self._state.copy()

    def _validate_qubit_index(self, qubit):
        if not (0 <= qubit < self._num_qubits):
            raise ValueError(
                f"Qubit {qubit} is out of bounds for this {self._num_qubits}-qubit circuit!")

    def _round_state(self):
        """
        Rounds small numerical values in the state to improve numerical stability.
        """
        threshold = 1e-6

        for i in range(len(self._state)):
            r, theta = self._state[i]
            if abs(r) < threshold:
                r = 0.0
            if abs(r - 1.0) < threshold:
                r = 1.0
            if theta < -np.pi:
                if abs(theta - (-np.pi)) > threshold:
                    theta += 2 * np.pi
                else:
                    theta = -np.pi
            if theta >= np.pi:
                if abs(theta - np.pi) > threshold:
                    theta -= 2 * np.pi
                else:
                    theta = -np.pi
            self._state[i] = (r, theta)

    @staticmethod
    def polar2cartesian(polar_states):
        """
        Translate the polar coordinate (r, theta) into Cartesian coordinate (a + bi).
        """
        cartesian_states = [
            complex(r * math.cos(theta), r * math.sin(theta)) for r, theta in polar_states]
        return cartesian_states

    @staticmethod
    def cartesian2polar(cartesian_states):
        """
        Translate the Cartesian coordinate (a + bi) into polar coordinate (r, theta) .
        """
        polar_states = []
        for complexelement in cartesian_states:
            r = math.sqrt(complexelement.real**2 + complexelement.imag**2)
            theta = math.atan2(complexelement.imag, complexelement.real)
            if abs(theta - math.pi) < 1e-6:
                theta = -math.pi
            polar_states.append((r, theta))

        return polar_states

    def NOT_polar(self, target_qubit):
        """
        Apply the NOT gate to the target_qubit.
        """
        self._validate_qubit_index(target_qubit)
        self._applied_gates.append({"gate": "NOT", "target": target_qubit})

        new_state = self._state.copy()

        for i in range(self._state_size):
            # Check if the target qubit is 0 or 1
            if (i >> target_qubit) & 1 == 0:
                index_0 = i
                index_1 = i | (1 << target_qubit)
            else:
                index_0 = i & ~(1 << target_qubit)
                index_1 = i

            # Update the new state
            new_state[index_0] = self._state[index_1]
            new_state[index_1] = self._state[index_0]

        self._state = new_state.copy()
        self._round_state()

    def CNOT_polar(self, control_qubit, target_qubit, log=True):
        """
        Apply the CNOT gate to the target_qubit only if the control_qubit is 1.
        """
        self._validate_qubit_index(target_qubit)
        if log:
            self._applied_gates.append(
                {"gate": "CNOT", "control": control_qubit, "target": target_qubit})

        new_state = self._state.copy()

        for i in range(self._state_size):
            # Check if the target qubit is 0 or 1
            if (i >> control_qubit) & 1 != 0:
                if (i >> target_qubit) & 1 == 0:
                    index_0 = i
                    index_1 = i | (1 << target_qubit)
                else:
                    index_0 = i & ~(1 << target_qubit)
                    index_1 = i

                # Update the new state
                new_state[index_0] = self._state[index_1]
                new_state[index_1] = self._state[index_0]

        self._state = new_state.copy()
        self._round_state()

    def MCX_polar(self, control_qubits, target_qubit):
        """
        Apply the MCX gate to the target_qubit only if control_qubits are 1s.
        """
        for i in control_qubits:
            self._validate_qubit_index(i)
        self._validate_qubit_index(target_qubit)
        self._applied_gates.append(
            {"gate": "MCX", "control": control_qubits, "target": target_qubit})

        new_state = self._state.copy()

        for i in range(self._state_size):
            # Check if the target qubit is 0 or 1
            flag = True
            for j in control_qubits:
                if (i >> j) & 1 == 0:
                    flag = False
            if flag:
                if (i >> target_qubit) & 1 == 0:
                    index_0 = i
                    index_1 = i | (1 << target_qubit)
                else:
                    index_0 = i & ~(1 << target_qubit)
                    index_1 = i

                # Update the new state
                new_state[index_0] = self._state[index_1]
                new_state[index_1] = self._state[index_0]

        self._state = new_state.copy()
        self._round_state()

    def hadamard_polar(self, target_qubit):
        """
        Apply the hadamard gate to the target_qubit.
        """
        self._validate_qubit_index(target_qubit)
        self._applied_gates.append(
            {"gate": "HADAMARD", "target": target_qubit})

        new_state = self._state.copy()

        for i in range(self._state_size):
            # Check if the target qubit is 0 or 1
            if (i >> target_qubit) & 1 == 0:
                index_0 = i
                index_1 = i | (1 << target_qubit)
            else:
                index_0 = i & ~(1 << target_qubit)
                index_1 = i

            # Get the polar components for |0> and |1> of the target qubit
            r0, theta0 = self._state[index_0]
            r1, theta1 = self._state[index_1]

            # calculate by the rotating property of complex number
            delta_theta = theta1 - theta0
            r_new_0 = np.sqrt(
                0.5 * ((r0 + r1 * np.cos(delta_theta)) ** 2 + (r1 * np.sin(delta_theta)) ** 2))
            theta_new_0 = theta0 + \
                np.arctan2(r1 * np.sin(delta_theta),
                           r0 + r1 * np.cos(delta_theta))

            r_new_1 = np.sqrt(
                0.5 * ((r0 - r1 * np.cos(delta_theta)) ** 2 + (r1 * np.sin(delta_theta)) ** 2))
            theta_new_1 = theta0 + \
                np.arctan2(- r1 * np.sin(delta_theta),
                           r0 - r1 * np.cos(delta_theta))

            # Update the new state
            new_state[index_0] = [r_new_0, theta_new_0]
            new_state[index_1] = [r_new_1, theta_new_1]

        self._state = new_state.copy()
        self._round_state()

    def PHASE_polar(self, target_qubit, phi):
        """
        Apply the PHASE gate to the 'target_qubit' with a phase change 'phi'.

            keep the theta in _state: in the range [-pi, pi)
        """
        self._validate_qubit_index(target_qubit)
        self._applied_gates.append(
            {"gate": "PHASE", "target": target_qubit, "phi": phi})

        new_state = self._state.copy()

        for i in range(self._state_size):
            # Check if the target qubit is 0 or 1
            if (i >> target_qubit) & 1 != 0:
                new_theta = self._state[i][1] + phi
                if new_theta < -np.pi and abs(new_theta+np.pi) > 1e-6:
                    new_theta += np.pi * 2
                if new_theta > np.pi and abs(new_theta-np.pi) > 1e-6:
                    new_theta -= np.pi * 2
                new_state[i] = (self._state[i][0], new_theta)

        self._state = new_state.copy()
        self._round_state()

    def CPHASE_polar(self, control_qubit, target_qubit, phi):
        """
        Apply the CPHASE gate to the 'target_qubit' with a phase change 'phi' only if the 'control_qubit' is 1.

            keep the theta in _state: in the range [-pi, pi)
        """
        self._validate_qubit_index(target_qubit)
        self._applied_gates.append(
            {"gate": "CPHASE", "control": control_qubit, "target": target_qubit, "phi": phi})

        new_state = self._state.copy()

        for i in range(self._state_size):
            # Check if the target qubit is 0 or 1
            if (i >> control_qubit) & 1 == 1:
                if (i >> target_qubit) & 1 == 1:
                    new_theta = self._state[i][1] + phi
                    new_state[i] = (self._state[i][0], new_theta)

        self._state = new_state.copy()
        self._round_state()

    def CCPHASE_polar(self, control_qubits, target_qubit, phi):
        """
        Apply the CCPHASE gate to the 'target_qubit' with a phase change 'phi' only if the 'control_qubits' are 1s.

            keep the theta in _state: in the range [-pi, pi)
        """
        for i in control_qubits:
            self._validate_qubit_index(i)
        self._validate_qubit_index(target_qubit)
        self._applied_gates.append(
            {"gate": "CCPHASE", "control": control_qubits, "target": target_qubit, "phi": phi})

        new_state = self._state.copy()

        for i in range(self._state_size):
            # Check if the target qubit is 0 or 1
            flag = True
            for j in control_qubits:
                if (i >> j) & 1 == 0:
                    flag = False
            if flag:
                if (i >> target_qubit) & 1 == 1:
                    new_theta = self._state[i][1] + phi
                    new_state[i] = (self._state[i][0], new_theta)

        self._state = new_state.copy()
        self._round_state()

    def MCPHASE_polar(self, control_qubits, target_qubit, phi=np.pi):
        """
        Apply the MCPHASE gate to the target_qubit only if control_qubits are 1s.
        default: MCZ gate when the phi is not given.
        """
        for i in control_qubits:
            self._validate_qubit_index(i)
        self._validate_qubit_index(target_qubit)
        self._applied_gates.append(
            {"gate": "MCZ", "control": control_qubits, "target": target_qubit})

        new_state = self._state.copy()
        for i in range(self._state_size):
            # Check if the target qubit is 0 or 1
            flag = True
            for j in control_qubits:
                if (i >> j) & 1 == 0:
                    flag = False
            if flag:
                # Check if the target qubit is 0 or 1
                if (i >> target_qubit) & 1 != 0:
                    new_theta = self._state[i][1] + phi
                    new_state[i] = (self._state[i][0], new_theta)
        self._state = new_state.copy()
        self._round_state()

    def SWAP_polar(self, target_qubit1, target_qubit2):
        """
        Apply the SWAP gate to the 'target_qubit1' and 'target_qubit2'.

            keep the theta in _state: in the range [-pi, pi)
        """
        self._validate_qubit_index(target_qubit1)
        self._validate_qubit_index(target_qubit2)
        self._applied_gates.append(
            {"gate": "SWAP", "target1": target_qubit1, "target2": target_qubit2})

        self.CNOT_polar(target_qubit1, target_qubit2, log=False)
        self.CNOT_polar(target_qubit2, target_qubit1, log=False)
        self.CNOT_polar(target_qubit1, target_qubit2, log=False)

        self._round_state()

    def QFT_polar(self):
        """
        Implements the Quantum Fourier Transform on the circuit.
        """
        for i in reversed(range(self._num_qubits)):
            # apply HADAMARD
            self.hadamard_polar(i)

            # apply CPHASE
            for j in reversed(range(i)):
                phi = np.pi / (2 ** (i - j))
                self.CPHASE_polar(i, j, phi)

        # apply SWAP
        for i in range(self._num_qubits // 2):
            self.SWAP_polar(i, self._num_qubits - i - 1)

    def _diffusion_polar(self):
        """
        diffuser creator
        """
        num_qubits = self._num_qubits

        for qubit in range(num_qubits):
            self.hadamard_polar(qubit)
        for qubit in range(num_qubits):
            self.NOT_polar(qubit)
        self.MCPHASE_polar(list(range(num_qubits - 1)), num_qubits - 1)
        for qubit in range(num_qubits):
            self.NOT_polar(qubit)
        for qubit in range(num_qubits):
            self.hadamard_polar(qubit)

    def _oracle_polar(self, target_positions):
        """
        Oracle creator, encode the key and flip the phase of the target positions.
        """
        if isinstance(target_positions, int):
            target_positions = [target_positions]
        
        for target_pos in target_positions:
            binary_state = bin(target_pos)[2:].zfill(self._num_qubits)

            for i, qubit in enumerate(reversed(binary_state)):
                if qubit == '0':
                    self.NOT_polar(i)
            self.MCPHASE_polar(list(range(self._num_qubits - 1)), self._num_qubits - 1)
            for i, qubit in enumerate(reversed(binary_state)):
                if qubit == '0':
                    self.NOT_polar(i)

    def Grover_polar(self, target_positions, iterations: int):
        """
        Implement Grover's algorithm with oracle and diffuser.
        """
        # create superposition
        for qubit in range(self._num_qubits):
            self.hadamard_polar(qubit)

        # Grover iterations
        for _ in range(iterations):
            self._oracle_polar(target_positions)
            self._diffusion_polar()

    def READ_polar_singlequbit(self, target_qubit, num_shots=1):
        """
        To measure the state of the 'target_qubit' for 'num_shots' of times.
        """
        self._validate_qubit_index(target_qubit)
        amplitute0, amplitute1 = 0, 0
        for i in range(self._state_size):
            # Check if the target qubit is 0 or 1
            if (i >> target_qubit) & 1 == 0:
                amplitute0 += np.square(self._state[i][0])
            else:
                amplitute1 += np.square(self._state[i][0])

        prob_qubit = [amplitute0, amplitute1]

        possible_outcome = np.arange(2)
        measurement = dict()
        for i in possible_outcome:
            measurement[str(i)] = 0
        # print(measurement)
        for i in range(num_shots):
            x = random.choices(possible_outcome, weights=prob_qubit)[0]
            measurement[str(x)] += 1
        return measurement

    def READ_polar_all(self):
        """
        To measure the entire quantum system, all qubits.
        """
        amplitute = []
        for i in range(self._state_size):
            amplitute.append(np.square(self._state[i][0]))

        possible_outcome = np.arange(self._state_size)
        measurement = dict()

        x = random.choices(possible_outcome, weights=amplitute)[0]
        measurement[str(x)] = 1

        return measurement

    def simulation(self, num_shots=1):
        """
        Running simulation of 'num_shots' of times by measuring all the qubits for each simulation.
        """
        result = dict()

        for _ in range(num_shots):
            temp = self.READ_polar_all()
            result = {key: result.get(key, 0) + temp.get(key, 0)
                      for key in result.keys() | temp.keys()}
        return result
        # sorted_result = sorted(result.items(), key=lambda x:(x[0]))
        # return sorted_result


def viz2(state_vector, num_of_qubit=3):
    """
    Output the visulization of the quantum states into circles, where the radius indicates the magnitude and the arrow indicates the phase.
    'state_vector' is Cartesian-based states.
    """
    n_states = int(math.pow(2, num_of_qubit))

    # calculate the amplitude and phase of the states
    prob_qubit = np.absolute(state_vector)
    phase_qubit = np.angle(state_vector)
    rows = int(math.ceil(n_states / 8.0))
    cols = min(n_states, 8)
    fig, axs = plt.subplots(rows, cols)
    for col in range(cols):
        # amplitude area
        circleExt = matplotlib.patches.Circle(
            (0.5, 0.5), 0.5, color='gray', alpha=0.1)
        circleInt = matplotlib.patches.Circle(
            (0.5, 0.5), prob_qubit[col]/2, color='b', alpha=0.3)
        axs[col].add_patch(circleExt)
        axs[col].add_patch(circleInt)
        axs[col].set_aspect('equal')
        state_number = "|" + str(col) + ">"
        axs[col].set_title(state_number)
        xl = [0.5, 0.5 + 0.5*prob_qubit[col] *
              math.cos(phase_qubit[col] + np.pi/2)]
        yl = [0.5, 0.5 + 0.5*prob_qubit[col] *
              math.sin(phase_qubit[col] + np.pi/2)]
        axs[col].plot(xl, yl, 'r')
        axs[col].axis('off')
    plt.show()

# for multi-qbits
def vizMulti(state_vector, num_of_qubit=3):
    n_states = int(math.pow(2, num_of_qubit))

    # calculate the amplitude and phase of the states
    prob_qubit = np.absolute(state_vector)
    phase_qubit = np.angle(state_vector)
    rows = int(math.ceil(n_states / 16.0))
    cols = min(n_states, 16)

    # Create the subplots
    fig, axs = plt.subplots(rows, cols)

    # Flatten axs in case it is a 2D array
    axs = axs.flatten() if isinstance(axs, np.ndarray) else [axs]

    for col in range(n_states):  # loop over all possible states
        # amplitude area
        circleExt = matplotlib.patches.Circle(
            (0.5, 0.5), 0.5, color='gray', alpha=0.1)
        circleInt = matplotlib.patches.Circle(
            (0.5, 0.5), prob_qubit[col] / 2, color='b', alpha=0.3)
        axs[col].add_patch(circleExt)
        axs[col].add_patch(circleInt)
        axs[col].set_aspect('equal')
        state_number = "|" + str(col) + ">"
        axs[col].set_title(state_number, fontsize=5)
        xl = [0.5, 0.5 + 0.5 * prob_qubit[col] *
              math.cos(phase_qubit[col] + np.pi / 2)]
        yl = [0.5, 0.5 + 0.5 * prob_qubit[col] *
              math.sin(phase_qubit[col] + np.pi / 2)]
        axs[col].plot(xl, yl, 'r')
        axs[col].axis('off')

    plt.show()
