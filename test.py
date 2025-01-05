from PolarQuantumSimulator import PolarQuantumCircuit, viz2, vizMulti

num_qubits = 3
qc = PolarQuantumCircuit(num_qubits)
polar_states = qc.get_state()
complex_states = PolarQuantumCircuit.polar2cartesian(polar_states)
viz2(complex_states, num_of_qubit=num_qubits)

# qc.NOT_polar(0)
# qc.NOT_polar(1)
# qc.NOT_polar(num_qubits-1)
# qc.hadamard_polar(0)
# qc.hadamard_polar(1)
# for i in range(num_qubits):
#     qc.hadamard_polar(i)
qc.Grover_polar([2, 7], 2)
polar_states = qc.get_state()
complex_states = PolarQuantumCircuit.polar2cartesian(polar_states)
viz2(complex_states, num_of_qubit=num_qubits)

# qc.MCPHASE_polar(list(range(num_qubits-1)), num_qubits-1)
qc.QFT_polar()
polar_states = qc.get_state()
complex_states = PolarQuantumCircuit.polar2cartesian(polar_states)
viz2(complex_states, num_of_qubit=num_qubits)