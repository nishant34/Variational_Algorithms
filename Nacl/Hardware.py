import cirq
#import tensorflow_quantum as tfq
#import tensorflow as tf
import numpy 
from cirq.devices import GridQubit
from Gate_set import *

class Device(cirq.Device):
    def __init__(self, gate_set_list, connectivity, num_qubits, name):
        ##connectivity should be in form a list with each element being a 2 element list determining both are connected
        self.num_gates = len(gate_set_list)
        self.gate_set_list = gate_set_list
        self.connectivity  = connectivity
        self.possible_configs = []
        self.num_qubits = num_qubits
        self.qubits = [GridQubit(i, 0) for i in range(num_qubits)]

        assert (num_qubits>=len(connectivity)),"All qubits are not utilised by device"
        self.name = name
        self.gate_interface = custom_gate_set()
        self.noise_method = 0


    
    def get_noise_model_for_gates(self):
        
        "Custom Quantum Channels or cirq.depolarize if directly not supported in tensorflow quantum "
        "Noise can be placed manually using gates also."

        ".............Adding manual error in form of gates....."
        self.one_qubit_rot_noise = cirq.rx(0.3)
        self.one_qubit_I_noise = cirq.rz(0.1)
        self.one_qubit_pauli_noise = cirq.rz(0.2)
        self.two_qubit_rot_noise = cirq.CZ**0.3
        self.two_qubit_cnot_noise = cirq.CZ**0.2
        self.n_rot_noise_angle = 0.3
        self.n_cnot_noise_angle = 0.3
        self.n_qubit_controlled_rot_noise_fn = self.gate_interface.recursive_n_bit_controlled
        self.n_qubit_cnot_noise_fn = self.gate_interface.recursive_CNOT
        

    def apply_noise_to_controlled_rot_2_qubits(self, circuit, qubits):

        "Adding noise to controlled rot"

        circuit.append(self.one_qubit_I_noise(qubits[0]))
        circuit.append(self.two_qubit_rot_noise(qubits[0],qubits[1]))
        return circuit
    
    def apply_noise_to_CNOT_2_qubits(self, circuit, qubits):

        "Adding noise to CNOT"

        circuit.append(self.one_qubit_I_noise(qubits[0]))
        circuit.append(self.two_qubit_cnot_noise[qubits[0],qubits[1]])
        return circuit
    
    def apply_noise_to_controlled_rot_N_qubits(self, circuit, qubits, num_qubits):

        "Adding noise to N-controlled rot"
        
        for i in range(num_qubits-1):
            circuit.append(self.one_qubit_I_noise(qubits[i]))
        self.n_qubit_controlled_rot_noise_fn(circuit, self.n_rot_noise_angle, qubits, num_qubits)
        return circuit

    def apply_noise_to_CNOT_N_qubits(self, circuit, qubits, num_qubits):
        
        "Adding noise to N-controlled NOT"

        for i in range(num_qubits-1):
            circuit.append(self.one_qubit_I_noise(qubits[i]))
        self.n_qubit_cnot_noise_fn(circuit, self.n_cnot_noise_angle, qubits, num_qubits)
        return circuit
    
    def apply_manual_error_gates(self, parent_gate_name, circuit, qubits, num_qubits):

        "Wrapper method to apply error given the gate name."
        
        if num_qubits==1:
            if parent_gate_name[0]=='r' or parent_gate_name[0] == 'R':
                circuit.append(self.one_qubit_rot_noise(qubits))
            elif parent_gate_name[0] == 'I':
                circuit.append(self.one_qubit_I_noise(qubits))
            else:
                circuit.append(self.one_qubit_pauli_noise(qubits))
        
        elif num_qubits==2:
            if parent_gate_name[1]=='N':
                self.apply_noise_to_CNOT_2_qubits(circuit, qubits)
            else :
                self.apply_noise_to_controlled_rot_2_qubits(circuit, qubits)

        else:
            if "NOT" in parent_gate_name:
                self.apply_noise_to_controlled_rot_N_qubits(circuit, qubits, num_qubits)
            else:
                self.apply_noise_to_CNOT_N_qubits(circuit, qubits, num_qubits)


                


        
    

        

    
        