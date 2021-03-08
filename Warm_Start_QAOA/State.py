import numpy as np
#import tensorflow as tf
#import tensorflow_quantum as tfq

import cirq
import sympy
from cirq.contrib.svg import SVGCircuit

#State class to initialise qubits and also to prepare intial hadamard state for the quibits

class states(object):
    
    def __init__(self, num_qubits, algo_name):
    
        self.algorithm = algo_name
        self.num_qubits = num_qubits
        #self.qubit_list = []
        self.gate_list = ["hadamard"]*num_qubits

    
    def get_qubits(self):

        #creates a list of qubit using list format to define cirq qubits
        #self.initial_qubit_list = cirq.LineQubit.range(self.num_qubits)
        #line_qubit doesn't serialize so use grid_qubit only to merge with tensorflow_quantum
        self.initial_qubit_list = cirq.GridQubit.rect(1,self.num_qubits)

    
    def get_params(self, num_params):
       
       #learnable params whose value is to be fed-----
       #a = init_name + "_" +str(1)
       #b = init_name + "_" +str(2) 
       self.named_params = sympy.symbols('param_1, param_2')
       curr_iters = num_params-2
       for i in range(curr_iters):
           self.named_params += (sympy.symbols('param_'+str(i+3)),)
       return self.named_params


    
    def get_hadamard_basis_state(self, circuit):
        #outputs the hadamarded stae with initialised circuit given as input
        self.get_qubits()
        #Applying hadamard gates to the intial qubits
        print(type(cirq.H(self.initial_qubit_list[0])))
        for i in range(self.num_qubits):
            circuit.append(cirq.H(self.initial_qubit_list[i]))
        
        #plotting the circuit-----
        
        return circuit
    
    
    def apply_hadamard(self, num_qubits, qubits, circuit):
        #applies hadamard to a given set of qubits in the given circuit
        
        for i in range(num_qubits):
            circuit.append(cirq.H(qubits[i]))
        return circuit

    
    def apply_pauli_gate(self, num_qubits, qubits, circuit, gate_index):
    #applies pauli gates to a given set of qubits in the given circuit
    #gate_index --> 1 means Z gate and so on upto 3
         for i in range(num_qubits):
            if gate_index==0:
             circuit.append(cirq.Z(qubits[i]))
            elif gate_index==1:
             circuit.append(cirq.X(qubits[i]))
            else:
                circuit.append(cirq.Y(qubits[i]))

         return circuit

    def prepare_controlled_gate_state(self, num_qubits, qubits, circuit, gate_index):
     #apply controlled pauli gates between every consecutive qubits in the given set
     # gate_index--> means Z gate and so on upto 3   
        for i in range(num_qubits-1):
            if gate_index==0:
             circuit.append(cirq.CZ(qubits[i],qubits[i+1]))
            elif gate_index==1:
             circuit.append(cirq.CX(qubits[i],qubits[i+1]))
            else:
                circuit.append(cirq.CX(qubits[i],qubits[i+1]))

        return circuit
    

    
    def plot_circuit(self, circuit):
        #to plot the circuit at any moment
        print("The circuit is as folows-----> :)")
        SVGCircuit(circuit)
    
    



    
#Checking the state calss implementation        

if __name__=="__main__":
     curr_circuit = cirq.Circuit()
     state_class  = states(4, "Grover")
     #state_class.get_qubits()
     state_class.get_hadamard_basis_state(curr_circuit)
     #state_class.plot_circuit(curr_circuit)
     SVGCircuit(curr_circuit)
    


