import numpy as np
import os
import cirq
from utility import *

#import tensorflow as tf
#import tensorflow_quantum as tfq
##class which acts as an interface to add gates to the circuit

class custom_gate_set:
    def __init__(self):
        self.gate_list = ["I","Rx","Ry","Rz","X","Y","Z","CNOT","CZ","CX","H"]
        self.continous_param_gate_list = ["Rx","Ry","Rz","CZ","CX"]
        self.gate_to_num_qubits_dict = {
            "Rx":1,
            "Ry":1,
            "Rz":1,
            "X":1,
            "Y":1,
            "Z":1,
            "CNOT":2,
            "CZ":2,
            "H":1,
            "I":1,
        }
        self.gates_available_currently = [1,1,1,1,1,1,1,1,1,1]
        self.gate_function_list = []
        
    def set_gates(self):
        self.gate_function_list.append(cirq.rx)
        self.gate_function_list.append(cirq.ry)
        self.gate_function_list.append(cirq.rz)
        self.gate_function_list.append(cirq.X)
        self.gate_function_list.append(cirq.Y)
        self.gate_function_list.append(cirq.Z)
        self.gate_function_list.append(cirq.CNOT)
        self.gate_function_list.append(cirq.CZ)
        self.gate_function_list.append(cirq.H)
        self.gate_function_list.append(cirq.IdentityGate)

    def get_gate_dict(self):
        gate_dict = {

        }
        count = 0
        for gate in self.gate_list:
            gate_dict[gate] = count
            count+=1
        
        return gate_dict

    def get_avaiable_gates(self, available_gate_list):
        gate_dict = self.get_gate_dict()
        gate_function = []
        for gate in available_gate_list:
            gate_function.append(gate_dict[gate])
        return gate_function
    
    def recursive_n_bit_controlled(self,circuit,angle,qubits,num_qubits):
        if len(qubits)==2:
           circuit.append(cirq.CZ(*[qubits[0],qubits[1]])**angle)
           return 
        angle1 = angle/2

        #helper_n_qubit(circuit,angle1,qubits,num_qubiits)
        circuit.append(cirq.CZ(*[qubits[-2],qubits[-1]])**angle1)
        self.recursive_CNOT(circuit,1,qubits[:-1],num_qubits-1)
        circuit.append(cirq.CZ(*[qubits[-2],qubits[-1]])**-angle1)
        self.recursive_CNOT(circuit,1,qubits[:-1],num_qubits-1)
    
        qubits_new = qubits[:num_qubits-2]
        qubits_new.append(qubits[-1])

        self.recursive_n_bit_controlled(circuit,angle1,qubits_new,num_qubits-1)
    
    def recursive_CNOT(self,circuit,angle,qubits,num_qubits):
       if len(qubits)==2:
        circuit.append(cirq.H(qubits[1]))
        circuit.append(cirq.CZ(*[qubits[0],qubits[1]])**angle)
        circuit.append(cirq.H(qubits[1]))
        return 
       angle1 = angle/2
    
       #helper_n_qubit(circuit,angle1,qubits,num_qubiits)
       circuit.append(cirq.H(qubits[-1]))
       circuit.append(cirq.CZ(*[qubits[-2],qubits[-1]])**angle1)
       circuit.append(cirq.H(qubits[-1]))
       self.recursive_CNOT(circuit,1,qubits[:-1],num_qubits-1)
       circuit.append(cirq.H(qubits[-1]))
       circuit.append(cirq.CZ(*[qubits[-2],qubits[-1]])**-angle1)
       circuit.append(cirq.H(qubits[-1]))
       self.recursive_CNOT(circuit,1,qubits[:-1],num_qubits-1)

       qubits_new = qubits[:num_qubits-2]
       qubits_new.append(qubits[-1])

       self.recursive_CNOT(circuit,angle1,qubits_new,num_qubits-1)

    
    def add_gate_to_circuit(self, circuit, qubits, gate_name, num_qubits_to_act, required_noise, conitnous_param, symbol):
        
        gate_dict = self.get_gate_dict()

        if num_qubits_to_act ==1:
            if not conitnous_param:
             circuit.append(self.gate_function_list[gate_dict[gate_name]](qubits[0]))
            
            else:
                circuit.append((self.gate_function_list[gate_dict[gate_name]]**symbol)(qubits[0]))

        if num_qubits_to_act ==2:
            if not conitnous_param:
             circuit.append(self.gate_function_list[gate_dict[gate_name]](qubits[0],qubits[1]))
            
            else:
                circuit.append((self.gate_function_list[gate_dict[gate_name]]**symbol)(qubits[0],qubits[1]))
        
        else:
            ## now for more than 2 qubits there is an n-bit controlled circuit
            if gate_name=="N-CNOT":
                self.recursive_CNOT(circuit,symbol,qubits,num_qubits_to_act)
            else:

             self.recursive_n_bit_controlled(circuit,symbol,qubits,num_qubits_to_act)
    


    #def generate_maximally_parallel_gate_sequence(self, qubitwise_gate_list):
            #num_qubits = len(qubitwise_gate_list)
            #total_time_steps = len(qubitwise_gate_list[0])
            #gate_sequence = []
            #for i in range(total_time_steps):
                

            
    def gates_commute(self, gate_1, gate_2, num_qubits_1, num_qubits_2):
            curr = max(num_qubits_1, num_qubits_2)
            qubits = cirq.GridQubit.rect(1,curr)
            circuit = cirq.Circuit()
            circuit_1 = cirq.Circuit()
            circuit.append(gate_1(*qubits[:num_qubits_1]))
            circuit.append(gate_2(*qubits[:num_qubits_2]))
            circuit_1.append(gate_2(*qubits[:num_qubits_2]))
            circuit_1.append(gate_1(*qubits[:num_qubits_1]))
            if circuit._unitary_() == circuit_1._unitary_():
                return True
            return False
    
    
            
            

            




    





