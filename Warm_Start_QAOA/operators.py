import cirq
import numpy as np
from utility import *

"""
A class that realizes the gamma and beta  operators using the gates described in cirq and which are supported by tensorflow quantum.
"""
class operators:
    def __init__(self, num_qubits, qubits, circuit, connectivity_graph, gamma, beta, initial_angles, connectivity="SK"):
        self.num_qubits= num_qubits
        self.qubits = qubits
        self.circuit = circuit
        """
        Initializing the operator names
        """
        self.operator_list = ["gamma_unitary", "driving_unitary"]
        self.gamma, self.beta = gamma, beta
        self.connectivity = connectivity
        self.gamma_unitary = gamma_unitary(num_qubits, qubits, connectivity_graph, self.gamma, circuit, connectivity)
        self.beta_unitary = mixer_hamiltonian(num_qubits, initial_angles)

    def apply_gamma_unitary(self, circuit):
             if self.connectivity=="SK":
                 self.gamma_unitary.apply_complete_unitary_for_SK()
             else:
                 self.gamma_unitary.apply_complete_unitary()
  
    
    def apply_beta_unitary(self, circuit):
        self.beta_unitary.apply_beta_unitary(self.beta, self.qubits, self.circuit)


class gamma_unitary:
    """
    CLass for the gamma unitary operator
    """
    def __init__(self, num_qubits, qubits, connectivity_graph, symbol, circuit, connectivity="SK"):
        self.num_qubits = num_qubits
        self.qubits = qubits
        self.graph = connectivity_graph
        self.symbol =  symbol
        self.circuit = circuit
        self.connectivity = connectivity
    
    def apply_SYC_gate(self, qubit_index_1, qubit_index_2):
        """
        applies the syc gate described in the paper
        """
        self.circuit.append(cirq.ISWAP(self.qubits[qubit_index_1],self.qubits[qubit_index_2])**(-1))
        self.circuit.append(cirq.CZ(self.qubits[qubit_index_1],self.qubits[qubit_index_2])**(-1/6))

    def apply_ZZ_swap_gate(self, weight, qubit_index_1, qubit_index_2):
        """
        intermediate gate which uses syc gate and is used in routing also, applied to every pair of connected qubits during the gamma_unitary
        and has gamma,w dependent single qubit rotation gates.
        """
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_1]))
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_2]))
        self.apply_SYC_gate(qubit_index_1, qubit_index_2)
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_1]))
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_2]))
        self.apply_SYC_gate(qubit_index_1, qubit_index_2)
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_1]))
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_2]))
        self.apply_SYC_gate(qubit_index_1, qubit_index_2)
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_1]))
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_2]))
    
    def apply_ZZ_gate(self, weight, qubit_index_1, qubit_index_2):
        """
        intermediate gate which uses syc gate and is applied to every pair of connected qubits during the gamma_unitary
        and has gamma,w dependent single qubit rotation gates.
        """
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_1]))
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_2]))
        self.apply_SYC_gate(qubit_index_1, qubit_index_2)
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_1]))
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_2]))
        self.apply_SYC_gate(qubit_index_1, qubit_index_2)
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_1]))
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123, exponent = weight*(self.symbol))(self.qubits[qubit_index_2]))
        
        
    

    def apply_complete_unitary_for_SK(self):
        """
        One pass of the complete unitary for the SK connectivity
        """
        #for i in range(self.num_qubits):
         #   for j in range(i):
          #      if self.graph[i][j]!=0:
           #         self.apply_ZZ_swap_gate(self.graph[i][j],i,j)
        flag = False
        k = 0
        for i in range(self.num_qubits):
          if flag:
              k=1 
          for j in range(k,self.num_qubits,2):
                self.apply_ZZ_swap_gate(self.graph[j][j+1],j,j+1)

        return self.circuit
    
    def apply_complete_unitary(self):
        for i in range(self.num_qubits):
           for j in range(i):
               if self.graph[i][j]!=0:
                   self.apply_ZZ_gate(self.graph[i][j],i,j)
        return self.circuit
    
        





class mixer_hamiltonian:
    def __init__(self, num_qubits, initial_angles):
        self.num_qubits = num_qubits
        self.initial_angles = initial_angles
    
    def apply_unitary(self, symbol, qubits, circuit):
        for i in range(self.num_qubits):
            circuit.append(cirq.ry(-self.initial_angles[i])(qubits[i]))
            circuit.append(cirq.rz(-2*symbol)(qubits[i]))
            circuit.append(cirq.ry(self.initial_angles[i])(qubits[i]))
        
        return circuit

class correlator_RQAOA:
    def __init__(self, num_qubits, weights):
        self.num_qubits = num_qubits
        self.weights = weights
    
    def apply_unitary(self, qubits, circuit, i, j, symbol):
        curr_w1_total = get_sum_except_given(self.weights, i, j)
        curr_w2_total = get_sum_except_given(self.weights, j, i)
        circuit.append(cirq.Z(qubits[i])**(symbol*curr_w1_total))
        circuit.append(cirq.Z(qubits[j])**(symbol*curr_w2_total))
        for k in range(self.num_qubits):
            if k!=i and k!=j:
                circuit.append(circuit.append(cirq.CZ(qubits[k],qubits[j])**(-2*symbol*self.weights[k][j])))
                circuit.append(circuit.append(cirq.CZ(qubits[k],qubits[i])**(-2*symbol*self.weights[k][i])))
        """
        Uij now-->
        """
        circuit.append(cirq.Z(qubits[i])**(symbol*self.weights[i][j]))
        circuit.append(cirq.Z(qubits[j])**(symbol*self.weights[i][j]))
        circuit.append(circuit.append(cirq.CZ(qubits[i],qubits[j])**(-2*symbol*self.weights[i][j])))

        
        







        

