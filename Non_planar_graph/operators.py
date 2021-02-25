import cirq
#import tensorflow_quantum as tfq
#import tensoflow as tf
import sympy as sp
import numpy as np

"""
A class that realizes the gamma and beta  operators using the gates described in cirq and which are supported by tensorflow quantum.
"""
class operators:
    def __init__(self, num_qubits, qubits, connectivity, circuit, connectivity_graph, gamma, beta):
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
        self.beta_unitary = beta_unitary(num_qubits, qubits, self.beta, circuit)

    def apply_gamma_unitary(self, circuit):
             if self.connectivity=="SK":
                 self.gamma_unitary.apply_complete_unitary_for_SK()
             else:
                 self.gamma_unitary.apply_complete_unitary()
  
    
    def apply_beta_unitary(self, circuit):
        self.beta_unitary.apply_beta_unitary()



class gamma_unitary:
    """
    CLass for the gamma unitary operator
    """
    def __init__(self, num_qubits, qubits, connectivity_graph, symbol, circuit, connectivity):
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
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123)(self.symbol*weight)(self.qubits[qubit_index_1]))
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123)(self.symbol*weight)(self.qubits[qubit_index_2]))
        self.apply_SYC_gate(qubit_index_1, qubit_index_2)
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123)(self.symbol*weight)(self.qubits[qubit_index_1]))
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123)(self.symbol*weight)(self.qubits[qubit_index_2]))
        self.apply_SYC_gate(qubit_index_1, qubit_index_2)
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123)(self.symbol*weight)(self.qubits[qubit_index_1]))
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123)(self.symbol*weight)(self.qubits[qubit_index_2]))
        self.apply_SYC_gate(qubit_index_1, qubit_index_2)
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123)(self.symbol*weight)(self.qubits[qubit_index_1]))
        self.circuit.append(cirq.PhasedXPowGate(phase_exponent=0.123)(self.symbol*weight)(self.qubits[qubit_index_2]))
    
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
    
    
class beta_unitary:
    """
    Class for the beta unitary operator
    """
    def __init__(self, num_qubits, qubits, symbol, circuit):
        self.num_qubits = num_qubits
        self.qubits = qubits
        self.symbol =  symbol
        self.circuit = circuit
    

    def apply_beta_unitary(self):
        """
        One pass of the complete beta unitary
        """
        for i in range(self.num_qubits):
            self.circuit.append(cirq.rx(self.symbol)(self.qubits[i]))
        return self.circuit
    


"""
SYC Gate-->
circuit = cirq.Circuit()
circuit.append(cirq.ISWAP(qubits[0],qubits[1])**(-1))
circuit.append(cirq.CZ(qubits[0],qubits[1])**(-1/6))
circuit._unitary_()
"""