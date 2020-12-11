import cirq
import os 
import numpy as np
import sympy
from scipy.linalg import expm
from cirq.contrib.svg import SVGCircuit
#This contains the "operator" class and "rank-1 projector" class to define custom operators and projectors

#for generating n-qubit hadamard tensor product
def generate_hadamard_matrix(num_qubits):
    a  = (1 / np.sqrt(2))*np.array([ [1, 1], [1, -1]])
    b = a
    for i in range(num_qubits-1):

        b = np.kron(b,a)

    return b    

#custom gate-----
class variational_oracle_gate(cirq.Gate):
    def __init__(self, num_qubits, alpha, projection):
        super(variational_oracle_gate,self)
        self.num_qubits = num_qubits
        self.param_type = "alpha"
        self.param = alpha
        self.projection = projection
    
    def _num_qubits_(self):
        return self.num_qubits


    def _unitary_(self):

        matrix = expm(1j*self.param*self.projection)
        return matrix        
        
    #def _circuit_diagram_info_(self, args):
        #return "VOG"
    def _circuit_diagram_info_(self, args):
        named_list = []
        #return "wire1","wire2","wire3","wire4"
        for i in range(self.num_qubits):
            named_list.append("wire_VOG_"+str(i))
        return named_list

    

#Projection matrix--------> :)
class projector_matrix(object):
    def __init__(self, rank, num_qubits):
        self.rank = rank
        self.num_qubits = num_qubits
        
    def generate_rank_1_projector_matrix(self, projection_index):
        assert (self.rank==1),"This is not a rank 1 projector"
        self.vec = np.zeros(2**self.num_qubits)
        self.vec[projection_index]=1

        self.projector_matrix =  np.outer(self.vec,self.vec)
        return self.projector_matrix
    
    def generate_perpendicular_subspace_projector_matrix(self, projection_matrix):
        return np.eye(self.num_qubits)-projection_matrix


#The variational diffusion custom gate---->
#zero projector will be used
class variational_diffusion_gate(cirq.Gate):
    def __init__(self, num_qubits, beta):
        super(variational_diffusion_gate,self)
        self.num_qubits = num_qubits
        self.param_type = "beta"
        self.param = beta
        self.projection_head = projector_matrix(1,num_qubits)
        self.projection = self.projection_head.generate_rank_1_projector_matrix(0)
        self.n_bit_hadamard = generate_hadamard_matrix(num_qubits)

    def _num_qubits_(self):
        return self.num_qubits


    def _unitary_(self):
        project_zero = self.n_bit_hadamard@self.projection@self.n_bit_hadamard
        matrix = expm(1j*self.param*project_zero)
        return matrix        
        
    #def _circuit_diagram_info_(self, args):
        #return "VOG"
    def _circuit_diagram_info_(self, args):
        named_list = []
        #return "wire1","wire2","wire3","wire4"
        for i in range(self.num_qubits):
            named_list.append("wire_VDG_"+str(i))
        return named_list


##Can be used as a gate to project the probability of lying perpendicular to a certain state. For example case n=3 it will be a 2D plane or rank 2 matrix 
class perpendicular_expectation_gate(cirq.Gate):
    def __init__(self, num_qubits):
        super(perpendicular_expectation_gate,self)
        self.num_qubits = num_qubits
        self.projection_head = projector_matrix(1,num_qubits)        
    
    def get_target_matrix(self, projection_index):
        self.target_matrix = self.projection_head.generate_rank_1_projector_matrix(projection_index)
  

    def _num_qubits_(self):
        return self.num_qubits

    def _unitary_(self):
        return self.projection_head.generate_perpendicular_subspace_projector_matrix(self.target_matrix)
    
    def _circuit_diagram_info_(self, args):
        return "PSE"
    

# to check the implementation---------
if __name__=="__main__":
    num_qubits = 8
    curr_circuit = cirq.Circuit()
    initial_qubit_list = cirq.LineQubit.range(num_qubits)
    custom_projector = projector_matrix(1,num_qubits)
    curr_matrix = custom_projector.generate_rank_1_projector_matrix(2)

    custom_gate_curr = variational_oracle_gate(num_qubits,np.pi,curr_matrix)
    
    custom_gate_curr_1 = variational_diffusion_gate(num_qubits,np.pi)
    curr_circuit.append(custom_gate_curr_1.on(*initial_qubit_list))
    SVGCircuit(curr_circuit)
       

