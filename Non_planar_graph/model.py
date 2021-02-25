import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import os
import scipy

from utilities import *
from train_helper import *
from State import *
from operators import *
#from utilities import *
from tensorflow.keras.models import Model
import math


"""
Class to prepare the variational circuit required for the grpah problem. It is called in the wrapper model class provided below.
"""
class variational_circuit:
    
    def __init__(self, num_qubits, target_problem, connectivity_graph, step_p, symbols):
        
          #self.circuit = circuit
          self.target_problem = target_problem
          self.step_p = step_p
          self.num_qubits  = num_qubits
          self.symbols = symbols
          self.connectivity_graph = connectivity_graph
          self.connectivity = target_problem
          
    def prepare_circuit(self, qubits, circuit):
        
        for i in range(0,self.step_p,2):
             self.curr_operator = operators(self.num_qubits, qubits, self.connectivity, circuit, self.connectivity_graph, self.symbols[i], self.symbols[i+1])
             self.curr_operator.apply_gamma_unitary(circuit)
             self.curr_operator.apply_beta_unitary(circuit)
        
        return circuit
    

"""
Wrapper model class to link cirq with tensorflow quantum through the PQC layer.
"""
class model(Model):
    
    def __init__(self, num_qubits, target_problem, step_p, connectivity_graph):
        super(model, self).__init__()
        self.num_qubits = num_qubits
        self.target_problem  = target_problem
        self.step_p = step_p
        self.connectivity_graph = connectivity_graph
        self.state_class = states(num_qubits, "Non planar graph")
        self.symbols = self.state_class.get_params(2*step_p)
        self.circuit_class  = variational_circuit(num_qubits, target_problem, connectivity_graph, step_p, self.symbols)
        
    def get_readouts(self, qubits):
        
       
       if self.target_problem == "Hardware Grid":
         self.readouts = get_cost_function_for_hardware_grid(self.connectivity_graph, qubits)
       else:
         self.readouts = get_cost_function(self.num_qubits, self.connectivity_graph, qubits)
    
    def prepare_circuit_for_quantum_layer(self, qubits):
        circuit = cirq.Circuit()
        return self.circuit_class.prepare_circuit(qubits, circuit)

    def prepare_quantum_layer(self, qubits):

        self.pqc_layer = tfq.layers.PQC(self.prepare_circuit_for_quantum_layer(qubits),self.readouts)

    
    def call(self, input_state):
       
       output  = self.pqc_layer(input_state[0])
       
       return output

    

    

