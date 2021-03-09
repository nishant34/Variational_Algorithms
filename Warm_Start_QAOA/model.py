import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import os
import scipy

from utility import *
from train_helper import *
from State import *
from operators import *
#from utilities import *
from tensorflow.keras.models import Model
import math
from warm_start import *
from train_helper import *
from config import *



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
          self.correlator_operator = correlator_RQAOA(num_qubits, connectivity_graph)
          
    def prepare_QAOA_circuit(self, qubits, circuit, initial_angles):
        
        for i in range(0,self.step_p,2):
             self.curr_operator = operators(self.num_qubits, qubits, circuit, self.connectivity_graph, self.symbols[i], self.symbols[i+1], initial_angles)
             self.curr_operator.apply_gamma_unitary(circuit)
             self.curr_operator.apply_beta_unitary(circuit)
        
        return circuit
    
    


"""
Wrapper model class to link cirq with tensorflow quantum through the PQC layer.
"""
class model(Model):
    
    def __init__(self, num_qubits, target_problem, step_p, connectivity_graph, warm_start_class):
        super(model, self).__init__()
        self.num_qubits = num_qubits
        self.target_problem  = target_problem
        self.step_p = step_p
        self.connectivity_graph = connectivity_graph
        self.state_class = states(num_qubits, "Warm Start")
        self.symbols = self.state_class.get_params(2*step_p)
        self.warm_start_class=  warm_start_class
        self.circuit_class  = variational_circuit(num_qubits, target_problem, connectivity_graph, step_p, self.symbols)
        
    def get_readouts(self, qubits, task_class):
        
       
       #if self.target_problem == "Hardware Grid":
       #  self.readouts = get_cost_function_for_hardware_grid(self.connectivity_graph, qubits)
       #else:
       if self.target_problem=="Max-cut":
         #self.readouts = get_cost_function(self.num_qubits, self.connectivity_graph, qubits)
         self.readouts = task_class.get_max_cut_formulation(qubits)
    
    def prepare_circuit_for_quantum_layer(self, qubits):
        circuit = cirq.Circuit()
        return self.circuit_class.prepare_QAOA_circuit(qubits, circuit, self.warm_start_class.initial_angles)

    def prepare_quantum_layer(self, qubits):

        self.pqc_layer = tfq.layers.PQC(self.prepare_circuit_for_quantum_layer(qubits),self.readouts)

    
    def call(self, input_state):
       
       output  = self.pqc_layer(input_state[0])
       
       return output
    
    

"""
wrapper class for solving by recursion--> depth_1_QAOA
"""
class R_QAOA:
    def __init__(self, num_qubits, graph_structure, symbols, threshold_nodes, warm_start_class, step=1):
        self.num_qubits = num_qubits
        self.graph_structure = graph_structure
        self.symbols = symbols
        self.QAOA = model(num_qubits, "Max-cut", 1, graph_structure, warm_up_start_class)
        self.threshold_nodes = threshold_nodes
        self.correlator = correlator_RQAOA(num_qubits, weights)
        #self.qubits = qubits
        self.warm_start_class = warm_start(num_qubits, qubits, sigma, mu)
        
    def set_up_problem_matrices(self, sigma, mu, qubits):
        self.sigma = sigma
        self.mu = mu
        self.qubits = qubits
        self.warm_start  = warm_start(self.num_qubits, qubits, sigma, mu)
    
    def assign_weights(self, weights):
        self.weights = weights

    def generate_good_solutions_from_GW(self, k, num_qubits, weights, graph_structure):
        """
        generates k good solutions from the GW alog which uses randomized rounding
        """
        nx_graph = generate_graph_for_gw(num_qubits, weights, graph_structure)
        sols_vector = []
        for i in range(k):
            sols_vector.append(self.warm_start.goemans_williamson(nx_graph))
        
        return sols_vector
    
    def get_recursive_QAOA_solution(self, circuit):
        nx_graph = generate_graph_for_gw(self.num_qubits, self.weights, self.graph_structure)
        if self.num_qubits  == self.threshold_nodes:
            return self.warm_start.goemans_williamson(nx_graph)
        
        solutions = self.generate_good_solutions_from_GW(self.num_qubits, self.num_qubits, self.weights, self.graph_structure)
        correlator_matrix = self.get_correlator_matrix(solutions, self.qubits, circuit, self.symbols)
        i, j = get_max_element(correlator_matrix, self.num_qubits)
        self.graph_structure, self.weights, self.qubits = remove_node(self.num_qubits, self.graph_structure, self.weights, self.qubits, i)
        self.num_qubits  = self.num_qubits-1
        self.get_recursive_QAOA_solution(circuit)
    
    
    def get_QAOA_solution(self, qubits, initializing_soln_vector, step_p, symbols_gamma, symbols_beta):
            #preparing the pQC layer
            self.QAOA.get_readouts(qubits)
            self.QAOA.prepare_quantum_layer(qubits)
            
            #Initialising the state class 
            state_initializer = states(self.num_qubits,"Grover")

            #Generating the train data
            train_state, train_label = generate_training_data(state_initializer)
    
            print("-------------The training for QAOA begins inside a recursive QAOA-----------")
            self.QAOA.compile(optimizers = tf.keras.optimizers.Adam(learning_rate= lr), 
                                                                    loss = tf.keras.losses.MeanAbsolutError)

            history = self.QAOA.fit(x=train_state,y=train_label,
                        batch_size=Batch_Size,epochs=Num_Epochs,
                        verbose=0) 
            
            #parameters = self.QAOA.weights()
            return history["train_loss"][-1]


    def get_correlator_matrix(self, initial_solution, qubits, circuit, symbol):
        correlator_matrix = np.zeros((self.num_qubits,self.num_qubits))
        for i in range(self.num_qubits):
            for j in range(self.num_qubits):
                if i!=j:
                  correlator_matrix[i][j] = self.correlator.apply_unitary(qubits, circuit, i, j, symbol)
        
        return correlator_matrix





        
        



            

        


        

