#import tensorflow as tf
import cirq
import numpy as np
from utilities import *
import random



"""
A class to realize the hardware properties for the tree problems given in the paper: Harware Grid, Three regular 
Max Cut and Sherrington-Kirkpatrick Model
"""
class Task:
    def __init__(self, num_qubits = 54, device_name = "Sycamore", connectivity = "Hardware Grid"):
        self.num_qubits = num_qubits
        self.device_name = device_name
        self.connectivity = connectivity
        """
        inititializing the 2D graph for depicitng connectivity among qubits.
        """   
        self.graph = []
        if connectivity=="Hardware Grid":
            self.graph =  get_hardware_grid_connectivity_connectivity()
        elif connectivity=="Max Cut 3-regular":
            self.graph =  get_k_regular_graph()
        else:
            self.graph =  get_fully_connected_graph()
        
        self.weights = [[0]*(len(self.graph))]*len(self.graph)
    
    def sample_task_weights(self):
        """
        To sample weights for a given graph
        """
        for i in range(len(self.graph)):
            for j in range(len(self.graph)):
                a = random.randint(0,1)
                if a==0:
                 self.weights[i][j] = 1
                else:
                    self.weights[i][j] = -1
    
    
    

    
        

#import tensorflow as tf
import cirq
import numpy as np
from utilities import *
import random



"""
A class to realize the hardware properties for the tree problems given in the paper: Harware Grid, Three regular 
Max Cut and Sherrington-Kirkpatrick Model
"""
class Task:
    def __init__(self, num_qubits = 54, device_name = "Sycamore", connectivity = "Hardware Grid"):
        self.num_qubits = num_qubits
        self.device_name = device_name
        self.connectivity = connectivity
        """
        inititializing the 2D graph for depicitng connectivity among qubits.
        """   
        self.graph = []
        if connectivity=="Hardware Grid":
            self.graph =  get_hardware_grid_connectivity_connectivity()
        elif connectivity=="Max Cut 3-regular":
            self.graph =  get_k_regular_graph()
        else:
            self.graph =  get_fully_connected_graph()
        
        self.weights = [[0]*(len(self.graph))]*len(self.graph)
    
    def sample_task_weights(self):
        """
        To sample weights for a given graph
        """
        for i in range(len(self.graph)):
            for j in range(len(self.graph)):
                a = random.randint(0,1)
                if a==0:
                 self.weights[i][j] = 1
                else:
                    self.weights[i][j] = -1
    
    
    

    
        

