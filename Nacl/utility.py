import cirq
import numpy as np
from random import randint
from skopt.plots import plot_convergence

def sample_device_connectivity(num_qubits, random=False):
    connectivity = []
    
    if random==False:
     for i in range(num_qubits):
        curr_list = []
        curr_list.append(i+1)
        curr_list.append(i-1)
        if i%2==0:
            curr_list.append(i-2)
        if i-3>0:
            curr_list.append(i-3)
        connectivity.append(curr_list)
    else:
        for i in range(num_qubits):
         curr_list = []
         for j in range(num_qubits):
             if j!=i:
                 curr = randint(0,8)
                 curr_1 = randint(0,10)
                 if curr>curr_1:
                     curr_list.append(j)
         connectivity.append(curr_list)
    return connectivity
    

def get_allowed_gate_list(num_qubits, random = False):
    sample_allowed_gates = ["RX","RY","RZ","CZ","CNOT","X","Y","Z","I","H"]
    allowed_gate_list = []
    for i in range(num_qubits):
        curr_gate_list = []
        if random==False:
         if i%2==0:
             curr_gate_list.append("RX")
         else:
             curr_gate_list.append("RY")
         if i<4:
             curr_gate_list.append("RZ")
         if i>3:
             curr_gate_list.append("X")
         else:
             curr_gate_list.append("X")
         curr_gate_list.append("I")
    
        else:
            for j in range(len(sample_allowed_gates)):
                curr = randint(0,8)
                curr_1 = randint(0,10)
                if curr>curr_1 :
                    curr_gate_list.append(sample_allowed_gates[j])
        
        allowed_gate_list.append(curr_gate_list)
    return allowed_gate_list

##a function to plot visualisation of thehyperparam search  
def progress_visualization(search_result):
    
    plot_convergence(search_result)

def extract_qubit_gates_from_multi_qubit(gate_name):
    ##The name of a multi qubit gate is of the folowing format--> 1,2 CNOT and extend it for n --> 1,2,3..,n CNOT
    components = gate_name.split(" ")
    qubit_indices = [int(index) for index in components[0].split(",")]
    gate_name = components[1]

    return qubit_indices, gate_name






