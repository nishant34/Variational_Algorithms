import cirq
import numpy as np
from random import randint
from skopt.plots import plot_convergence


"""
This contains the code for various utility functions to generate a custom device connectivity to test various possible cases along with the interface
for the ourense device described in the Nacl paper such as allowed gate list. 

"""



"""
Function to generate a custom device connectivity.
"""
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
    
"""
This is used to generate a custom allowed gate set for various qubits.
"""
def get_allowed_gate_list(num_qubits, random = False):
    sample_allowed_gates = ["RX","RY","RZ","1,2 CZ","2,3 CNOT","X","Y","Z","I","H"]
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

"""
a function to plot visualisation of the hyperparam search  
"""
def progress_visualization(search_result):
    
    plot_convergence(search_result)

"""
utility function to extract qubit indices form the multi qubit gate name.
"""
def extract_qubit_gates_from_multi_qubit(gate_name):
    ##The name of a multi qubit gate is of the folowing format--> 1,2 CNOT and extend it for n --> 1,2,3..,n CNOT
    components = gate_name.split(" ")
    qubit_indices = [int(index) for index in components[0].split(",")]
    gate_name = components[1]

    return qubit_indices, gate_name

"""
A function to generate all possible combination of n_qubit unitary matrices that can bve applied to a particular layer.
"""
def get_possible_unitaries(allowed_gate_list, num_qubits):
    
    if num_qubits==1:
        curr_combinations = [[gate] for gate in allowed_gate_list[0]]
        curr_unitary_combinations = curr_combinations
        #print(curr_combinations)
        return curr_unitary_combinations
    
    unitary_combinations = []
    
    prev_unitary_combinations = get_possible_unitaries(allowed_gate_list, num_qubits-1)
    #print("function_called")
    for gate in allowed_gate_list[num_qubits-1]:
        if gate[0].isdigit():
            qubit_indices, gate_name = extract_qubit_gates_from_multi_qubit(gate)
            curr_pass = False
            
            for index in qubit_indices:
              if index>num_qubits-1:
                  curr_pass = True
            
            if curr_pass:
                pass
            #required_qubits = qubits[qubit_indices]
            for possible_order in prev_unitary_combinations:
                include = True
                new_order = [gate1 for gate1 in possible_order ]
                
                new_order.append(gate)
                #print(gate_name)
                for index in qubit_indices[:-1]:
                  if possible_order[index-1]=="used" or possible_order[index-1][0].isdigit():
                    include=False
                  #print(index)
                  new_order[index-1] = "used"
                #new_order[qubit_indices[-1]-1] = gate
                if include:
                 unitary_combinations.append(new_order)
            
            
        else:
          for possible_order in prev_unitary_combinations:
            curr_order = []
            for gate_name in possible_order:
              curr_order.append(gate_name)
            curr_order.append(gate)
            unitary_combinations.append(curr_order)
    #print("in the possible_unitary function")
    #print(unitary_combinations)
    #print(len(unitary_combinations))
    unitary_combinations_final = [] 
    [unitary_combinations_final.append(x) for x in unitary_combinations if x not in unitary_combinations_final] 
    return unitary_combinations_final

"""
The gate list for the ourense device.
"""
def get_ourense_gate_list(num_qubits=5):
    gate_list_1 = [[ "RZ", "X"]]
    gate_list_1.append(["1,2 CNOT", "RZ", "X"])
    gate_list_1.append(["2,3 CNOT", "RZ"])
    gate_list_1.append(["2,4 CNOT", "RZ", "X"])
    gate_list_1.append(["4,5 CNOT", "RZ", "X"])
    return gate_list_1


"""
generating all the possible k_sequences to iterate over for the method described in Nacl paper.
Inputs--> number of possible unitary matrices for a given layer which will be generated given an allowed gate set using the fucntion described above.
Output--> All possible k_sequences to iterate over.
"""
def generate_possible_k_sequences(L_value, num_possible_unitaries):
  if L_value==1:
    k_values=  [[i] for i in range(num_possible_unitaries)]
    return k_values
  
  k_values_prev = generate_possible_k_sequences(L_value-1, num_possible_unitaries)
  k_values = []
  for seq in k_values_prev:
    for i in range(num_possible_unitaries):
      curr_value = [j for j in seq]
      curr_value.append(i)
      k_values.append(curr_value)
  
  return k_values
      

 
    
        
  





        





