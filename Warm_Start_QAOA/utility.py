import numpy as np
import random
import math
import networkx as nx

def check_edge(p, q, edge_set):
    for i in range(len(edge_set[0])):
        for j in range(len(edge_set[1])):
            if edge_set[i] == p and edge_set[j] ==True:
                return True
    return False

def get_sum_except_given(weights, i, j):
    k = 0
    for  k in range(len(weights)):
        #for q in range(len(weights[p])):
            if k!=i and k!=j:
             k += weights[i][k]
    return k

def get_theta_from_classical_solutions(soln_vector, epsilon):
    """
    input: soln vector
    output: inverse sine values for each element
    """
    angles = 2*np.arcsin(soln_vector)
    for i in range(len(soln_vector)):
      if soln_vector[i]<=epsilon:
        angles[i] =  2*math.asin(math.sqrt(epsilon))
      elif soln_vector[i]>=1-epsilon:
        angles[i] =  2*math.asin(math.sqrt(1-epsilon))
      else:
        angles[i] =  2*math.asin(math.sqrt(soln_vector[i]))

    return angles


def generate_graph(num_nodes, ranom = False):
    """
    returns a grah in adjacency matrix form with 1 denoting the
    """
    graph = np.zeros((num_nodes,num_nodes))
    if random:
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i!=j:
                    a = random.randint(0,2)
                    if a==0:
                        graph[i][j] = 1
    
    else:
        for i in range(num_nodes):
            for j in range(i):
                        graph[i][j] = 1
    
    return graph 
  
def get_weights(num_nodes, graph):
  weights = np.zeros((num_nodes,num_nodes))
  for i in range(num_nodes):
    for j in range(num_nodes):
      if graph[i][j]==1:
        weights[i][j] = random.randint(-5,5)
  return weights 


def generate_graph_for_gw(num_nodes, weights, graph):
    graph_1 = nx.Graph()
    for i in range(num_nodes):
        for j in range(num_nodes):
            if(graph[i][j]==1):
                graph_1.add_edge(i,j,weight=weights[i][j])
    
    return graph_1 

def remove_node(num_nodes, graph_structure, weights, qubits, i):
    """
    Removes an edge i from the graph
    """
    new_num_nodes = num_nodes-1
    new_graph_structure = np.zeros((new_num_nodes,new_num_nodes))
    new_weights = np.zeros((new_num_nodes,new_num_nodes))
    r = 0
    t = 0
    for j in range(new_num_nodes):
        for k in range(new_num_nodes):
            if j!=k and j!=i and k!=i:
                if j<i:
                   r = j
                else:
                    r = j-1
                if k<i:
                    t = k
                else:
                    t = k-1
                new_graph_structure[r][t] = graph_structure[j][k]
                new_weights[r][t] = weights[j][k]
    qubits_new = []
    for k in range(len(qubits)):
        if k!=i:
            qubits_new.append(qubits[k])
    return new_graph_structure, new_weights, qubits_new
                     

"""
function to average multiple states
"""
def aggregrate_states(expectation_list):
    """
    expectation_list: 2D list to store expectation value of each qubit of each state in the standard basis
    """
    prob_values= []
    for j in range(len(expectation_list[0])):
        curr_prob = 0
        for i in range(len(expectation_list)):
            curr_prob += (1-expectation_list[i][j])/2
        curr_prob = curr_prob/len(expectation_list)
        prob_values.append(curr_prob)
    
    return prob_values

def get_max_element(matrix_1, n):
    curr_max = matrix_1[0][1]
    curr_i=  0
    curr_j = 1
    for i in range(n):
        for j in range(n):
            if matrix_1[i][j]>curr_max:
                curr_i = i
                curr_j = j
    return curr_i, curr_j


    















