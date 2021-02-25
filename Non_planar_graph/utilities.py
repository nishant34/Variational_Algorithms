import numpy as np 
import sympy 
import cirq


def check_unit_distance_on_a_grid(a,b):
    """
    To check if 2 points are at unit distance or not
    """
    return abs(a[0]-b[0])+(a[1]-b[1])==1

def get_hardware_grid_connectivity_connectivity(grid_size = 7):
    
    """
    return the connectivity of the hardware grid(a subgraph of device connectivity) as described in the paper
     a 2D list 'a' is returned where 1 depicts a particular location is there in the subgraph.
    """
    a = [[0]*7]*7
    for i in range(1,grid_size-1):
        a[i][i] = 1
    
    for i in range(1,grid_size-1):
        if a[i-1][i-1]==1 and a[i+1][i+1]==1:
            a[i][i+1] =  2
            a[i][i-1]  = 2
            a[i+1][i] = 2
            a[i-1][i] = 2

    for i in range(grid_size):
        for j in range(grid_size):
          if a[i][j] == 1:
            a[i][j+1] =  3
            a[i][j-1]  = 3
            a[i+1][j] = 3
            a[i-1][j] = 3
    
    for i in range(grid_size):
        for j in range(grid_size):
          if a[i][j] > 0:
              a[i][j] = 1
    
    return a

def get_degree(graph, v):
    """
    returns degree of a given vertex in a graph
    """
    count  = 0
    for i in range(len(graph[v])):
        if graph[v][i]>0:
            count+=1
    return count
 
def get_k_regular_graph(num_vertices = 22, k=3):
    """
    returns a 3-regular graph in the form of a 2D list depticitng an adjacency matrix 
    """
    a = [[0]*num_vertices]*num_vertices
    for i in range(num_vertices):
        counter = 1
        while get_degree(a, i)<k:
            a[i][i+counter] = 1
            a[i+counter][i] = 1
            counter+=1

    return a

def get_fully_connected_graph(num_vertices=17):
    """
    returns a fully connected graph in the form of a 2D list depticitng an adjacency matrix 
    """
    a = [[0]*num_vertices]*num_vertices
    for i in range(num_vertices):
        for j in range(num_vertices):
            if j!=i:
                a[i][j] = 1
    
    return a




    
        
        
