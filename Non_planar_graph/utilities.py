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
    #a = [[0]*7]*7
    a = np.zeros((7,7))
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

def check_valid_regular(N,K):
  if N%2==0:
    return K<=N-1 
  
  if K!=2 and K!=N-1:
      return False

  return True  

def create_regular_polygon(N, graph):
  for i in range(N):
    graph[i][(i+1)%N] = 1


def get_k_regular_graph(n = 22, k= 3):
    """
    returns a 3-regular graph in the form of a 2D list depticitng an adjacency matrix 
    """
    #a = [[0]*num_vertices]*num_vertices
    adj = np.zeros((n,n))
    if (n %2!=0) :
        if (k == 2): 
            for i in range(n):
                 r = (i + 1) % n
                 l = i-1
                 if i-1<0:
                  l = n-1
                 adj[i][l] = 1
                 adj[i][r] = 1
                 adj[l][i] = 1
                 adj[r][i] = 1
            
            print(adj)
            return
        elif (k == n-1):
            for i in range(n):
                for j in range(n):
                    if(i != j):
                        adj[i][j] = 1
                    
                
            
            print(adj)
            return
        else:
            print("not possible")
            return
        
    else :
        if(k < 1) :
            print("not possible")
            return
        else: 
            # making polygon
            for i in range(n): 
                 r = (i + 1) % n
                 l = i-1
                 if i-1<0:
                  l = n-1
                 adj[i][l] = 1
                 adj[i][r] = 1
                 adj[l][i] = 1
                 adj[r][i] = 1
            
            pp = k - 2
            mid = (n/2)
            mid = int(mid)
            for j in range(n): 
                 temp = pp
                 if (temp % 2 == 1): 
                    adj[(mid+j) % n][j] = 1
                    adj[j][(mid+j) % n] = 1
                    temp-=1
                
                 zz = int(temp/2)
                 for i in range(zz): 
                     r = (mid + j + i + 1) % n
                     l = (mid + j - i - 1) % n
                     #print(j)
                     
                     if (mid + j - i - 1) < 0:
                       l = n-1
                     
                     adj[j][int(l)] = 1
                     adj[j][int(r)] = 1
                     adj[int(l)][j] = 1
                     adj[int(r)][j] = 1
                
            
       
    return adj



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




    
        
        
