import cirq
import numpy as np 
from utility import *
import pandas as pd
import numpy as np
import pandas_datareader.data as web
"""
contains classes to implement various tasks presented in the paper.
"""
def get_data_for_portfolio():
  data = web.DataReader(['BAC', 'GS', 'JPM', 'MS'],data_source="yahoo",start='12/01/2017',end='12/31/2017')['Adj Close']
  data.round(2)
  cov_matrix=data.cov()
  cov_matrix.round(2)
  data_1 = data.pct_change().apply(lambda x: np.log(1+x))
  mu = data_1.mean()
  return cov_matrix, mu

"""
Formulation for the Portfolio optimisation problem
"""
class portfolio_optimisation:
    def __init__(self, sigma, mu, q=1):
        self.q = q
        self.sigma = sigma
        self.mu = mu 
        self.maximise  = False

    def set_constraint(self, B):
        self.B = B

    def get_constraint_LHS(self, x):
        return np.matmul(np.transpose(np.ones_like(x)),x)

    def get_formulation(self, x):
        return np.matmul(np.matmul(np.matmul(self.q,np.transpose(x)), self.sigma), x) - np.dot(np.transpose(self.mu), x)


"""
Formulation for the MAX_CUT problem
"""

class MAX_CUT:
    def __init__(self, num_nodes, edge_set, weights, graph):
        self.num_nodes = num_nodes
        self.edge_set = edge_set
        self.weights = weights
        self.maximise = True
        self.graph = graph
    
    def get_max_cut_formulation(self, qubits):
        readouts = self.weights[0][1]*(1-cirq.Z(qubits[0])*cirq.Z(qubits[1]))
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if check_edge(i, j, self.edge_set) and i!=0 and j!=0:
                    readouts+= self.weights[i][j]*(1-cirq.Z(qubits[i])*cirq.Z(qubits[j]))
        return readouts


    
        