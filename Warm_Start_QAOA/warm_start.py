import numpy as np
import cirq
from State import *
import cvxpy as cvx
from cvxopt import matrix, solvers
import networkx as nx
from typing import Tuple

"""
Class which contains the code for warm start. It contains the code to first find the classical solution for the Quadratic Convex program 
and then finding the corresponding theta for warm start.
"""

class warm_start:
    def __init__(self, num_qubits, qubits):
        self.state_class  = states(num_qubits, "warm start") 
        self.num_qubits = num_qubits
        self.qubits  = qubits
        
    
    def set_sigma_mu(self, sigma, mu):
        self.sigma  = sigma
        self.mu = mu
    

    def goemans_williamson(self, graph: nx.Graph) -> Tuple[np.ndarray, float, float]:
        
        """
        SDP Based Warm Start-----> for max cut 
        The Goemans-Williamson algorithm for solving the maxcut problem.
        Ref:
            Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
            algorithms for maximum cut and satisfiability problems using
            semidefinite programming. Journal of the ACM (JACM), 42(6), 1115-1145
        Returns:
            np.ndarray: Graph coloring (+/-1 for each node)
            float:      The GW score for this cut.
            float:      The GW bound from the SDP relaxation
        """
        # Kudos: Originally implementation by Nick Rubin, with refactoring and
        # cleanup by Jonathon Ward and Gavin E. Crooks
        laplacian = np.array(0.25 * nx.laplacian_matrix(graph).todense())

        # Setup and solve the GW semidefinite programming problem
        psd_mat = cvx.Variable(laplacian.shape, PSD=True)
        obj = cvx.Maximize(cvx.trace(laplacian * psd_mat))
        constraints = [cvx.diag(psd_mat) == 1]  # unit norm
        prob = cvx.Problem(obj, constraints)
        prob.solve(solver=cvx.CVXOPT)

        evals, evects = np.linalg.eigh(psd_mat.value)
        sdp_vectors = evects.T[evals > float(1.0E-6)].T

        # Bound from the SDP relaxation
        bound = np.trace(laplacian @ psd_mat.value)

        random_vector = np.random.randn(sdp_vectors.shape[1])
        random_vector /= np.linalg.norm(random_vector)
        colors = np.sign([vec @ random_vector for vec in sdp_vectors])
        score = colors @ laplacian @ colors.T

        return colors, score, bound

    def get_zero_one_constraint(self):
        """
        Generates the G and h matrices corresponding to various constraints: GX<H in a QP. Here the constraint is x belongs to [0,1].
        """
        g = [[0]*self.num_qubits]*(2*self.num_qubits)
        count = 0
        for i in range(self.num_qubits):
            g[i][i] = -1
            count+=1
        for i in range(self.num_qubits):
            g[count][i] = 1
        
        h = [0]*(2*self.num_qubits)
        for i in range(len(h)):
            if i<self.num_qubits:
                h[i] = 0
            else:
                h[i] = 1
        return g, h

    """
    QP Based warm start for portfolio optimisation where only positive semi definite matrices  are taken into account
    """
    def QP_based_warm_start(self, sigma):
        sigma = matrix(sigma)
        q  = [0]*self.num_qubits
        q = matrix(q)
        G, h = self.get_zero_one_constraint()
        G = matrix(G)
        h = matrix(h)
        sol = solvers.qp(sigma, q, G, h)

        return sol






    
        




            



        

        

