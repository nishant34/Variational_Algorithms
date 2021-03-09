import numpy as np
import cirq
import tensorflow_quantum as tfq
import tensorflow as tf
from Task import *
from warm_start import *
from utility import *

def get_task_dict(num_qubits, random_graph = False):
    sigma, mu = get_data_for_portfolio()
    graph = generate_graph(num_qubits, random_graph)
    weights  = get_weights(num_qubits, graph)
    edge_set = []
    edge_set_1 = []
    edge_set_2 = []
    for i in range(num_qubits):
      for j in range(num_qubits):
        if graph[i][j] == 1:
          edge_set_1.append(i)
          edge_set_2.append(j)
    edge_set.append(edge_set_1)      
    edge_set.append(edge_set_2)      
    target_task_dict = {
     
    }
    target_task_dict["protfolio"]= portfolio_optimisation(sigma, mu)
    target_task_dict["Max-Cut"] = MAX_CUT(num_qubits, edge_set, weights, graph)
    return target_task_dict

    

def get_cost_function(num_qubits, weights, qubits):
    cost_circuit = weights[0][1]*(cirq.Z(qubits[0]))*(cirq.Z(qubits[1]))
    for i in range(2, num_qubits):
        for j in range(i):
            cost_circuit += weights[j][i]*(cirq.Z(qubits[j]))*(cirq.Z(qubits[i]))
    
    return cost_circuit

def QUBO(x, sigma, mu):
    """
    Problem formulation for Quantum Unconstrained Binary Optimisation
    """
    return np.matmul(np.matmul(np.transpose(x),sigma), x) + np.dot(np.transpose(mu), x)


def QP(x, sigma, mu):
    """
    Problem formulation for Quadratic Prgoramme where sigma is postive semidefinite
    """
    return np.matmul(np.matmul(np.transpose(x),sigma), x)


#def get_optimized_state():


#def get_warm_start_dict(type):
    
def load_model(model_path, load_path, load_iter):
    model_path = os.path.join(load_path+"model_"+str(load_iter))
    curr_model = tf.keras.models.load_model(model_path)
    return curr_model


def save_model(model_path, save_path, save_iter):
    model_path = os.path.join(save_path+"model_"+str(save_iter))
    model.save(model_path)
    
   
def initialize_tensorboard_summary(log_dir):
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
    return tensorboard_callback

#@tf.function
def custom_accuracy_metric(y_true, y_pred):
    y_true = tf.squeeze(y_true)
    y_pred = tf.map_fn(lambda x: 1.0 if x >= 0 else -1.0, y_pred)
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true, y_pred))

def loss_function(model_output,actual=1):
    
    return actual-model_output


def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 90
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr



def generate_training_data(qubits):
    
    #to generate the inital hadamard state as the train example and prob_vale--> 1 as the label
    
    curr_circuit = cirq.Circuit()
    #state_class.get_hadamard_basis_state(curr_circuit)
    for qubit in qubits:
      curr_circuit.append(cirq.H(qubit))
    curr_tensor = tfq.convert_to_tensor([curr_circuit])
    #curr_tensor = curr_tensor[None,:]
    print(type(curr_tensor))
    labels = []
    labels.append(1)
    
    labels = np.array(labels)
    #labels = labels[None,:]
    
    return curr_tensor,labels


def generate_warm_started_training_data(qubits, warm_start_class, intermediate, task_class, epsilon):
    soln_vector = []
    if intermediate =="QP":
        soln_vector = warm_start_class.QP_based_warm_start(task_class.sigma)
    
    else:
        nx_graph = generate_graph_for_gw(task_class.num_nodes, task_class.weights, task_class.graph)
        soln_vector, score, _ = warm_start_class.goemans_williamson(nx_graph)
    
    angles = get_theta_from_classical_solutions(soln_vector, epsilon)
    warm_start_class.initial_angles = angles
    curr_circuit = cirq.Circuit()
    #state_class.get_hadamard_basis_state(curr_circuit)
    count = 0
    for qubit in qubits:
      curr_circuit.append(cirq.ry(angles[count])(qubit))
      count+=1
    curr_tensor = tfq.convert_to_tensor([curr_circuit])
    #curr_tensor = curr_tensor[None,:]
    print(type(curr_tensor))
    labels = []
    labels.append(10)
    
    labels = np.array(labels)
    #labels = labels[None,:]
    
    return curr_tensor,labels





