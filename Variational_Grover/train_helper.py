import numpy as np
import tensorboard
from tqdm import tqdm
import scipy
#from model import *
import tensorflow as tf
import tensorflow_quantum as tfq
#from Variational_Grover_Model import *
from temp import *

def initialize_all_models(num_qubits, required_string, step_p):
    #returns a model dict containing all combinations of oracles and diffusion -->constant and varying
    models = {

    }
    models["oracle_constant_diff_not_equal"] = complete__model(num_qubits,required_string,step_p,True,False)
    models["oracle_constant_diff_equal"] = complete__model(num_qubits,required_string,step_p,True,True)
    models["oracle_not_constant_diff_equal"] = complete__model(num_qubits,required_string,step_p,False,True)
    models["oracle_varied_diff_varied"] = complete__model(num_qubits,required_string,step_p,False,False)
    
    return models
 
def load_model(model, load_path, load_iter):
    model_path = os.path.join(load_path+"model_"+str(load_iter))
    curr_model = tf.keras.models.load_model(model_path)
    return curr_model


def save_model(model, save_path, save_iter):
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

        











