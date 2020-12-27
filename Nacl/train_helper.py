import cirq
import numpy as np
from random import randint
import os
import tensorflow as tf
import tensorflow_quantum as tfq


def get_train_data_for_observable_extraction(num_qubits,qubits,num_datapoints):
    train_data = []
    labels = []

    for i in range(num_datapoints):
        circuit = cirq.Circuit()
        labels.append(1)
        for j in range(num_qubits):
            curr = randint(0,8)
            curr_1 = randint(0,10)
            if curr>curr_1:
                circuit.append(cirq.H(qubits[j]))
                circuit.append(cirq.X(qubits[j]))
            
            else:
                circuit.append(cirq.H(qubits[j]))
                circuit.append(cirq.Z(qubits[j]))

            if curr<2*curr_1:
                if j<num_qubits-1:
                 circuit.append(cirq.CNOT(qubits[j],qubits[j+1]))
                 circuit.append(cirq.Y(qubits[j]))
                
            else:
                if j>0:
                 circuit.append(cirq.CNOT(qubits[j],qubits[j-1]))
                 circuit.append(cirq.Z(qubits[j]))
                
            if curr>3:
                
                circuit.append(cirq.rz(np.pi/6)(qubits[j]))

            if curr_1>4:
                    circuit.append(cirq.rz(np.pi/4)(qubits[j]))
        train_data.append(circuit)
        labels.append(1)
    
    curr_tensor = tfq.convert_to_tensor(train_data)
    labels = np.array(labels)

    return curr_tensor, labels


#utility funciton to generate train data having the 2-design states
def get_training_data_for_unitary_compilation(num_qubits,qubits):
    #handle the tensor conversion
    train_data = []
    labels = []
    for i in range(2**num_qubits):
        circuit = cirq.Circuit()
        k = i
        counter = 0
        while k>0:
            if k%2==1:
                circuit.append(cirq.X(qubits[-counter]))
            
            k = k/2
            k = int(k)
            counter+=1

        train_data.append(circuit)
        labels.append(1)
    
    curr_tensor = tfq.convert_to_tensor(train_data)
    labels = np.array(labels)

    return curr_tensor, labels

    

#def get_training_data_for_state_compilation(num_qubits,qubits):
    ##### how to get random states????????????





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



#def custom_cost_function():

##class Custom_Quantum_Natrual_Gradient():

    