import cirq
import numpy as np
from random import randint
import os
import tensorflow as tf
import tensorflow_quantum as tfq

"""
This contains the helper function required to train the architecture search model to generate train data, save model, load model etc.
along with to compare the results of model with the circuit to prepare 5 qubit state given in Nacl paper.
"""
theta_1 = 0.785398
theta_2 = 2.356194 
theta_3 = 3.141593
theta_4 = 0.615480
theta_5 = 2.526113
theta_6 = 3.141593
theta_7 = 1.369438 
theta_8 = 0.785398 
theta_9 = 2.356194
theta_10 = 3.141593
theta_11 = np.pi/2

"""
To generate train data for the observable extraction where input is a stae and output is a numerical value.
"""
def get_train_data_for_observable_extraction(num_qubits,qubits,num_datapoints):
    train_data = []
    labels = []

    for i in range(num_datapoints):
        circuit = cirq.Circuit()
        #labels.append(1)
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
    
    #curr_tensor = tfq.convert_to_tensor(train_data)
    labels = np.array(labels)

    return train_data, labels


"""
utility funciton to generate train data having the 2-design states.
"""
def get_training_data_for_unitary_compilation_helper(num_qubits,qubits):
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
    
    #curr_tensor = tfq.convert_to_tensor(train_data)
    labels = np.array(labels)

    return train_data, labels

    

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




"""
Function to generate a random input train circuit.
"""
def get_input_circuit(num_qubits, qubits, circuit):
    ##getting the one qubit gates 
    #L_value = 15
    counter = 0
    for qubit in qubits:
        if counter%2==0:
         circuit.append(cirq.X(qubit))
        else :
            circuit.append(cirq.Z(qubit))
        
        if counter%3==0:
            circuit.append(cirq.rz(0.3)(qubit))
        else:
            circuit.append(cirq.rx(0.3)(qubit))    

    return circuit

"""
Function to generate train data for unitary compilation where data contains complete 2-design states along with the output states.
"""
def get_training_data_for_unitary_compilation(num_qubits, qubits):
    train_data, _ = get_training_data_for_unitary_compilation_helper(num_qubits, qubits)
    expectation_layer  = tfq.layers.Expectation()
    train_labels = []
    readouts = [cirq.Z(qubit) for qubit in qubits]
    for curr_data in train_data:
        curr_data = get_input_circuit(num_qubits, qubits, curr_data)
        curr_label = expectation_layer(curr_data, operators=readouts)
        curr_label = curr_label[0]
        #print("in the training_data gennerator")
        #print(type(curr_label))
        train_labels.append(curr_label)
    train_data_final,_ = get_training_data_for_unitary_compilation_helper(num_qubits, qubits)
    #curr_tensor = train_labels_1[0]
    #for curr_label in train_labels_1[1:]:
     # curr_tensor = tf.concat([curr_label,curr_tensor],0)
    train_labels = np.array(train_labels)
    train_data_1 = tfq.convert_to_tensor(train_data)
    #print(train_data_1.shape)
    #print(train_labels.shape)
    return train_data_final, train_labels

def train_RL_model(model, num_episodes, L_value):
   #max episode length will be the depth value
   #max_episodes: a hyper param
   optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
   model_pqc = model.create_continous_param_model(L_value)
        #model.set_readouts(self.category, self.qubits)
        #print(k_values)
   model.train_circuits = []
   train_state, train_labels = get_train_data_for_observable_extraction(model.num_qubits, model.qubits, 1)
   #for state in train_state:
   #     model.train_circuits.append(model.continous_param_model.prepare_circuit(state, model.qubits))
   model.train_circuits = train_state
   model.train_circuits = tfq.convert_to_tensor(model.train_circuits)
        
       
   callbacks = initialize_tensorboard_summary(log_dir=tensorboard_log_path)
    
   for i in range(num_episodes):
      
      with tf.GradientTape() as tape:
        sampled_k = model_pqc.sample(L_value)
        model.set_k_sequence(K_sequence)
        history = model_pqc.fit(x=model.train_circuits, y=model.train_label, epochs=Num_Epochs, batch_size=4, callbacks=[callbacks])
        curr_loss = history.history['loss'][-1]
        reinforce_loss = -tf.math.log(model_pqc.probs)*curr_loss
        
      grads = tape.gradient(curr_loss, model_pqc.complete_dense.trainable_weights)
      optimizer.apply_gradients(zip(grads, model_pqc.complete_dense.trainable_weights)) 
  
"""
function to generate the 5 qubit state given in the paper.
"""
def generate_train_data_for_state(num_qubits, qubits):
    
    train_data = []
    circuit = cirq.Circuit()
    circuit.append((cirq.X**theta_11)(qubits[0]))
    circuit.append((cirq.X**theta_11)(qubits[2]))
    circuit.append((cirq.X**theta_11)(qubits[3]))
    circuit.append((cirq.X**theta_11)(qubits[4]))
    circuit.append(cirq.I(qubits[1]))
    circuit.append((cirq.Z**theta_1)(qubits[0]))
    circuit.append((cirq.Z**theta_7)(qubits[2]))
    circuit.append((cirq.Z**theta_8)(qubits[4]))
    circuit.append(cirq.I(qubits[0]))
    circuit.append(cirq.I(qubits[1]))
    circuit.append(cirq.I(qubits[4]))
    circuit.append(cirq.I(qubits[0]))
    circuit.append(cirq.CNOT(qubits[2],qubits[1]))
    circuit.append(cirq.I(qubits[3]))
    circuit.append(cirq.I(qubits[4]))
    circuit.append(cirq.I(qubits[0]))
    circuit.append(cirq.CNOT(qubits[1],qubits[3]))
    circuit.append(cirq.I(qubits[2]))
    circuit.append(cirq.I(qubits[4]))
    circuit.append(cirq.I(qubits[0]))
    circuit.append(cirq.CNOT(qubits[2],qubits[1]))
    circuit.append(cirq.I(qubits[3]))
    circuit.append(cirq.I(qubits[4]))
    circuit.append(cirq.I(qubits[0]))
    circuit.append(cirq.CNOT(qubits[1],qubits[3]))
    circuit.append(cirq.I(qubits[2]))
    circuit.append(cirq.I(qubits[4]))
    circuit.append(cirq.I(qubits[0]))
    circuit.append(cirq.CNOT(qubits[3],qubits[4]))
    circuit.append(cirq.I(qubits[2]))
    circuit.append((cirq.X**theta_11)(qubits[1]))
    circuit.append((cirq.Z**theta_4)(qubits[1]))
    circuit.append((cirq.Z**theta_9)(qubits[1]))
    circuit.append(cirq.I(qubits[0]))
    circuit.append(cirq.CNOT(qubits[2],qubits[1]))
    circuit.append(cirq.I(qubits[3]))
    circuit.append((cirq.X**theta_11)(qubits[4]))
    circuit.append((cirq.Z**theta_5)(qubits[4]))
    circuit.append((cirq.Z**theta_10)(qubits[1]))
    circuit.append(cirq.I(qubits[0]))
    circuit.append(cirq.CNOT(qubits[4],qubits[3]))
    circuit.append(cirq.I(qubits[2]))
    circuit.append((cirq.X**theta_11)(qubits[1]))
    circuit.append((cirq.Z**theta_6)(qubits[1]))
    circuit.append(cirq.I(qubits[0]))
    circuit.append(cirq.CNOT(qubits[2],qubits[1]))
    circuit.append(cirq.I(qubits[3]))
    circuit.append(cirq.I(qubits[4]))
    circuit.append(cirq.I(qubits[2]))
    circuit.append(cirq.CNOT(qubits[1],qubits[0]))
    circuit.append(cirq.I(qubits[3]))
    circuit.append(cirq.I(qubits[4]))
    circuit.append((cirq.Z**theta_2)(qubits[0]))
    circuit.append((cirq.X**theta_11)(qubits[0]))
    circuit.append((cirq.Z**theta_3)(qubits[0]))
    circuit.append(cirq.I(qubits[1]))
    circuit.append(cirq.I(qubits[3]))
    circuit.append(cirq.I(qubits[4]))
    circuit.append(cirq.I(qubits[2]))
    circuit.append(cirq.I(qubits[3]))
    circuit.append(cirq.I(qubits[4]))
    circuit.append(cirq.I(qubits[2]))
    circuit.append(cirq.CNOT(qubits[0],qubits[1]))
    circuit_1 = cirq.Circuit()
    train_data.append(circuit_1)
    expectation_layer  = tfq.layers.Expectation()
    train_labels = []
    readouts = [cirq.Z(qubit) for qubit in qubits]
    curr_label = expectation_layer(train_data[0], operators=readouts)
    curr_label = curr_label[0]
    train_labels.append(curr_label)
    train_labels = np.array(train_labels)
    return train_data, train_labels
    
    

    
    
    
    
    
    
    
    







    
    

