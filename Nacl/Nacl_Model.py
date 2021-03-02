import cirq
import tensorflow_quantum as tfq
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
import skopt
from State import *
import sympy
from sklearn.model_selection import GridSearchCV
from skopt import gp_minimize
from config import *
import tensorboard
from train_helper import *
from cirq.contrib.svg import SVGCircuit
from skopt.space.space import Real, Integer, Categorical
import tensorflow.keras.backend as K
from Gate_set import *
from skopt.utils import use_named_args
#import tensorflow_probaility as tfp
from utility import *

"""
It contains the code for Nacl model along with approach similar to differentiable quantum architecture search using policy based gradients.
It contains circuit class to describe the basic circuit structure in cirq and then Nacl procedure class which acts as a wrapper to the circuit and connects to 
tensorflow quantum using the PQC layer.
"""
sequence_tester = []

"""
This class acts as a circuit interface for developing a circuit given depth values and gate lists. It has 2 kinds of fucntions. One uses a complete matrix of gates
varying in both dimensions -> gate_index and depth. The other is uniatry base circuit way where given gates available--> all possibble n-qubit unitary matrices are formed for a 
to act as a particular layer and then hyperparams are just depth value d and a list of len d --> [k1,.......kd] where ki is an index corresponding to unitary.

"""
class Nacl_circuit:
    def __init__(self, num_qubits, allowed_gate_set, noisy_device, max_depth=15):
        #allowed gate set is 2-D list with 1st dimension determining the quibit and 2nd the allowed gates
        self.num_qubits = num_qubits
        self.allowed_gate_set = allowed_gate_set
        self.max_depth = max_depth
        self.gate_interface = custom_gate_set()
        self.gate_interface.set_gates()
        self.noisy_device = noisy_device
        #print("in the Nacl_circuit initializer")
        #print(self.noisy_device)
        self.possible_unitaries = get_possible_unitaries(allowed_gate_set,num_qubits)
    
    """
     This funciton act as a gate interface to deal with the custom_gate_set class and is used in preparing the circuit given just the gate names and qubits.
    """
    def manage_gate_interface(self, gate_name, qubits, circuit, symbol, index):
        if gate_name[0].isdigit():
            qubit_indices, gate_name = extract_qubit_gates_from_multi_qubit(gate_name)
            required_qubits = [qubits[i-1] for i in qubit_indices]
            if gate_name in self.gate_interface.continous_param_gate_list:
                self.gate_interface.add_gate_to_circuit(circuit,required_qubits,gate_name,len(qubit_indices),False,True,symbol)
            else:
                self.gate_interface.add_gate_to_circuit(circuit,required_qubits,gate_name,len(qubit_indices),False,False,symbol)
            if self.noisy_device.noise_method==1:
                self.noisy_device.apply_manual_error_gates(gate_name,circuit,required_qubits,len(qubit_indices))

        else:
            if gate_name in self.gate_interface.continous_param_gate_list:
              self.gate_interface.add_gate_to_circuit(circuit,qubits[index],gate_name,1,False,True,symbol)
            else:
                self.gate_interface.add_gate_to_circuit(circuit,qubits[index],gate_name,1,False,False,symbol)
            if self.noisy_device.noise_method==1:
                self.noisy_device.apply_manual_error_gates(gate_name,circuit,qubits[index],1)

    """
    This function is used for preparing the complete circuit where gates for each qubi  are given for each timestep in a 2D form.
    Inputs--> depth value, gate indices, circuit and qubit_set
    Output--> circuit
    """
    def prepare_complete_circuit(self, L_value, k_values, symbols, circuit, qubits):
            #k-values is also 2D similar to allowed gate set
            ##Handling the controlled qubit case---->
            for i in range(L_value):
             for j,qubit in enumerate(qubits):
                gate_name = self.allowed_gate_set[k_values[j][i]]
                if gate_name[0].isdigit():
                    qubit_indices, gate_name = extract_qubit_gates_from_multi_qubit(gate_name)
                k_values[i][qubit_indices[:-1]] = -1 

            counter = 0
            for i in range(L_value):
                for j,qubit in enumerate(qubits):
                    if k_values[j][i]!=-1:
                    ##circuit.append(self.allowed_gate_set[k_values[j][i]](qubit))
                      gate_name = self.allowed_gate_set[k_values[j][i]]
                      self.manage_gate_interface(gate_name, qubits, circuit, symbols[counter], j)
                    counter+=1
            
            return circuit
    
    def prepare_unitary_base_circuit(self, k_sequence, symbols, circuit, qubits):
        
        """
        This is based on matrix for a particular layer described above and then just a list of d indices as the hyperparam along with depth value.
        """
         counter = 0
         for k_value in k_sequence:
             #print(k_value)
             curr_combination = self.possible_unitaries[int(k_value)]
             #print(curr_combination)
             curr_symbols = symbols[counter*self.num_qubits:(counter+1)*self.num_qubits]
             self.append_combination(circuit, curr_combination, qubits, curr_symbols)
             counter+=1
         
         return circuit

    def append_combination(self, circuit, curr_layer_list, qubits, symbols):
        """
        This is used to append a given combination of gates to the circuit which is desribed by a particular index in self.possible_unitaries.
        """
        index = 0
        #print("in the append combination")
        #print(len(curr_layer_list))
        for gate in curr_layer_list:
            if gate!= "used":
             self.manage_gate_interface(gate, qubits, circuit, symbols[index], index)
            index+=1


    def plot_circuit(self, circuit):
        
        SVGCircuit(circuit)
    
    
"""
This class uses the circuit class to build a circuit and then evaluate and train the continous params using back_prop by integrating with tensorflow quantum.
"""
class Nacl_procedure(Model):
    """
    In the initializer function continous_parameters(which later are converted to variables or weights) are initialised  along with discrete hyperparams.
    
    """
    def __init__(self, num_qubits, noisy_device, category, max_allowed_depth = 15):
        super(Nacl_procedure, self).__init__()
        self.num_qubits = num_qubits
        self.device_model = noisy_device
        self.state_class = states(num_qubits, "Nacl")
        total_params = max_allowed_depth*(num_qubits+num_qubits)+1
        self.all_symbols = self.state_class.get_params(num_params=total_params)
        ###self.L_values = self.all_symbols[0]
        ##k_values and theta values will be indexed accordingly with respect to 3D to 1D index conversion
        k_values_end_index = num_qubits*max_allowed_depth+1
        ###self.k_values = self.all_symbols[1:k_values_end_index]
        self.theta_values  = self.all_symbols[k_values_end_index:total_params]
        self.category = category
        self.circuit_class = Nacl_circuit(num_qubits, noisy_device.gate_set_list, noisy_device, max_allowed_depth)
        self.expectation_layer = tfq.layers.Expectation()
        if noisy_device.noise_method==2:
            #applying deploarizing channel at last
            self.expectation_layer =  tfq.layers.Expectation(backend=cirq.DensityMatrixSimulator(noise=cirq.depolarize(0.001)))
        self.k_space_shape = len(self.circuit_class.possible_unitaries)


    def set_cost_function(self, cost_function):
        
        ###### set the custom cost function ######
        self.cost_function = cost_function
    

    ##def make_gates_maximally_parallel(self):

    def get_input_circuit(self, L_value, k_values, symbols, circuit, qubits):
        """
        To get the input circuit for PQC_layer given all hyper_params and continous_params value.
        """
        self.input_circuit = cirq.Circuit()
        #depends if we need a device dependent input circuit or not-------------
        for i in range(L_value):
                for j,qubit in enumerate(qubits):
                    self.input_circuit.append(self.allowed_gate_set[k_values[j][i]](qubit))
        
        return self.input_circuit


    def get_training_data(self, train_examples, train_labels):
        self.train_inputs = train_examples
        self.train_outputs  = train_labels
    
    def set_readouts(self, category, qubits):
        """
        To set the readout bits for the pqc_layer.
        """
        self.qubits = qubits
        self.readouts = (cirq.Z(qubits[-1]))

        if category>0:
            self.readouts = [cirq.Z(qubit) for qubit in qubits]
        
    
    def set_hyperparams_value(self, L_value):
        self.hyperparam_L = L_value
        #self.hyperparam_k = k_values

    def prepare_circuit(self, qubits):
        """
        To prepare the circuit by sampling values using a dense layer similar to Reinforce method.
        """
        circuit = cirq.Circuit()
        #print("in he prepare circuit")
        #print(qubits)
        #circuit = cirq.Circuit()
        #self.hyperparam_L=10
        ####curr = 10
        #self.get_prob_layers(curr)
        k_sequences  = self.sample(curr)
        #print(k_sequences)
        #s#top
        circuit = self.circuit_class.prepare_unitary_base_circuit(k_sequences, self.theta_values, circuit, qubits)
        ##circuit = self.circuit_class.prepare_complete_circuit(self.hyperparam_L, self.hyperparam_k, self.symbols, circuit, qubits)
        #self.curr_circuit = circuit
        return circuit
    
    def prepare_circuit_1(self, qubits):
        """
        Preparing circuit by iterating iover all possible values in the k's space.
        """
        circuit = cirq.Circuit()
        #print("in he prepare circuit")
        #print(qubits)
        #circuit = cirq.Circuit()
        #self.hyperparam_L=10
        ####curr = 3
        #print(k_sequences)
        #s#top
        circuit = self.circuit_class.prepare_unitary_base_circuit(self.k_sequences_1, self.theta_values, circuit, qubits)
        ##circuit = self.circuit_class.prepare_complete_circuit(self.hyperparam_L, self.hyperparam_k, self.symbols, circuit, qubits)
        #self.curr_circuit = circuit
        return circuit

    def set_k_sequence(self, k_sequence):
      self.k_sequences_1 = k_sequence
    
    """
    To initialise the layers for policy function.
    """
    def get_prob_layers(self, depth_value):
        #p-->num_depth, c--> num_available_unitaries
        #self.classical_prob_layers = []
        #self.dense_layers = []
        #for i in range(depth_value):
         #   self.classical_prob_layers.append(tf.keras.layers.Softmax())
         #   self.dense_layers.append(tf.keraslayers.Dense(1,len(self.possible_unitaries)))
        self.complete_dense = tf.keras.layers.Dense(len(self.circuit_class.possible_unitaries))
        self.complete_softmax = tf.keras.layers.Softmax(axis=-1)

    
    """
    This is used for sampling list of indices of unitary matrices given a depth d using dense and softmax layers. This is simialr Deep RL.
    """
    def sample(self, depth_value):
        depth_value = 10
        temp_input = tf.convert_to_tensor(np.ones((depth_value, 1),dtype=np.float32))
        temp_output = self.complete_dense(temp_input)
        temp_output = self.complete_softmax(temp_output)
        #self.alpha_param =  temp_output    
        self.probs = temp_output
        #sampled_k --> batch_size,1 where batch_size = depth_value
        ####sampled_k = tf.random.categorical(self.probs, 1)
        sampled_k  = tf.math.argmax(self.probs,axis=-1)
        sequence_tester.append(sampled_k)
        #sampled_k = sampled_k.numpy()
        #print("sample is called")
        return sampled_k
    """
    Preparing the PQC layer for optimisation.
    """
    def prepare_quantum_layer(self, qubits):
        #self.pqc_layer = tfq.layers.PQC(self.prepare_circuit(qubits),self.readouts)
        self.pqc_layer = tfq.layers.PQC(self.prepare_circuit_1(qubits),self.readouts)
        
    
    def call(self, input_state):
        #curr_circuit = self.prepare_circuit(input_state[0], self.qubits)
        #output = self.expectation_layer(input_state[0], symbol_names=self.theta_values, operators=self.readouts)
        self.prepare_quantum_layer(self.qubits)
        #print("sample is called")
        output = self.pqc_layer(input_state[0])
        return output
    
    """
    Losses in the reinforce method--> 
    """
    def get_reinforce(self, y_true, y_pred):
        return -(tf.math.log(self.probs))*(tf.keras.losses.mean_absolute_error(y_true, y_pred))

    def get_policy_gradient_loss(self, y_true, y_pred):
        #return tf.math.log(self.probs)*(tf.math.reduce_sum(abs(y_pred-y_true)))
        #return -tf.math.log(self.probs)*(tf.keras.losses.mean_absolute_error(y_true, y_pred))
        #return [-tf.reduce_mean(tf.math.log(self.probs)),(tf.keras.losses.mean_absolute_error(y_true, y_pred))]
        return (tf.keras.losses.mean_absolute_error(y_true, y_pred))
    
    

    

""" 
complete_model class to wrap up the hyper parameter optimisation process. . 
"""

class complete_model:
    
    def __init__(self, num_qubits, noisy_device, category, num_datapoints, qubits, search_by_loop=False,  max_allowed_depth=15):
        """
        Initialising the training detail and pQC_layer model.
        """
        self.continous_param_model = Nacl_procedure(num_qubits, noisy_device, category, max_allowed_depth)
        #self.L_values = self.continous_param_model.all_symbols[0]
        #self.k_values = self.continous_param_model.all_symbols[1:k_values_end_index]
        self.L_values_range = [i for i in range(max_allowed_depth)]
        self.k_values_range = [len(noisy_device.gate_set_list[i]) for i in range(num_qubits)]
        self.dict_for_grid_search = dict(L_values = self.L_values_range, k_values = self.k_values_range)
        self.num_datapoints = num_datapoints
        self.num_qubits = num_qubits
        self.category = category
        self.max_depth = max_allowed_depth
        self.qubits = qubits
        self.train_loss = 10000
        self.search_by_loop = search_by_loop
    
    """
    To compile a tf.keras model as  the continous param model for a given L_value.
    """
    def create_continous_param_model(self, L_values):
        curr_model = self.continous_param_model
        curr_model.set_hyperparams_value(L_values)
        curr_model.set_readouts(self.category, self.qubits)
        curr_model.get_prob_layers(L_values)
        #curr_model.prepare_quantum_layer(self.qubits)
        #curr_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.02),
         #                                               loss=tf.keras.losses.MeanAbsoluteError,
        #)
        curr_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.02),
                                                  loss=[self.continous_param_model.get_policy_gradient_loss]
                                                        ,
                                                        run_eagerly=True,
        )
        return curr_model
      
    def set_train_data_points(self, train_state, train_label):
         self.train_state = train_state
         self.train_label = train_label

    """
    def execute_L_k_theta_optimisation(self, L_values):
        model = self.create_continous_param_model(L_values)
        #model.set_readouts(self.category, self.qubits)
        #print(k_values)
        self.train_circuits = []
        train_state, train_labels = generate_train_data_for_state(self.num_qubits, self.qubits, 1)
        #for state in train_state:
         #     self.train_circuits.append(self.continous_param_model.prepare_circuit(state, self.qubits))
        self.train_circuits = train_state
        self.train_circuits = tfq.convert_to_tensor(self.train_circuits)
        
        #callbacks = tf.keras.callbacks.Tensorboard(
        #log_dir=tensorboard_log_path,
        #histogram_freq=0,
        #write_graph=True,
        #write_grads=False,
        #write_images=False)
        callbacks = initialize_tensorboard_summary(log_dir=tensorboard_log_path)
        #get the train data

        ##qubits = cirq.GridQubit.rect(1,self.num_qubits)
        ##train_state, train_label = get_train_data_for_observable_extraction(self.num_qubits, qubits, self.num_datapoints)
        
        ##if self.category==1:
          ##  train_state, train_label = get_training_data_for_unitary_compilation(self.num_qubits, qubits)
        
       ## elif self.category==2:
         ##    train_state, train_label = get_training_data_for_unitary_compilation(self.num_qubits, qubits)
        print("Training the model for continous params")
        history = model.fit(x=self.train_circuits, y=self.train_label, epochs=Num_Epochs, batch_size=4, callbacks=[callbacks], verbose=1 if epoch % 100 == 0 else 0)
        
        train_loss = history.history['loss'][-1]
        #print(history.history.keys())
        
        print("The train_loss is:{0:.4%}".format(train_loss))
        global best_train_loss

        if self.train_loss>train_loss:
            self.train_loss = train_loss
            ##saving the best model 
            #save_model(model, model_save_path, Num_Epochs)
        
        del model
        K.clear_session()
        return train_loss
    """
    """
    To generate the discrete search space for hyper params which can be used in gp_minimize.
    """
    def generate_discrete_search_space(self):
        self.k_default_list = [0]*self.num_qubits
        self.L_space = Integer(low=1, high=self.max_depth, name="L_values")
        self.k_space_list = [Integer(low=0,high=self.k_values_range[i],
                                    name="k_value_for"+str(i)+"th_qubit") 
                                    for i in range(self.num_qubits)]
        self.dimensions = [self.L_space]
        #self.dimensions += self.k_space_list

        self.default_params = [1]
        #self.default_params += self.k_default_list
    
    """
    Using the gp_minimize inbuilt function in skopt over L_values. 
    """
    def optimize_params(self):
        curr_method = self.search_by_loop
        @use_named_args(dimensions=self.dimensions)
        def fitness_wrapper(*args, **kwargs):
             if curr_method:
               return self.execute_L_k_theta_optimisation_1(*args, **kwargs) 
               #return self.train_RL_model(*args, **kwargs)
             return self.train_RL_model(*args, **kwargs)
        search_result = gp_minimize(func=fitness_wrapper,
                            dimensions=self.dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=11,
                            x0=self.default_params)
        return search_result
    
    """
    Running the complete optimisation process which means optimising hyper params along with the continous params.
    """
    def execute_L_k_theta_optimisation_1(self, L_values):
        L_values = 10
        model = self.create_continous_param_model(L_values)
        #model.set_readouts(self.category, self.qubits)
        #print(k_values)
        self.train_circuits = []
        train_state, train_labels = generate_train_data_for_state(self.num_qubits, self.qubits)
        print("train data has been generated for sate prep task")
        #for state in train_state:
         #     self.train_circuits.append(self.continous_param_model.prepare_circuit(state, self.qubits))
        self.train_circuits = train_state
        self.train_circuits = tfq.convert_to_tensor(self.train_circuits)
        
       
        callbacks = initialize_tensorboard_summary(log_dir=tensorboard_log_path)
        #get the train data

        ##qubits = cirq.GridQubit.rect(1,self.num_qubits)
        ##train_state, train_label = get_train_data_for_observable_extraction(self.num_qubits, qubits, self.num_datapoints)
        
        ##if self.category==1:
          ##  train_state, train_label = get_training_data_for_unitary_compilation(self.num_qubits, qubits)
        
       ## elif self.category==2:
         ##    train_state, train_label = get_training_data_for_unitary_compilation(self.num_qubits, qubits)
        possible_k_sequences = generate_possible_k_sequences(3, len(self.continous_param_model.circuit_class.possible_unitaries))
        train_loss_curr = 1000000
        best_list = []
        for K_sequence in possible_k_sequences:
          #print(K_sequence)
          #stop
          print("Training the model for continous params for sequence:{}".format(K_sequence))
          model.set_k_sequence(K_sequence)

          history = model.fit(x=self.train_circuits, y=self.train_label, epochs=Num_Epochs, batch_size=4, callbacks=[callbacks], verbose=0)
          curr_loss = history.history['loss'][-1]
          print(curr_loss)
          K.clear_session()
          if curr_loss<train_loss_curr:
            train_loss_curr = curr_loss
            best_list.append(K_sequence)
          
        train_loss = train_loss_curr
        #print(history.history.keys())
        
        print("The train_loss is:{0:.4%}".format(train_loss))
        global best_train_loss

        if self.train_loss>train_loss:
            self.train_loss = train_loss
            ##saving the best model 
            #save_model(model, model_save_path, Num_Epochs)
        
        del model
        print(best_list[-1])
        print(train_loss_curr)
        print("just checking for the L_value=10")
        #stop
        return train_loss
    
    """
    To train the poliy based model for a given L_value which samples a K-sequence using dense layers and train for max_episode_length epoch.
    """
    def train_RL_model(self, L_values):
        num_episodes  = max_episode_length
        #max episode length will be the depth value
        #max_episodes: a hyper param
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        model_pqc = self.create_continous_param_model(L_values)
              #model.set_readouts(self.category, self.qubits)
              #print(k_values)
        self.train_circuits = []
        train_state, train_labels = generate_train_data_for_state(self.num_qubits, self.qubits, 1)
        #for state in train_state:
        #     model.train_circuits.append(model.continous_param_model.prepare_circuit(state, self.qubits))
        self.train_circuits = train_state
        self.train_circuits = tfq.convert_to_tensor(self.train_circuits)
              
            
        callbacks = initialize_tensorboard_summary(log_dir=tensorboard_log_path)
          
        for i in range(num_episodes):
            
            #with tf.GradientTape() as tape:
            sampled_k = model_pqc.sample(L_values)
             
            
            model_pqc.set_k_sequence(sampled_k)
            model_pqc.prepare_quantum_layer(self.qubits)
              #model_pqc.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.02),
               #                                   loss=[self.continous_param_model.get_policy_gradient_loss]
                #                                        ,
                 #                                       run_eagerly=True,
       # )
            history = model_pqc.fit(x=self.train_circuits, y=self.train_label, epochs=Num_Epochs, batch_size=4, callbacks=[callbacks], verbose=1 if epoch % 100 == 0 else 0)
            curr_loss = history.history['loss'][-1]
            with tf.GradientTape() as tape:
             sampled_k = model_pqc.sample(L_values)
             reinforce_loss = -tf.math.log(model_pqc.probs)*curr_loss
            grads = tape.gradient(reinforce_loss, model_pqc.complete_dense.trainable_weights) 
            
            optimizer.apply_gradients(zip(grads, model_pqc.complete_dense.trainable_weights))
                        
    
    

    
    

    
    
    
        

        
    
            



                        



        
        


    
    
    



    
    






        

        

        
            
        





    

    
        
    


    



            

               



        

        
        
