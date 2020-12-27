import cirq
import tensorflow_quantum as tfq
import tensorflow as tf
import numpy as np
from tf.keras.models import Model
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
import tf.keras.backend as K
from Gate_set import *


class Nacl_circuit:
    def __init__(self, num_qubits, allowed_gate_set, max_depth=15):
        #allowed gate set is 2-D list with 1st dimension determining the quibit and 2nd the allowed gates
        self.num_qubits = num_qubits
        self.allowed_gate_set = allowed_gate_set
        self.max_depth = max_depth
        self.gate_interface = custom_gate_set()
        self.gate_interface.set_gates()
    
    def manage_gate_interface(self, gate_name, qubits, circuit, symbol, index):
        if gate_name[0].isdigit():
            qubit_indices, gate_name = extract_qubit_gates_from_multi_qubit(gate_name)
            required_qubits = qubits[qubit_indices]
            if gate_name in self.gate_interface.continous_param_gate_list:
                self.gate_interface.add_gate_to_circuit(circuit,required_qubits,gate_name,len(qubit_indices),False,True,symbol)
            else:
                self.gate_interface.add_gate_to_circuit(circuit,required_qubits,gate_name,len(qubit_indices),False,False,symbol)
        else:
            if gate_name in self.gate_interface.continous_param_gate_list:
              self.gate_interface.add_gate_to_circuit(circuit,qubits[index],gate_name,1,False,True,symbol)
            else:
                self.gate_interface.add_gate_to_circuit(circuit,qubits[index],gate_name,1,False,False,symbol)

        
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
    

    def plot_circuit(self, circuit):
        
        SVGCircuit(circuit)

    

   
                     
class Nacl_procedure(Model):

    def __init__(self, num_qubits, noisy_device, category, max_allowed_depth = 15):
        self.num_qubits = num_qubits
        self.device_model = noisy_device
        self.state_class = states(num_qubits, "Nacl")
        total_params = max_allowed_depth*(num_qubits+num_qubits)+1
        self.all_symbols = states.get_params(num_params=total_params)
        ###self.L_values = self.all_symbols[0]
        ##k_values and theta values will be indexed accordingly with respect to 3D to 1D index conversion
        k_values_end_index = num_qubits*max_allowed_depth+1,
        ###self.k_values = self.all_symbols[1:k_values_end_index]
        self.theta_values  = self.all_symbols[k_values_end_index:total_params]
        self.category = category
        self.circuit_class = Nacl_circuit(num_qubits, noisy_device.get_allowed_gate_set(), max_allowed_depth)
        self.expectation_layer = tfq.layers.Expectation()


    def set_cost_function(self, cost_function):
        
        ###### set the custom cost function ######
        self.cost_function = cost_function
    

    ##def make_gates_maximally_parallel(self):

    def get_input_circuit(self, L_value, k_values, symbols, circuit, qubits):
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
        self.readouts = (cirq.Z(qubits[-1]))

        if category>0:
            self.readouts = self.input_circuit
        
    
    def set_hyperparams_value(self, L_value, k_values):
        self.hyperparam_L = L_value
        self.hyperparam_k = k_values

    def prepare_circuit(self, qubits):
        circuit = cirq.Circuit()
        circuit = self.circuit_class.prepare_complete_circuit(self.hyperparam_L, self.hyperparam_k, self.symbols, circuit, qubits)
        #self.curr_circuit = circuit
        return circuit
    

    def call(self, input_state):
        curr_circuit = self.prepare_circuit(input_state)
        output = self.expectation_layer(curr_circuit, self.theta_values, operators=self.readouts)
        return output


## complete class to wrap up the hyper parameter optimisation process:--->
## gaussian proceses will be used for hyper param optimisation
class complete_model:
    
    def __init__(self, num_qubits, noisy_device, category, num_datapoints, max_allowed_depth=15):
        self.continous_param_model = Nacl_procedure(num_qubits, noisy_device, category, max_allowed_depth)
        #self.L_values = self.continous_param_model.all_symbols[0]
        #self.k_values = self.continous_param_model.all_symbols[1:k_values_end_index]
        self.L_values_range = [i for i in max_allowed_depth]
        self.k_values_range = [len(noisy_device.allowed_gates[i]) for i in range(num_qubits)]
        self.dict_for_grid_search = dict(L_values = self.L_values_range, k_values = self.k_values_range)
        self.num_datapoints = num_datapoints
        self.num_qubits = num_qubits
        self.category = category
        self.max_depth = max_allowed_depth

    def create_continous_param_model(self, L_values, k_values):
        curr_model = self.continous_param_model
        curr_model.set_hyperparam_values(L_values, k_values)
        curr_model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.02),
                                                        loss=tf.keras.losses.MeanAbsoluteError,
        )
        return curr_model

    
    def execute_L_k_theta_optimisation(self, L_values, train_state, train_label, qubits, *k_values):
        
        model = self.create_continous_param_model(L_values, k_values)
        
        callbacks = tf.keras.callbacks.Tensorboard(
        log_dir=tensorboard_log_path,
        histogram_freq=0,
        write_graph=True,
        write_grads=False,
        write_images=False)

        #get the train data

        ##qubits = cirq.GridQubit.rect(1,self.num_qubits)
        ##train_state, train_label = get_train_data_for_observable_extraction(self.num_qubits, qubits, self.num_datapoints)
        
        ##if self.category==1:
          ##  train_state, train_label = get_training_data_for_unitary_compilation(self.num_qubits, qubits)
        
       ## elif self.category==2:
         ##    train_state, train_label = get_training_data_for_unitary_compilation(self.num_qubits, qubits)
        print("Training the model for continous params")
        history = model.fit(x=train_state, y=train_label, epochs=Num_Epochs, batch_size=4, callbacks=[callbacks])
        
        train_loss = history.history['train_loss'][-1]
        
        print("The train_loss is:{0:.4%}".format(train_loss))
        global best_train_loss

        if best_train_loss>train_loss:
            best_train_loss = train_loss
            ##saving the best model 
            save_model(model, model_save_path, Num_Epochs)
        
        del model
        K.clear_session()
        return train_loss
    
    def generate_discrete_search_space(self):
        self.k_default_list = [0]*self.num_qubits
        self.L_space = Integer(low=1, high=self.max_depth, name="L_values")
        self.k_space_list = [Integer(low=0,high=self.k_values_range[i],
                                    name="k_value_for"+str(i)+"th_qubit") 
                                    for i in range(self.num_qubits)]
        self.dimensions = [self.L_space]
        self.dimensions += self.k_space_list

        self.default_params = [1]
        self.default_params += self.k_default_list
    
    def optimize_params(self):
        search_result = gp_minimize(func=self.execute_L_k_theta_optimisation,
                            dimensions=self.dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=40,
                            x0=self.default_params)
        return search_result


                        



        
        


    
    
    



    
    






        

        

        
            
        





    

    
        
    


    



            

               



        

        
        
