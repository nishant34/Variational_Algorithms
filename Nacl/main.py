import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
from config import *
from Data_prepare import *
from Hardware import *
from Nacl_Model import *
from Gate_set import *
from State import *
from train_helper import *
from utility import *
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubits",help="The number of qubits for which the variational algrithm is to be ananlysed")
    parser.add_argument("--num_training_epochs",help="number of epoch to train")
    parser.add_argument("--max_depth",help="max allowed depth of the noisy circuit")
    parser.add_argument("--category",help="the category out of 3 possibleof the task to perform")
    parser.add_argument("--num_examples",help="Number of examples in the train set")
    parser.add_argument("--add_noise",help="whether to include gates or channels")
    parser.add_argument("--device_name",help="name for the  quantum device you need to include")
    parser.add_argument("--see_all_calls",help="whether to see all calls of a gaussian process optimixer")
    


    args = parser.parse_args()
    curr_task  = args.category
    num_qubits = args.num_qubits
    num_datapoints = args.num_examples

    print("The classical hyperparams are.......")
    print("The learning rate for parametrs is:{}".format(lr))
    print("Number of qubits :{}".format(num_qubits))
    print("The size of space is:{}".format(2**num_qubits))
    print("The batch size is:{}".format(Batch_Size))
    print("The number of epochs will be:{}".format(Num_Epochs))
    print("---------------------The procedure begins-----------------")

    #Initializing the state class....................
    state_initializer = states(num_qubits,"Nacl")

    #generating the training data for the current task 
    qubits = state_initializer.get_qubits()  
    train_state, train_labels = get_train_data_for_observable_extraction(num_qubits, qubits, num_datapoints)
    if curr_task == 1:
        train_state, train_labels = get_training_data_for_unitary_compilation(num_qubits, qubits)
    elif curr_task == 2:
        train_state, train_labels = get_training_data_for_unitary_compilation(num_qubits, qubits)

    #Getting the device parameters
    connectivity = sample_device_connectivity(num_qubits, random=True)
    allowed_gate_list = get_allowed_gate_list(num_qubits, random=True)
    
    #Initializing The device  class to caputre complete  properties of a given device
    curr_device = Device(allowed_gate_list, connectivity, num_qubits, "customised_device")

    if args.add_noise:
        ##doubtfull function
        curr_device.get_noise_model_for_gates()
    
    ##Initiliazing the complete wrapper model to optimize all of the parameters
    Model = complete_model(num_qubits, curr_device, curr_task, num_datapoints, args.max_depth)

    print("The  wrapper model has been initialized............")
    
    #generating the finite search space for discrete params
    Model.generate_discrete_search_space()
    
    #performing the optimization and best model will be saved and graphs will also be plotted in tensorboard
    search_result = Model.optimize_params()
    print("The optimization has been completed and the optimal paramters are......")
    print(search_result.x)
    
    if args.see_all_calls:
        sorted(zip(search_result.func_vals, search_result.x_iters))

    print("plotting visualisations for the loss w.r.t discrete set.....")
    progress_visualization(search_result)
    
    
    



    
    
    
    



    

    








    






