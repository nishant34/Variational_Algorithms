import tensorflow as tf 
import tensorflow_quantum as tfq
import argparse
import numpy as np
import cirq
import os
from model import *
from utilities import *
from config import *
from train_helper import *


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubits",help="The number of qubits for which the variational algrithm is to be ananlysed")
    parser.add_argument("--plot_loss_vs_num_qubits",help="to plot the loss obtained for various number of qubits")
    parser.add_argument("--analyse_noise",help="to analyse the effect of noise")
    parser.add_argument("--required_string",help="string to be found")
    parser.add_argument("--num_training_epochs",help="number of epoch to train")
    parser.add_argument("--len_alpha_beta",help="length of list of aplha and beta params for grover oprators")
    parser.add_argument("--alpha_constant",help = "decides if oracle is to be varied")
    parser.add_argument("--beta_constant",help="decides if diffusion operator is varied")


    args  = parser.parse_args()

    #getting index to be searched from the inpput string
    required_index = get_index_from_string(args.required_string)
    num_qubits= args.num_qubits

    #getting all types of models in a dictionary 
    step_p = args.len_alpha_beta
    model_dict = initialize_all_models(args.num_qubits, args.required_string, step_p)
    
    #Analysing the current selected case out of 4 possibe scenarios
    
    Model = model_dict["oracle_constant_diff_constant"]
    if args.alpha_constant and not args.beta_constant:
        Model = model_dict["oracle_constant_diff_varied"]
    elif not args.alpha_constant and args.beta_constant:
        Model = model_dict["oracle_varied_diff_constant"]
    elif  not args.alpha_constant and not args.beta_constant:   
        Model = model_dict["oracle_varied_diff_varied"]
    
    oracle = "constant"
    diffusion = "constant"
    if not args.alpha_constant:
        oracle = "varied"
    if not args.beta_constant:
        diffusion = "varied"

    #printing the useful information
    print("The learning rate for parametrs is:{}".format(lr))
    print("Number of qubits :{}".format(num_qubits))
    print("The size of search space is:{}".format(2**num_qubits))
    print("Number of angles for each operator:{}".format(args.len_alpha_beta))
    print("Current case is -> Oracle {} and Diffusion {} ".format(oracle,diffusion))
    print("The batch size is:{}".format(Batch_Size))
    print("The number of epochs will be:{}".format(Num_Epochs))
    print("---------------------The procedure begins-----------------")
    
    #Training the model---------
    
    #Initialising the state class 
    state_initializer = states(num_qubits,"Grover")

    #Generating the train data
    train_state, train_label = generate_training_data(state_initializer)
    
    #initializing the tensorboad-->
    callbacks = initialize_tensorboard_summary(tensorboard_log_path)

    #Curent_corrections-------> :)
    print("----------------The training begins-------------")
    Model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.02),
                                                        loss=tf.keras.losses.MeanAbsoluteError,
                                                        metrics=[custom_accuracy_metric])

    history = Model.fit(x=train_state,y=train_label,
                        batch_size=Batch_Size,epochs=Num_Epochs,
                        verbose=1)
    
    print("-------------Training_completed-------------------")
    
    print("saving the model after {} iters".format(Saving_step))
    save_model(model, model_save_path, Saving_step)
    
    #print("-------------Testing_Model------------------------")

    

    
    
    


    






    



    


    