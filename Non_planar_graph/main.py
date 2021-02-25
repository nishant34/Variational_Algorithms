import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import argparse
from utilities import *
from tasks import *
from model import *
from config import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_qubits", help = "The number of vertices/qubits in the probelm graph")
    parser.add_argument("--num_training_epochs", help = "The number of training epochs for the model")
    parser.add_argument("--len_beta_gamma", help = "The number of repitions of the operator")
    parser.add_argument("--target_problem", help="The probelm to be analysed out of the tree  given in the paper")
    parser.add_argument("--lr", help = "Learning rate for gradient descent")
    parser.add_argument("--device_name", help = "The planar processor used for simulation.")
    parser.add_argument("--plot_loss_landscape", help= "Whether to plot the loss heatmap or not and to analyze the decrement path.")

    args = parser.parse_args()

    
    #Setting up the connectivity graph according to the target probelm. It  will be in the adjacency matrix form.
    
    task = Task(args.num_qubits, args.device_name, args.target_problem)
    task.sample_task_weights()
    step_p = args.len_beta_gamma
    
    
    #initialising the model.........
    
    Model = model(args.num_qubits, args.target_problem, step_p, task.weights)

    #printing the useful information
    print("The learning rate for parametrs is:{}".format(lr))
    print("Number of qubits :{}".format(args.num_qubits))
    print("The size of search space is:{}".format(2**args.num_qubits))
    print("Number of angles for each operator:{}".format(args.len_beta_gamma))
    print("Current case is -> {} ".format(args.target_problem))
    print("The batch size is:{}".format(Batch_Size))
    print("The number of epochs will be:{}".format(Num_Epochs))
    print("---------------------The procedure begins-----------------")

    #Initialising the state class 
    state_initializer = states(args.num_qubits,"Grover")

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


    












        

    
