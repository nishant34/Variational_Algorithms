import os 
import sys
import matplotlib.pyplot as plt
import scipy
import random
import numpy as np
import seaborn as sns
import tensorflow as tf


def reverse(string): 
    string = string[::-1] 
    return string 

def get_index_from_string(curr_string):
    curr_string_rev = reverse(curr_string)
    index = 0
    for i in range(len(curr_string_rev)):
        index += int(curr_string_rev[i])*(2**i)

    return index  


def generate_random_string(required_length):
    curr_str = ""

    for i in range(required_length):
        curr = random.randint(0,1)
        curr_str+= str(curr)
    
    return curr_str

def plot_noise_prob_heatmap(A1,A2,z):
    xlabels = ['{:3.1f}'.format(x) for x in A1]
    ylabels = ['{:3.1f}'.format(y) for y in A2]
    ax = sns.heatmap(z, xticklabels=A1, yticklabels=A2)
    ax.set_xticks(ax.get_xticks()[::3])
    ax.set_xticklabels(xlabels[::3])
    ax.set_yticks(ax.get_yticks()[::3])
    ax.set_yticklabels(ylabels[::3])

def plot_heatmap_from_array(curr_arr):
    #curr_arr = np.concatenate((T1_noise,T2_noise,prob_values),axis=0)
    plt.imshow(curr_arr,cmap='viridis')
    plt.colorbar()
    plt.show()


def plot_keras_model(model):
    tf.keras.utils.plot_model(model,  show_shapes =True, dpi=70)

def plot_variatonal_vs_normal_grover(output_probs, grover_probs, max_number_of_qubits):
    num_qubit_axis = [i+1 for i in range(max_number_of_qubits)]
    plt.plot(num_qubit_axis,output_probs)
    plt.plot(num_qubit_axis,grover_probs)

    plt.legend(["Variational_Grover","Grover"])

def generate_percentage_increase_between_highst_prob_table(output_probs, grover_probs,  num_qubit_list, step_pmax, angle_list):
    num_rows = len(output_probs)
    percentage_increase_list = [(100*(output_probs[i]-grover_probs[i])/grover_probs[i]) for i in range(num_rows)]
    curr_table = [[2**num_qubit_list[i],percentage_increase_list[i],step_pmax[i],angle_list[i]] for i in range(num_rows)]
    curr_table.insert(0,["N", "100 × (Pvariational − PGrover)/PGrove", "step pmax", "angle"])
    for a,b,c,d in zip(*curr_table):
        print(a,b,c,d)


    



if __name__=="__main__":
    str1 = "1100"
    print(generate_random_string(4))
    a  = np.random.randn(10)
    b = np.random.randn(10)
    prob = np.random.randn(10,1)
    plot_noise_prob_heatmap(a,b,prob)