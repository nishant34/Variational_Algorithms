import cirq
import tensorflow as tf
import tensorflow_quantum as tfq
import numpy as np
import os
import scipy


from State import *
from operators import *
#from utilities import *
from tensorflow.keras.models import Model
import math


class PQC_circuit_grover(object):
    def __init__(self, required_string, num_qubits, step_p):
        self.num_qubits = num_qubits
        self.required_string = required_string
        #self.required_index = get_index_from_string(required_string)
        

    #######Function to implement the grover diffusion operator using recursively implemented n-bit controlled,X and hadamard gates#########
    def circuit_diffusion(self, qubits, circuit, symbol):
      
      count = 0
      for qubit in qubits:
       #if curr_string[count]=="0":
       circuit.append(cirq.H(qubit))
       circuit.append(cirq.X(qubit))
       count+=1
      
      ######self.N_qubit_controlled_gate(qubits,symbol,count,circuit,1)
      ##############self.circuit_4_qubit(qubits,symbol,count,circuit)
      self.recursive_n_bit_controlled(circuit,symbol,qubits,count)
      
      
      count = 0
      for qubit in qubits:
       #if curr_string[count]=="0":
       circuit.append(cirq.X(qubit))
       circuit.append(cirq.H(qubit))
      count+=1
      return circuit    
    
    ########Oracle operator for variational grover circuit###########
    def circuit_oracle(self, qubits, circuit, curr_string, symbol):
       
       count = 0
       for qubit in qubits:
          if curr_string[count]=="0":
            circuit.append(cirq.X(qubit))
          count+=1
     
       
       #self.N_qubit_controlled_gate(qubits,symbol,count,circuit)
       #self.circuit_4_qubit(qubits,symbol,count,circuit)
       self.recursive_n_bit_controlled(circuit,symbol,qubits,count)
       #circuit.append((cirq.Z**symbol)(qubits[-1]).controlled_by(*qubits[:-1]))



       count = 0
       for qubit in qubits:
          if curr_string[count]=="0":
            circuit.append(cirq.X(qubit))
          count+=1
     
       return circuit 
    

    ####implementation of N-bit controlled rotation gate using ancilla bits#############
    def N_qubit_controlled_gate(self, qubits, angle, num_qubits, circuit,start=0):
       print("The angle is:{}".format(angle))
       #qubits = qubits1[:self.num_qubits]
       if start==1:
        ancilla = cirq.GridQubit.rect(1, num_qubits-2, 11, 11)
       else: 
        ancilla = cirq.GridQubit.rect(1, num_qubits-2, 7, 7)
       #ancilla = qubits1[self.num_qubits:]
       print(ancilla)
     
       self.Toffoli_circuit(qubits[0],qubits[1],ancilla[0],circuit)
     
       for i in range(2,num_qubits-1):
         self.Toffoli_circuit(qubits[i],ancilla[i-2],ancilla[i-1],circuit)
     
       #circuit.append((cirq.CX**angle)(ancilla[-1],qubits[-1]))
       ##self.get_rotation_from_CNOT(circuit, ancilla[-1], qubits[-1], angle)
       self.get_controlled_rot(circuit, ancilla[-1], qubits[-1], angle)

     
       for i in reversed(range(2,num_qubits-1)):
         self.Toffoli_circuit(qubits[i],ancilla[i-2],ancilla[i-1],circuit)
     
       self.Toffoli_circuit(qubits[0],qubits[1],ancilla[0],circuit)
    
    #######implementing 4 bit controlled rotation gate with basic gates and without any recursion################
    def circuit_4_qubit(self,qubits,angle,num_qubits,circuit):
         
         angle = angle/4
         circuit.append(cirq.CZ(*[qubits[0],qubits[-1]])**angle)
         circuit.append(cirq.CNOT(qubits[0],qubits[1]))
         circuit.append(cirq.CZ(*[qubits[1],qubits[-1]])**-angle)
         circuit.append(cirq.CNOT(qubits[0],qubits[1]))
         circuit.append(cirq.CZ(*[qubits[1],qubits[-1]])**angle)
         circuit.append(cirq.CNOT(qubits[1],qubits[2]))
         circuit.append(cirq.CZ(*[qubits[2],qubits[-1]])**-angle)
         circuit.append(cirq.CNOT(qubits[0],qubits[2]))
         circuit.append(cirq.CZ(*[qubits[2],qubits[-1]])**angle)
         circuit.append(cirq.CNOT(qubits[1],qubits[2]))
         circuit.append(cirq.CZ(*[qubits[2],qubits[-1]])**-angle)
         circuit.append(cirq.CNOT(qubits[0],qubits[2]))
         circuit.append(cirq.CZ(*[qubits[2],qubits[-1]])**angle)
     
    ##**********************Recursive way to implement N-bit Controlled Rotation gate without ancilla bits and using basic gates supported by tfq******##
    def recursive_n_bit_controlled(self,circuit,angle,qubits,num_qubits):
        if len(qubits)==2:
           circuit.append(cirq.CZ(*[qubits[0],qubits[1]])**angle)
           return 
        angle1 = angle/2

        #helper_n_qubit(circuit,angle1,qubits,num_qubiits)
        circuit.append(cirq.CZ(*[qubits[-2],qubits[-1]])**angle1)
        self.recursive_CNOT(circuit,1,qubits[:-1],num_qubits-1)
        circuit.append(cirq.CZ(*[qubits[-2],qubits[-1]])**-angle1)
        self.recursive_CNOT(circuit,1,qubits[:-1],num_qubits-1)
    
        qubits_new = qubits[:num_qubits-2]
        qubits_new.append(qubits[-1])

        self.recursive_n_bit_controlled(circuit,angle1,qubits_new,num_qubits-1)
    

    ##**********************Recursive way to implement N-bit Controlled CNOT gate without ancilla bits and using basic gates supported by tfq******##
    def recursive_CNOT(self,circuit,angle,qubits,num_qubits):
       if len(qubits)==2:
        circuit.append(cirq.H(qubits[1]))
        circuit.append(cirq.CZ(*[qubits[0],qubits[1]])**angle)
        circuit.append(cirq.H(qubits[1]))
        return 
       angle1 = angle/2
    
       #helper_n_qubit(circuit,angle1,qubits,num_qubiits)
       circuit.append(cirq.H(qubits[-1]))
       circuit.append(cirq.CZ(*[qubits[-2],qubits[-1]])**angle1)
       circuit.append(cirq.H(qubits[-1]))
       self.recursive_CNOT(circuit,1,qubits[:-1],num_qubits-1)
       circuit.append(cirq.H(qubits[-1]))
       circuit.append(cirq.CZ(*[qubits[-2],qubits[-1]])**-angle1)
       circuit.append(cirq.H(qubits[-1]))
       self.recursive_CNOT(circuit,1,qubits[:-1],num_qubits-1)

       qubits_new = qubits[:num_qubits-2]
       qubits_new.append(qubits[-1])

       self.recursive_CNOT(circuit,angle1,qubits_new,num_qubits-1)

    
    ########Implementation of TOFFOLI gates by basic gates supported by tfq#########
    def Toffoli_circuit(self,qubit_0,qubit_1,qubit_2,circuit):
        
        circuit.append(cirq.H(qubit_2))
        circuit.append(cirq.CNOT(qubit_1,qubit_2))
        #circuit.append(cirq.rz(-np.pi/4)(qubit_2))
        circuit.append((cirq.Z**-0.25)(qubit_2))
        circuit.append(cirq.CNOT(qubit_0,qubit_2))
        #circuit.append(cirq.rz(np.pi/4)(qubit_2))
        circuit.append((cirq.Z**0.25)(qubit_2))
        circuit.append(cirq.CNOT(qubit_1,qubit_2))
        #circuit.append(cirq.rz(-np.pi/4)(qubit_2))
        circuit.append((cirq.Z**-0.25)(qubit_2))
        circuit.append(cirq.CNOT(qubit_0,qubit_2))
        #circuit.append(cirq.rz(np.pi/4)(qubit_1))
        circuit.append((cirq.Z**0.25)(qubit_1))
        #circuit.append(cirq.rz(np.pi/4)(qubit_2))
        circuit.append((cirq.Z**0.25)(qubit_2))
        circuit.append(cirq.CNOT(qubit_0,qubit_1))
        circuit.append(cirq.H(qubit_2))
        #circuit.append(cirq.rz(-np.pi/4)(qubit_1))
        circuit.append((cirq.Z**-0.25)(qubit_1))
        circuit.append(cirq.CNOT(qubit_0,qubit_1))
        #circuit.append(cirq.rz(np.pi/4)(qubit_0))
        circuit.append((cirq.Z**0.25)(qubit_0))


    ####controlled rotation gate######
    def get_controlled_rot(self,circuit,qubit_0,qubit_1,angle):
  
      circuit.append(cirq.CZ(*[qubit_0,qubit_1])**angle)
    
    
######Defining the complete rapper  model for the variational grover circuit########
class complete__model(Model):
    
    def __init__(self, num_qubits, required_string, step_p, alpha_constant=False, beta_equal=False):
        super(complete__model, self).__init__()
        self.num_qubits = num_qubits
        self.required_string = required_string
        self.step_p = step_p
        self.circuit_class  = PQC_circuit_grover(required_string, num_qubits, step_p)
        self.state_class = states(num_qubits,"Grover")
        self.symbols_alpha = self.state_class.get_params(2*step_p)
        print(self.symbols_alpha)
        #self.symbols_beta = self.state_class.get_params(step_p)
        self.alpha_constant  = alpha_constant
        self.beta_equal = beta_equal
        self.state_layer = tfq.layers.State()

    
    def get_readouts(self, qubits):
      self.readouts_1 = [cirq.Z(bit) for bit in qubits]   
      self.readouts = (1+cirq.Z(qubits[0]))*0.5
      count = 0
      for curr_str in self.required_string:
        if count==0:
          if curr_str=="1":
               
           a=2 
           #self.readouts = (1-cirq.Z(qubits[count]))*0.5
        else:
          if curr_str=="1":
            self.readouts *= (1-cirq.Z(qubits[count]))*0.5
          else:
            self.readouts *= (1+cirq.Z(qubits[count]))*0.5
        count+=1


    def prepare_dummy_circuit(self, qubits):
        
        circuit = cirq.Circuit()
      
        count = 0
        if not self.alpha_constant and not self.beta_equal:
          for i in range(self.step_p):
             self.circuit_class.circuit_oracle(qubits, circuit, self.required_string, self.symbols_alpha[count])
             self.circuit_class.circuit_diffusion(qubits, circuit, self.symbols_alpha[count+1])
             count+=2
        
        elif not self.alpha_constant and self.beta_equal:
            count = 0
            for i in range(self.step_p):
             self.circuit_class.circuit_oracle(qubits, circuit, self.required_string, self.symbols_alpha[count])
             self.circuit_class.circuit_diffusion(qubits, circuit, self.symbols_alpha[count])
             
        elif self.alpha_constant and self.beta_equal:
            count = 0
            for i in range(self.step_p):
             self.circuit_class.circuit_oracle(qubits, circuit, self.required_string, 1)
             self.circuit_class.circuit_diffusion(qubits, circuit, self.symbols_alpha[count])
        
        else:
            count = 0
            for i in range(self.step_p):
             self.circuit_class.circuit_oracle(qubits, circuit, self.required_string, 1)
             self.circuit_class.circuit_diffusion(qubits, circuit, self.symbols_alpha[count])
             count+=1 


        self.dummy_circuit = circuit
        return circuit
    
    

    
    def prepare_quantum_layer(self, qubits):
        
        
        self.pqc_layer = tfq.layers.PQC(self.prepare_dummy_circuit(qubits),self.readouts)
        #self.pqc_layer_1 = tfq.layers.PQC(self.prepare_dummy_circuit(qubits),self.readouts_1)
        
    

    def call(self, input_state):
       
       output  = self.pqc_layer(input_state[0])
       
       
       
       return output
