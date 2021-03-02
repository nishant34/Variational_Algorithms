# Quantum Approximate Optimization of Non-Planar Graph Problems on a Planar Superconducting Processor
* Implementation of Nacl model given in the following [paper](https://arxiv.org/abs/2004.04197).
* Basic Gates (only the ones supported by tensorflow quantum) have been used.
* Three structures provided in the paper: Hardware Grid, K-regular graph and Fully Connected have been analysed in the implementation.
* 2 sorts of unitary operators are given in the paer: The gamma unitary for the relative phase between 2 qubits and the mixer hamiltonian. 
* Swap operation has also been used for routing where graph connectivity is not a subragph of processor connectivity.

# Requirements
If running on a terminal Anaconda can be used to manage environments.

```javascript
conda create -n myenv
source activate myenv
```
The following packages will be useful for running the code:
* tensorflow 
* tensorflow-quantum
* numpy
* cirq
* sympy
* scipy

To install tensorflow quantum:
```javascript
 pip install -q tensorflow==2.3.1
 pip install -q tensorflow-quantum
 
```

# Code Structure
The code contaisn the following files:
* tasks.py --> Describe the tasks along with weight initialization for the three kinds of graphs provided in the paper.
* Operators.py --> Code for unitaries described in the paper.
* Model.py --> This contains the code for the complete wrapper model to merge cirq with tensorflow quantum.
* config --> contions info related to paths and num_epochs.
* utility.py --> contains various utility functions along with functions.
* train_helper.py --> This contains the helper function required to train the architecture search model to generate train data, save model, load model etc.
* main.py --> To run the complete code by parsing the arguements required.
* State.py






