# Machine Learning of noise resilient quantum circuits
* Implementation of Nacl model given in the following [paper](https://arxiv.org/pdf/1805.09337.pdf).
* The paper  proposes an idea of selecting architecture using iteration over a pool of gates along with varying the depth value of circuit and making the operations as parallel as possible.
# Differentiable quantum architecture search
* Along with the Nacl model a policy based quantum architetcure search approach by indexing unitary as a particular layer at a given depth approach similar to the one described in the paper differentiable quantum architeture search is also implemented.
* A Dense layer is used to sample k_sequence using the reinforce method.

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
* Gate_set.py --> It is used to create a custom gate set along with acting as an interface to add gates in a circuit given gate names or indices along with utility functions for gates.
* Hardware.py --> This contains the code to modify harware properties of a device such as noise using multiplicative model.
* Nacl_Model.py --> This contains the code for the Nacl model described in the paper along with the differentiable quantum architecture search approach based on Reinforce.
* config --> contions info related to paths and num_epochs.
* utility.py --> contains various utility functions along with functions.
* train_helper.py --> This contains the helper function required to train the architecture search model to generate train data, save model, load model etc.
* main.py --> To run the complete code by parsing the arguements required.

The Gradient.py file will be updated soon.

