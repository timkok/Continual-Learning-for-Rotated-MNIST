#  The Application of Continual Learning Methods to the Rotated MNIST Dataset
##### By: Tian Liao and Shreena Mehta
---
## Introduction
### Background
Humans and the way we think, learn, and grow are the primary inspiration for the field of machine learning. The advancement of deep neural networks has allowed for an advanced section of deep learning known as **continual learning**. Based on the idea of acquiring knowledge constantly, and adding to it while utilizing, sharing, and maintaining prior knowledge is what inspired this field of study. An often used dataset in this field is known as the Modified National Institute of Standards and Technology database (MNIST), which consists of 60000 training images and 10000 testing images of handwritten arabic numerals. While usually used for training and testing classifiers as is, the rotation of the images can pose a secondary challenge to those wanting to add another layer to their classification and identification models. 

### Purpose
Our purpose in applying GEM to the rotated MNIST dataset was to create a model using continual machine learning methods, such as mentioned in the *Continual Lifelong Learning with Neural Networks: A Review [1]* paper, in order to create a model which first "observes, once and one by one, examples concerning a sequence of tasks [2]" and subsequently uses these observations to allow growing accuracy of classification, and expand the the knowledge it holds without forgetting the knowledge it has already learned. This would allow us to relatedly mimic the way in which humans are constantly acquiring, processing, and transferring knowledge through their lifespans. In order to do this, we decided to implement the Gradient Episodic Memory model, which is an implementation of continual and supervised learning, both of which we discussed in class. In using this algorithm, our goal was to prioritize the accuracy and results of the memory modeling.

### Key Terms
- **Xavier Weight** - sometimes known as **Glorot Uniform**, draws a spread of weight distributions following a uniform curve
- **Perceptron** - a computational neuron used in supervised learning for classification
- **Convolutional Neural Network** - a Deep Learning algorithm which can take in an input image, assign importance (learnable weights and biases) to various aspects/objects in the image and be able to differentiate one from the other [6]
- **Residual Network** - an artificial neural network that may skip layers in the network in order to lessen the training needed
- **Confusion Matrix** - a table that consists of comparisons between predicted and actual outcomes of classifications and allows for the visualization of supervised learning performance
- **Backward transfer** - The average accuracy of each task i after each of the previous tasks are completed. <br />BWT = $\frac{1}{T - 1}$$\sum_{i=1}^{T - 1} R_{T,i} - R_{i,i}$ <br />
- **Forward transfer** - The average accuracy of each task i before each of the previous tasks are completed. <br />FWT = $\frac{1}{T - 1}$$\sum_{i=2}^{T} R_{i-1,i} - b_{i}$ <br />


## Setup
Most of the code is explained within the Colab Notebook's comments. Below are explained the key components of the code, and definitions of the key components of our implementation methods. 
#### Environment
We ran our Python code in Google Colab, importing from our shared Google Drive folder.
```
from google.colab import drive
drive.mount('/content/drive')

import os
for dirname, _, filenames in os.walk('/content/drive/My Drive/NJIT/CS675 Project I/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
##### Quadprog and Related Libraries
Quadprog was installed for aid in quadratic programming methods during the implementation of the GEM algorithm. Numpy and Pandas were imported for dataset manipulation on a surface level. Subprocess and Pickle were for management, multiprocessing, and error checking. The rest of the libraries were either related to os connection, image manipulation for the MNIST data, or for common use, such as the Random library. 
```
!pip install quadprog

import numpy as np 
import pandas as pd 
import subprocess
import pickle
import torch
import os
from torchvision import transforms
from PIL import Image
import argparse
import os.path
import random
```
##### Processing the MNIST Dataset
The MNIST data was downloaded as a .npz file and then uploaded onto our shared workspace drive. After loading it in, the `torch` library was used to split the data into training and testing sets, which were then saved in the drive as .pt files. 
```
f = np.load('/content/drive/My Drive/NJIT/CS675 Project I/mnist.npz')
x_tr = torch.from_numpy(f['x_train'])
y_tr = torch.from_numpy(f['y_train']).long()
x_te = torch.from_numpy(f['x_test'])
y_te = torch.from_numpy(f['y_test']).long()
f.close()
torch.save((x_tr, y_tr), '/content/drive/My Drive/NJIT/CS675 Project I/mnist_train.pt')
torch.save((x_te, y_te), '/content/drive/My Drive/NJIT/CS675 Project I/mnist_test.pt')
```
A function was made called `rotate_dataset()` that takes inputs of a dataset or subset, as well as a degree of rotation, and outputs a resulting rotated image. We then used `torch` to create 10 tasks which randomly rotates images in the training and testing sets, on angles from 0 to 90 degrees.

##### Modeling
A Xavier weight function was set up with a linear base, and a Multi-layer Perceptron class was created in conjunction. A 2-dimensional Convolutional Neural Network building function was also specified along with a class for building blocks in it. Finally, a ResNet structural class was built in order to make and manage layers in the Neural Network. 

After creating the structural functions and classes, we specified the metrics for evaluation in the form of a Confusion Matrix in order to determine the performance of our coming GEM classification model.

All of the related code is marked appropriately in the Colab Notebook.

## The GEM Algorithm
> "The main feature of GEM is an episodic memory **Mt**, which stores a
subset of the observed examples from task **t**"[2]

We begin our definition of the GEM algorithm by creating the appropriate functions and classes we need to perform each step. Below are detailed their specific names and what their intended purpose is:
- `compute_offets` - function to compute offsets for cifar to determine which outputs to select for a given task.
- `store_grad` - function that stores parameter gradients of past tasks.
- `overwrite_grad` - function used to overwrite the gradients with a new gradient vector, whenever violations occur.
- `project2cone2` - function that solves the GEM dual QP described in the paper [2] given a proposed gradient "gradient", and a memory of task gradients "memories"; Overwrites "gradient" with the final projected update.
- `Net` - a class with many functions: 
    - `__init__` - allocates episodic memory, temporary synaptic memory, and counters.
    - `forward` - makes sure classes within the current task are predicted.
    - `observe` - updates the memory and ring buffer to store examples from the current task, computes the gradient on previoius tasks and the current minibatch, and checks if the gradients violate the constraints.

Beyond these initial functions and classes, the previously split training and testing subsets are loaded from a function `load_datasets`. This is followed by the class `Continuum` which will be used later during the model evaluation in order to define the continuum of data. 

To evaluate the model's performance, the `eval_tasks` function was created, which takes in the model, its tasks, and a set of predefined arguments such as detailed below, and returns a list of performance ratings for every task in the model. This function is called by the `life_experience` function, which congregates the results of the predicted and actual outcomes of the model, in order to determine the actual results. Both of these rely on the following `args` :
```
args = {
    'model' : 'gem',
    'lr' : 0.1,
    'n_memories' : 256,
    'memory_strength' : 0.5,
    'seed' : 0,
    'cuda' : 'no',
    'finetune' : 'no',
    'batch_size' : 10,
    'shuffle_tasks' : 'no',
    'samples_per_task' : -1,
    'n_epochs' : 1,
    'n_layers' : 2,
    'n_hiddens' : 100,
    'data_file' : 'mnist_rotations.pt',
    'log_every' : 100
}

args['cuda'] = True if args['cuda'] == 'yes' else False
args['finetune'] = True if args['finetune'] == 'yes' else False
```

#### Parameters:

- number of hidden neurons at each layer (n_hiddens): 100
- number of hidden layers (n_layers): 2
- n_memories (number of memories per task): 256
- memory dependency: 0.5
- number of epochs per task (n_epochs): 1
- batch_size: 10
- learning rate (lr): 0.1

The `seed` component is used to initialize the `random` settings in a few library-derived functions:
```
torch.backends.cudnn.enabled = False
torch.manual_seed(args['seed'])
np.random.seed(args['seed'])
random.seed(args['seed'])
if args['cuda']:
        torch.cuda.manual_seed_all(args['seed'])
```

It is at this point that the `load_datasets` function from before is called and the resulting training vector is passed through the aforementioned `Continuum` class.  

After all of this, the metrics were evaluated through the forward transfer, backward transfer, and overall accuracy of the predicted versus actual results inputed in the Confusion Matrix throughout the program.

#### Parameters:
We calculate Average Accuracy on the test data, Backward transfer, Forward transfer. <br />
- Notation : <br />
R ∈ R<sup>T × T</sup> where R<sub>i,j</sub> is the test classification accuracy on task j after observing the last sample of task i.
- Average Accuracy : $\frac{1}{T}$$\sum_{i=1}^{T} R_{T,i}$

All results were saved in the workspace and are discussed in the next section.

## Results

We ran GEM on rotated MNIST datasets for 10 ratation tasks from 0 to 90 degree and from 0 to 180 degree. The results seem comparative for the two scenarios.

#### 10 ratation tasks from 0 to 90 degree :
- Final accuracy on the test dataset is 0.9182
- Minimized the negative backward transfer to -0.0557
- Achieved positive forward transfer to 0.7719

#### 10 ratation tasks from 0 to 180 degree :
- Final accuracy on the test dataset is 
- Minimized the negative backward transfer to -0.0557
- Achieved positive forward transfer to 0.7719

## Citations
1. https://arxiv.org/pdf/1802.07569.pdf
2. http://papers.nips.cc/paper/7225-gradient-episodic-memory-for-continual-learning.pdf
3. https://arxiv.org/pdf/1801.01423.pdf
4. https://github.com/facebookresearch/GradientEpisodicMemory
5. https://pantelis.github.io/cs677/docs/common/projects/continual-learning-CORe50/
6. https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
7. https://en.wikipedia.org/wiki/Confusion_matrix
