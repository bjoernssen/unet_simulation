# Geometric Deep Learning in comparison with U-Net

This is the repository accompanying the master thesis of Björn Przybyla 
"Geometric Deep Learning in Medical Image Segmentation and Comparisons with UNET".

## Prerequisites

This project is based on python 3.7. All necessary packages are listed in: 
```
requirements.txt
```
To run this repository, please run 
```
$ pip install -r requirements.txt
```
This will install all necessary packages. In case you have multiple python distributions installed,
e.g. a python 2 and a python 3 version, it may be necessary to run 
```
$ pip3 install -r requirements.txt
```
to specify your python distribution. Please make sure that the correct versions of the packages are isntalled,
since the dependencies may not be given for differing versions.

### Installing pytorch-geometric

In additon to these packages it might be necessary to install the pytorch-geometric package seperately.
To do this, run

```
$ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
$ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
$ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
$ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
$ pip install torch-geometric
```

where ${CUDA} and ${TORCH} should be replaced by your specific CUDA version (cpu, cu92, cu101, cu102) and PyTorch version (1.4.0, 1.5.0), respectively.
If no CUDA version is installed, replace it with cpu. To check the CUDA version run
```
$ nvcc --version
```

## Directory explanations

This part will give a short breackdown of the directories in the project folder.

### datasets

This folder contains a sorted version of the Tumor data set used in the thesis. The files are sorted
based on whether or not they contain a tumor and whether or not they are a mask/label. E.g.

```
datasets/Tumor_MRI/Yes/Image/TCGA_CS_4941_19960909_11.tif
datasets/Tumor_MRI/Yes/Mask/TCGA_CS_4941_19960909_11_mask.tif
```
are from the Tumor_MRI set, they contain a tumor, and are image and mask respectively.
### kaggle_3m

This directory contain the data set in its orinial order.

### mlruns
This directory contains the result of all experiments conducted during the implementation phase.
Each folder represents one experiment and contains the logged parameters and metrics of the experiment; e.g.
depth of the model and the loss during the training.

## models
This is an importable python package containing various models and functions used for the experiments.
```
models/datasets.py
```
contains the relevant classes and functions to create the Image and Graph data sets used. 
```
models/nets.py
```
contains the classes for building the neural networks used in this project; such as the U-Net models and their building blocks.

### Runable scripts
This directory contains the scripts that can be adapted to carry out the experiments. They are named
for their purpose, e.g.
```
Runable scripts/graph_simulation_run.py
```
can create graph sets, based on the dataset.py models and run them on the simulation dataset.
### utils
This importable python package contains various functions elemental to the project.
```
simulation.py
```
includes the functions to create the simulation data set.
```
loss.py
```
contains functions modelling th loss of the neural networks.
```
keypoint_functions.py
```
contains the functions that perform the graph creation, such as generating edges and measuring distances.
```
helper.py
```
includes several plotting methods.

### Results
To utilise the mlflow package used to track the results of the experiments, please make sure, that it is installed.
Run
```
$ pip install mlflow 
```
to install it. Afterwards run
```
$ mlflow ui
```
to start a web front end that can visualise the results.
## Built With

* [pytorch-geometric](https://github.com/rusty1s/pytorch_geometric) - The framework used for Geometric Deep Learning
* [pytorch](https://pytorch.org/) - The framework used for conventional deep learning
* [pytorch-unet](https://github.com/usuyama/pytorch-unet) - The simulation data set used for this project
* [mlflow](https://mlflow.org/) - The tool to track the results of the machine learning experiments
* [Google colab](https://colab.research.google.com/) - Hosting all experiment runs
