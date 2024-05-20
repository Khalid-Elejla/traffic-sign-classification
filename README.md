## Traffic Sign Classifier
This repository contains a Jupyter Notebook that implements a traffic sign classifier using both TensorFlow and PyTorch. The notebook demonstrates how to build, train, and evaluate sequential models in both frameworks to classify traffic signs from images.


### Dataset
The dataset used in this notebook is the German Traffic Sign Recognition Benchmark (GTSRB). It contains images of traffic signs belonging to 43 different classes.


### TensorFlow Sequential Model
The TensorFlow model is built using the Sequential API. The architecture consists of several convolutional layers followed by max-pooling layers, dropout for regularization, and fully connected layers for classification.

### PyTorch Sequential Model
The PyTorch model is built using the torch.nn.Sequential module. It also consists of convolutional layers, max-pooling layers, dropout, and fully connected layers similar to the TensorFlow model.

### Training the Model
The notebook includes code for training both the TensorFlow and PyTorch models. The training process involves:

Loading and preprocessing the dataset.
Defining the model architecture.
Compiling the model (for TensorFlow).
Training the model on the training data.
Monitoring the training process using validation data.
Evaluation


### Usage
To use this notebook, simply open it in Jupyter Notebook or JupyterLab and run the cells in order. Make sure you have installed all the required dependencies.