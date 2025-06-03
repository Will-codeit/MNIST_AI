# mnist_loader.py  
# This script is designed to load the MNIST dataset, a popular dataset for training and testing neural networks.
# The MNIST dataset consists of images of handwritten digits (0-9) and their corresponding labels.
# This script provides functions to load the images and labels from the dataset files.

import numpy as np  # NumPy is used for numerical operations and handling arrays.
import os  # The os module is used to handle file paths.
import struct  # The struct module is used to unpack binary data from the dataset files.

# Function to load images from the MNIST dataset
def load_images(filename):
    with open(filename, 'rb') as f:
        # The first 16 bytes contain metadata: magic number, number of images, rows, and columns.
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the remaining bytes as image data and reshape it into (num_images, rows, cols).
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images

# Function to load labels from the MNIST dataset
def load_labels(filename):
    with open(filename, 'rb') as f:
        # The first 8 bytes contain metadata: magic number and number of labels.
        magic, num = struct.unpack(">II", f.read(8))
        # Read the remaining bytes as label data.
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Function to load the entire MNIST dataset (training and testing data)
def load_mnist_data():

    #theres a good chance if you just put your first name in here it will run, otherwise you have to
    #find your mnsit files and replace the input_path with their path
    user = " "
    # Define the path to the dataset files. Update this path to where your MNIST files are located.
    input_path = "/Users/" + user + "/Downloads/archive"  
    
    # Construct the full paths to the dataset files.
    # These files are in the IDX format, which is a binary format used for storing the MNIST dataset.
    xTrain = os.path.join(input_path, "train-images-idx3-ubyte", "train-images-idx3-ubyte")
    yTrain = os.path.join(input_path, "train-labels-idx1-ubyte", "train-labels-idx1-ubyte")
    xTest = os.path.join(input_path, "t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte")
    yTest = os.path.join(input_path, "t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte")

    # Load the training and testing images and labels using the helper functions.
    X_train = load_images(xTrain)
    y_train = load_labels(yTrain)
    X_test = load_images(xTest)
    y_test = load_labels(yTest)

    # Return the loaded data as a tuple.
    return X_train, y_train, X_test, y_test

# This script is useful for beginners who want to work with neural networks.
# The MNIST dataset is often used as a starting point for learning how to train and test models.
# After loading the data using this script, you can use it to train a neural network using libraries like PyTorch or TensorFlow.