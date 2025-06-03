# This program demonstrates how to build a Convolutional Neural Network (CNN) to classify images from the famous MNIST dataset.
# The MNIST dataset contains grayscale images of handwritten digits (0-9) and their corresponding labels.

# First, we import the necessary libraries.

# PyTorch is a popular deep learning library that provides tools to build and train neural networks.
import torch
import torch.nn as nn  # Contains modules for building neural network layers.
import torch.optim as optim  # Provides optimization algorithms like Adam, SGD, etc.
from torch.utils.data import DataLoader, TensorDataset  # Utilities for handling datasets and batching.

# We use a custom loader to load the MNIST dataset.
# The loader is included in the git repository
import mnist_loader as mnist

# Load the MNIST dataset using the custom loader.
# The dataset is split into training and testing sets.
# X_train and X_test contain the image data, while y_train and y_test contain the corresponding labels, or examples of what the output from the network should be.
X_train, y_train, X_test, y_test = mnist.load_mnist_data()

# Convert the data into PyTorch tensors and normalize the pixel values.
# Tensors are PyTorch's data structures, similar to multi-dimensional arrays.
# Normalization scales the pixel values (originally between 0 and 255) to a range of 0 to 1, which helps the model train more effectively.
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1) / 255.0  # Add a channel dimension for grayscale images.
y_train = torch.tensor(y_train, dtype=torch.long)

# Similarly, normalize the test data.
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1) / 255.0
y_test = torch.tensor(y_test, dtype=torch.long)

# Use DataLoader to create batches of data for training.
# Batching divides the dataset into smaller groups (batches) to process during training, which reduces memory usage and speeds up computation.
batch_size = 64  # Number of samples per batch.
train_dataset = TensorDataset(X_train, y_train)  # Combine the training data and labels into a dataset.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle the data to improve training.

# Define the CNN model.
# A CNN is a type of neural network designed to process image data by extracting spatial features.
cnn_model = nn.Sequential(
    # First convolutional layer:
    # - Input: 1 channel (grayscale image).
    # - Output: 16 feature maps.
    # - Kernel size: 3x3 (small sliding window to detect patterns).
    # - Stride: 1 (move the kernel one pixel at a time).
    # - Padding: 1 (add a border of zeros to preserve image dimensions).

    # Conv function looks at a kernel, in this case a 3x3 matrix of doubles. it bultiplies it by the input elements, applies a bias
    # All products are summed to create one output value.
    # It detects simple and complex feature. in this case edges and intersections of lines in the number.
    nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),  # Activation function: Introduces non-linearity to the model.
    nn.MaxPool2d(kernel_size=2, stride=2),  # Max pooling: Reduces the spatial dimensions by half.

    # Second convolutional layer:
    # - Input: 16 feature maps.
    # - Output: 32 feature maps.
    nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),  # Activation function.
    nn.MaxPool2d(kernel_size=2, stride=2),  # Further reduces spatial dimensions.

    # Flatten layer:
    # Converts the 2D feature maps into a 1D vector to feed into fully connected layers.
    nn.Flatten(),

    # Fully connected layer:
    # - Input: Flattened vector (32 * 7 * 7).
    # - Output: 64 hidden units.
    nn.Linear(32 * 7 * 7, 64),
    nn.ReLU(),  # Activation function.

    # Output layer:
    # - Input: 64 hidden units.
    # - Output: 10 classes (digits 0-9).
    nn.Linear(64, 10)
)

# Check if a GPU is available and move the model to the GPU for faster computation.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cnn_model.to(device)
# This runs in about 5-10 minutes on cpu, 30s-1 minute on a gpu.

# Define the loss function and optimizer.
# - Loss function: Measures the difference between the predicted and actual labels. We use CrossEntropyLoss for classification tasks.
# - Optimizer: Updates the model's parameters to minimize the loss. We use the Adam optimizer with a learning rate of 0.001.
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

# Training loop: Train the model for a specified number of epochs (iterations over the entire dataset).
epochs = 10  # Number of times the model sees the entire dataset.
for epoch in range(epochs):
    cnn_model.train()  # Set the model to training mode (enables features like dropout, if used).
    epoch_loss = 0  # Initialize the total loss for the epoch.
    for batch_x, batch_y in train_loader:  # Iterate over batches of data.
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)  # Move data to the GPU (if available).
        optimizer.zero_grad()  # Reset gradients from the previous step.
        outputs = cnn_model(batch_x)  # Perform a forward pass (predict outputs).
        loss = loss_fn(outputs, batch_y)  # Compute the loss for the batch.
        loss.backward()  # Perform a backward pass (compute gradients).
        optimizer.step()  # Update the model's parameters using the optimizer.
        epoch_loss += loss.item()  # Accumulate the loss for the epoch.
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")  # Print the average loss for the epoch.

# Save the trained model's parameters to a file for later use.
torch.save(cnn_model.state_dict(), "cnn_model.pth")

# Testing the model: Evaluate its performance on the test dataset.
cnn_model.eval()  # Set the model to evaluation mode (disables features like dropout).
X_test, y_test = X_test.to(device), y_test.to(device)  # Move test data to the GPU (if available).
with torch.no_grad():  # Disable gradient computation (not needed during testing).
    outputs = cnn_model(X_test)  # Perform a forward pass on the test data.
    _, predicted = torch.max(outputs, 1)  # Get the predicted class for each image.
    accuracy = (predicted == y_test).sum().item() / y_test.size(0)  # Calculate the accuracy.
    print(f"Test Accuracy: {accuracy * 100:.2f}%")  # Print the accuracy as a percentage.
