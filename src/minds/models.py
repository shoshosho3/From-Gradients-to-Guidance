import torch.nn as nn
import torch.nn.functional as F


class LightWeightCNN(nn.Module):
    """
    A lightweight Convolutional Neural Network (CNN) for image classification.
    This model consists of two convolutional layers followed by a fully connected layer.
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB images).
        conv1_out (int): Number of output channels for the first convolutional layer (default: 16).
        conv2_out (int): Number of output channels for the second convolutional layer (default: 32).
        kernel_size (int): Size of the convolutional kernel (default: 3).
        pool_size (int): Size of the max pooling window (default: 2).
        fc_hidden (int): Number of hidden units in the fully connected layer (default: 128).
        num_classes (int): Number of output classes for classification (default: 10).
    """

    def __init__(self, in_channels=3, conv1_out=16, conv2_out=32,
                 kernel_size=3, pool_size=2, fc_hidden=128, num_classes=10):
        super().__init__()

        pad = kernel_size // 2 # padding to maintain the input size

        # Define the convolutional layers
        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size, padding=pad)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size, padding=pad)
        self.pool = nn.MaxPool2d(pool_size)

        # Initialize the fully connected layer parameters
        self.fc_hidden, self.num_classes = fc_hidden, num_classes

        # Classifier will be initialized lazily
        self.classifier = None

    def create_classifier(self, input_size):
        """
        Creates the classifier part of the model.
        This is called lazily when the forward method is invoked for the first time.
        """
        self.classifier = nn.Sequential(
            nn.Linear(input_size, self.fc_hidden), # fully connected layer
            nn.ReLU(),
            nn.Linear(self.fc_hidden, self.num_classes) # output layer for classification
        )

    def forward(self, x):
        """
        forward pass of the model
        """

        x = self.pool(F.relu(self.conv1(x))) # first convolutional layer
        x = self.pool(F.relu(self.conv2(x))) # second convolutional layer
        x = x.view(x.size(0), -1) # flattening the tensor

        if self.classifier is None:  # lazy initialization
            self.create_classifier(x.size(1))

        return self.classifier(x) # fully connected layer for classification
