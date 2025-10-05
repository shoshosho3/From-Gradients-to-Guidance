import torch
import torch.nn as nn
import torchvision.models as models

def replace_first_conv_layer(backbone, n_channels):
    """
    Replaces the first convolutional layer to match the desired input channels.
    :param backbone: The ResNet backbone model.
    :param n_channels: Number of input channels.
    :return: The modified backbone model and the original first conv layer.
    """
    original_conv = backbone.conv1
    backbone.conv1 = nn.Conv2d(
        n_channels, original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias
    )
    return backbone, original_conv


def transfer_pretrained_weights_for_grayscale(backbone, original_conv):
    """
    Transfers pretrained RGB weights to grayscale by averaging across channels.
    :param backbone: The ResNet backbone model.
    :param original_conv: The original first convolutional layer.
    :return: The modified backbone model.
    """
    with torch.no_grad():
        backbone.conv1.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)
    return backbone

def handle_non_rgb_input(backbone, n_channels, pretrained):
    """
    Adjusts the first convolutional layer of the backbone to accommodate a different number of input channels.
    :param backbone: The ResNet backbone model.
    :param n_channels: Number of input channels.
    :param pretrained: Whether to use pretrained weights.
    :return: The modified backbone model.
    """
    backbone, original_conv = replace_first_conv_layer(backbone, n_channels)
    if pretrained and n_channels == 1:
        backbone = transfer_pretrained_weights_for_grayscale(backbone, original_conv)
    return backbone


def adapt_resnet_for_dataset(backbone, n_channels, dataset_name):
    """
    Adjusts ResNet for CIFAR-style datasets by changing conv1 and removing maxpool.
    :param backbone: The ResNet backbone model.
    :param n_channels: Number of input channels.
    :param dataset_name: Name of the dataset to determine architecture adjustments.
    :return: The modified backbone model.
    """
    cifar_style_datasets = ['cifar10', 'cifar100', 'tinyimagenet', 'breakhis', 'pneumoniamnist']
    if dataset_name.lower() in cifar_style_datasets:
        backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
    return backbone


def create_adapted_resnet18(num_classes, pretrained=True, n_channels=3, dataset_name='cifar10'):
    """
    Creates a ResNet18 model adapted for different datasets, PRECISELY following
    the specifications from the reference paper.

    - CIFAR-like datasets get a 3x3 kernel and no initial max-pooling.
    - MNIST-like datasets use the standard 7x7 kernel PyTorch ResNet18.
    :param num_classes: Number of output classes.
    :param pretrained: Whether to use pretrained weights.
    :param n_channels: Number of input channels.
    :param dataset_name: Name of the dataset to determine architecture adjustments.
    :return: The adapted ResNet18 model and the feature dimension before the classifier.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    backbone = models.resnet18(weights=weights)
    if n_channels != 3: # handling non-RGB inputs
        backbone = handle_non_rgb_input(backbone, n_channels, pretrained)
    backbone = adapt_resnet_for_dataset(backbone, n_channels, dataset_name)
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feature_dim

class StandardResNet18(nn.Module):
    """
    A standard ResNet18-based backend for classification.
    Returns both logits and embeddings.
    """

    def __init__(self, num_classes, pretrained=True, n_channels=3, dataset_name='cifar10'):
        """
        Initializes the StandardResNet18 model.
        :param num_classes: Number of output classes.
        :param pretrained: Whether to use pretrained weights.
        :param n_channels: Number of input channels.
        :param dataset_name: Name of the dataset to determine architecture adjustments.
        """
        super().__init__()
        self.features, self.feature_dim = create_adapted_resnet18(
            num_classes, pretrained, n_channels, dataset_name
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through the network.
        :param x: Input tensor.
        :return: Logits and embeddings.
        """
        embeddings = self.features(x).view(x.size(0), -1)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def get_embedding_dim(self):
        """
        Returns the dimension of the embeddings.
        :return: The embedding dimension.
        """
        return self.feature_dim