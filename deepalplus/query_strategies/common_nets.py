import torch
import torch.nn as nn
import torchvision.models as models

def handle_non_rgb_input(backbone, n_channels, pretrained):
    """
    Adjusts the first convolutional layer of the backbone to accommodate a different number of input channels.
    :param backbone: The ResNet backbone model.
    :param n_channels: Number of input channels.
    :param pretrained: Whether to use pretrained weights.
    :return: The modified backbone model.
    """

    original_conv = backbone.conv1
    backbone.conv1 = nn.Conv2d(n_channels, original_conv.out_channels,
                               kernel_size=original_conv.kernel_size,
                               stride=original_conv.stride,
                               padding=original_conv.padding,
                               bias=original_conv.bias)
    if pretrained and n_channels == 1:
        with torch.no_grad():
            # Average RGB weights for grayscale transfer
            backbone.conv1.weight[:] = original_conv.weight.mean(dim=1, keepdim=True)

    return backbone


def create_adapted_resnet18(num_classes, pretrained=True, n_channels=3, dataset_name='cifar10'):
    """
    Creates a ResNet18 model adapted for different datasets, PRECISELY following
    the specifications from the reference paper.

    - CIFAR-like datasets get a 3x3 kernel and no initial max-pooling.
    - MNIST-like datasets use the standard 7x7 kernel PyTorch ResNet18.
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    backbone = models.resnet18(weights=weights)

    # Group datasets based on the required ResNet implementation style
    cifar_style_datasets = ['cifar10', 'cifar100', 'tinyimagenet', 'breakhis', 'pneumoniamnist']

    if n_channels != 3: # handling non-RGB inputs
        backbone = handle_non_rgb_input(backbone, n_channels, pretrained)

    if dataset_name.lower() in cifar_style_datasets:
        backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()

    # Storing feature dimension and replacing the final layer remains the same
    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()  # We will manage the classifier head separately

    return backbone, feature_dim


class StandardResNet18(nn.Module):
    """
    A standard ResNet18-based backend for classification.
    Returns both logits and embeddings.
    """

    def __init__(self, num_classes, pretrained=True, n_channels=3, dataset_name='cifar10'):
        super().__init__()
        self.features, self.feature_dim = create_adapted_resnet18(
            num_classes, pretrained, n_channels, dataset_name
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        embeddings = self.features(x).view(x.size(0), -1)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def get_embedding_dim(self):
        return self.feature_dim