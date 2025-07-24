import torch
from torchvision import datasets, transforms


def get_cifar10_dataloader(batch_size, num_workers=0, train=True):
    """
    Returns a DataLoader for the CIFAR-10 dataset.

    Args:
        batch_size (int): Number of samples per batch.
        num_workers (int): Number of subprocesses to use for data loading.
        train (bool): If True, returns the training set; otherwise, returns the test set.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the CIFAR-10 dataset.
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CIFAR10(root='./data', train=train, download=True, transform=transform)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
