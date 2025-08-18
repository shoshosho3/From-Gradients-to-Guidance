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


def load_mnist_classes(digits, batch_size, num_workers=0, train=True):
    """
    Load MNIST dataloader with a given list of digits.

    Args:
        digits (list[int]): Digits to include
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for DataLoader
        train (bool): Whether to load training set or test set

    Returns:
        dataset: Filtered MNIST dataset with digits mapped
    """
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=train, download=True, transform=transform)

    # Filter dataset
    mask = torch.zeros_like(dataset.targets, dtype=torch.bool)
    for d in digits:
        mask |= (dataset.targets == d)

    dataset.data = dataset.data[mask]
    dataset.targets = dataset.targets[mask]

    # Remap labels: digits -> 0..n_classes-1
    mapping = {d: i for i, d in enumerate(digits)}
    dataset.targets = torch.tensor([mapping[int(t)] for t in dataset.targets], dtype=torch.long)

    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

