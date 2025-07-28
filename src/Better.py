import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import random

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
initial_labeled = 100  # Initial seed size
query_size = 50  # How many to query each round
batch_size = 32
lambda_rank = 0.1
margin = 1.0
num_epochs = 5
num_rounds = 5  # Number of active-learning rounds

# 1. Data transforms and datasets
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
full_train = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform)

# Split into initial labeled and unlabeled pools
indices = list(range(len(full_train)))
random.shuffle(indices)
labeled_indices = indices[:initial_labeled]
unlabeled_indices = indices[initial_labeled:]

# Validation loader
val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)


# Define model with two heads
class ActiveLearningModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.cls_head = nn.Linear(self.feature_dim, num_classes)
        self.up_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        f = self.features(x).view(x.size(0), -1)
        logits = self.cls_head(f)
        pred_update = self.up_head(f).squeeze(1)
        return logits, pred_update


# Helper: compute true gradient norm
def compute_true_grad_norm(model, x, y, criterion):
    model.zero_grad()
    logits, _ = model(x.unsqueeze(0))
    loss = criterion(logits, y.unsqueeze(0))
    grads = torch.autograd.grad(loss, model.parameters())
    return torch.sqrt(sum((g ** 2).sum() for g in grads))


# Helper: evaluate accuracy
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


# Active learning loop
results = []
for round_idx in range(num_rounds):
    # Create loaders for current labeled pool
    labeled_set = Subset(full_train, labeled_indices)
    labeled_loader = DataLoader(labeled_set, batch_size=batch_size, shuffle=True, num_workers=2)

    # Initialize model for this round
    model = ActiveLearningModel(num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion_cls = nn.CrossEntropyLoss()

    # Warm-up train
    model.train()
    for epoch in range(num_epochs):
        for x_batch, y_batch in labeled_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits, pred_update = model(x_batch)
            loss_cls = criterion_cls(logits, y_batch)

            # Compute true gradient norms
            true_norms = torch.stack([
                compute_true_grad_norm(model, xi, yi, criterion_cls)
                for xi, yi in zip(x_batch, y_batch)
            ])

            # Pairwise ranking loss
            idx = torch.randperm(len(true_norms))
            ranking_loss = 0.0
            pairs = len(idx) // 2
            for i in range(pairs):
                a, b = idx[2 * i], idx[2 * i + 1]
                sign = torch.sign(true_norms[a] - true_norms[b])
                ranking_loss += F.relu(-sign * (pred_update[a] - pred_update[b]) + margin)
            ranking_loss = ranking_loss / pairs

            loss = loss_cls + lambda_rank * ranking_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate and record
    acc = evaluate(model, val_loader)
    results.append((len(labeled_indices), acc))

    # Score unlabeled pool
    model.eval()
    scores = []
    unlabeled_loader = DataLoader(Subset(full_train, unlabeled_indices), batch_size=batch_size, shuffle=False,
                                  num_workers=2)
    with torch.no_grad():
        for batch_idx, (x_pool, _) in enumerate(unlabeled_loader):
            x_pool = x_pool.to(device)
            _, preds = model(x_pool)
            start = batch_idx * batch_size
            for i, score in enumerate(preds):
                global_idx = unlabeled_indices[start + i]
                scores.append((score.item(), global_idx))
    # Select top-K most impactful
    scores.sort(key=lambda t: -t[0])
    new_indices = [idx for _, idx in scores[:query_size]]

    # Update pools
    labeled_indices.extend(new_indices)
    unlabeled_indices = [idx for idx in unlabeled_indices if idx not in new_indices]

# Plot results
labels, accuracies = zip(*results)
plt.figure()
plt.plot(labels, accuracies)
plt.xlabel("Number of labeled samples")
plt.ylabel("Validation accuracy")
plt.title("Active Learning Performance")
plt.show()
