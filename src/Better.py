import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import random
import multiprocessing

# Ensure Windows multiprocessing works
multiprocessing.freeze_support()

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
initial_labeled = 100
query_size = 50
batch_size = 32
lambda_rank = 0.001
margin = 0.1
num_epochs = 5
num_rounds = 5

# 1. Prepare data
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
full_train = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform)
test_set   = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform)
val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

# 2. Split into labeled/unlabeled pools
indices = list(range(len(full_train)))
random.shuffle(indices)
labeled_indices   = indices[:initial_labeled]
unlabeled_indices = indices[initial_labeled:]
initial_labeled_indices   = labeled_indices.copy()
initial_unlabeled_indices = unlabeled_indices.copy()

# 3. Model and helpers
class ActiveLearningModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.cls_head = nn.Linear(self.feature_dim, num_classes)
        self.up_head  = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        f = self.features(x).view(x.size(0), -1)
        return self.cls_head(f), self.up_head(f).squeeze(1)

def compute_true_grad_norm(model, x, y, criterion):
    model.zero_grad()
    logits, _ = model(x.unsqueeze(0))
    loss = criterion(logits, y.unsqueeze(0))
    params = list(model.features.parameters()) + list(model.cls_head.parameters())
    grads = torch.autograd.grad(loss, params, retain_graph=False)
    return torch.sqrt(sum((g**2).sum() for g in grads))

def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# 4. Active-learning runner
def run_active_learning(strategy):
    labeled = initial_labeled_indices.copy()
    unlabeled = initial_unlabeled_indices.copy()
    results = []

    for round_idx in range(num_rounds):
        # Train from scratch on current labeled set
        model = ActiveLearningModel(num_classes).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
        criterion_cls = nn.CrossEntropyLoss()
        train_loader = DataLoader(Subset(full_train, labeled),
                                  batch_size=batch_size, shuffle=True, num_workers=0)

        model.train()
        for epoch in range(num_epochs):
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits, pred_update = model(x_batch)
                loss_cls = criterion_cls(logits, y_batch)

                true_norms = torch.stack([
                    compute_true_grad_norm(model, xi, yi, criterion_cls)
                    for xi, yi in zip(x_batch, y_batch)
                ])
                idx_perm = torch.randperm(len(true_norms))
                ranking_loss = sum(
                    F.relu(-torch.sign(true_norms[a]-true_norms[b])*(pred_update[a]-pred_update[b]) + margin)
                    for a, b in zip(idx_perm[::2], idx_perm[1::2])
                ) / (len(true_norms)//2)

                loss = loss_cls + lambda_rank * ranking_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Evaluate
        acc = evaluate(model, val_loader)
        results.append((len(labeled), acc))

        # Query
        if strategy == 'weight_change':
            scores = []
            pool_loader = DataLoader(Subset(full_train, unlabeled),
                                     batch_size=batch_size, shuffle=False, num_workers=0)
            model.eval()
            with torch.no_grad():
                for batch_idx, (x_pool, _) in enumerate(pool_loader):
                    x_pool = x_pool.to(device)
                    _, preds = model(x_pool)
                    start = batch_idx * batch_size
                    for i, score in enumerate(preds):
                        global_idx = unlabeled[start + i]
                        scores.append((score.item(), global_idx))
            scores.sort(key=lambda t: -t[0])
            new = [idx for _, idx in scores[:query_size]]
        else:  # random
            new = random.sample(unlabeled, query_size)

        # Update pools
        labeled.extend(new)
        unlabeled = [i for i in unlabeled if i not in new]

    return results

# 5. Run and plot both strategies
results_w = run_active_learning('weight_change')
results_r = run_active_learning('random')

plt.figure()
xs_w, ys_w = zip(*results_w)
plt.plot(xs_w, ys_w, marker='o', label='Weight-Change')
xs_r, ys_r = zip(*results_r)
plt.plot(xs_r, ys_r, marker='x', label='Random')
plt.xlabel("Number of labeled samples")
plt.ylabel("Validation accuracy")
plt.title("Active Learning: Weight-Change vs Random")
plt.legend()
plt.show()
