import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import random
import numpy as np

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

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data & model definitions (unchanged)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
full_train = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform)
test_set   = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform)
val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

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

# Run one full active‑learning experiment for a given seed
def run_active_learning_seed(seed):
    set_seeds(seed)
    # 1. split
    indices = list(range(len(full_train)))
    random.shuffle(indices)
    labeled   = indices[:initial_labeled].copy()
    unlabeled = indices[initial_labeled:].copy()

    results = []
    for round_idx in range(num_rounds):
        # train from scratch
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

        # evaluate
        acc = evaluate(model, val_loader)
        results.append((len(labeled), acc))

        # query new points (random strategy here)
        new = random.sample(unlabeled, query_size)
        labeled.extend(new)
        unlabeled = [i for i in unlabeled if i not in new]

    return results

# 1) Generate 5 random 32‑bit seeds
seeds = [random.getrandbits(32) for _ in range(5)]
print("Using seeds (use this seeds from now on):", seeds)

# 2) Run experiments
all_results = [run_active_learning_seed(s) for s in seeds]

# 3) Compute average accuracy at each round
avg_results = []
for round_idx in range(num_rounds):
    n_labeled = all_results[0][round_idx][0]
    mean_acc = np.mean([res[round_idx][1] for res in all_results])
    avg_results.append((n_labeled, mean_acc))

# 4) Print
print("\nAverage accuracy over seeds at each round:")
for n, acc in avg_results:
    print(f"  {n:3d} labeled →  avg acc = {acc:.4f}")

# 5) (Optional) Plot
plt.figure()
xs, ys = zip(*avg_results)
plt.plot(xs, ys, marker='s', label='Avg over 5 seeds')
plt.xlabel("Number of labeled samples")
(plt.
 ylabel("Validation accuracy"))
plt.title("Active Learning (average over 5 seeds)")
plt.legend()
plt.show()
