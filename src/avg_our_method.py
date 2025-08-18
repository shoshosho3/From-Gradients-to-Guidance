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
batch_size = 32


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Data & model definitions (unchanged)
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
])
full_train = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform)
val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)


class ActiveLearningModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
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
        return self.cls_head(f), self.up_head(f).squeeze(1)


def compute_true_grad_norm(model, x, y, criterion):
    model.zero_grad()
    logits, _ = model(x.unsqueeze(0))
    loss = criterion(logits, y.unsqueeze(0))
    params = list(model.features.parameters()) + list(model.cls_head.parameters())
    grads = torch.autograd.grad(loss, params, retain_graph=False)
    return torch.sqrt(sum((g ** 2).sum() for g in grads))


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

def run_active_learning_seed(seed, margin, lambda_rank, num_classes, initial_labeled=100, query_size=50,
                             batch_size=32, num_epochs=5, num_rounds=20):
    set_seeds(seed)
    # 1. split
    indices = list(range(len(full_train)))
    random.shuffle(indices)
    labeled = indices[:initial_labeled].copy()
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
                    F.relu(-torch.sign(true_norms[a] - true_norms[b]) * (pred_update[a] - pred_update[b]) + margin)
                    for a, b in zip(idx_perm[::2], idx_perm[1::2])
                ) / (len(true_norms) // 2)

                loss = loss_cls + lambda_rank * ranking_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # evaluate
        acc = evaluate(model, val_loader)
        results.append((len(labeled), acc))

        # query new points (random strategy here)
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
        scores.sort(key=lambda t: t[0])
        discard = int(0.1 * len(scores))
        candidates = scores[discard:]
        selected = random.sample(
            candidates,
            k=min(query_size, len(candidates))
        )
        new = [idx for _, idx in selected]
        labeled.extend(new)
        unlabeled = [i for i in unlabeled if i not in new]
    return results


# 1) Put here the seeds from the random
seeds = [1]

# 100 labeled →  avg acc = 0.2380
#   150 labeled →  avg acc = 0.3406
#   200 labeled →  avg acc = 0.3974
#   250 labeled →  avg acc = 0.4327
#   300 labeled →  avg acc = 0.4840
# 2) Run experiments

num_classes = 10
initial_labeled = 100
query_size = 50
num_epochs = 20
num_rounds = 20
# 4) Print
print("\nAverage accuracy over seeds at each round:")
lambda_rank, margin = 0.01, 0.1
all_results = [run_active_learning_seed(s, margin=margin, lambda_rank=lambda_rank, num_classes=num_classes,
                                        initial_labeled=initial_labeled, query_size=query_size, batch_size=batch_size,
                                        num_epochs=num_epochs, num_rounds=num_rounds) for s in seeds]

# 3) Compute average accuracy at each round
avg_results = []
for round_idx in range(num_rounds):
    n_labeled = all_results[0][round_idx][0]
    mean_acc = np.mean([res[round_idx][1] for res in all_results])
    avg_results.append((n_labeled, mean_acc))

output_path = f"avg_results_lr{lambda_rank}_m{margin}.txt"
with open(output_path, "w") as f:
    for n, acc in avg_results:
        line = f"  {n:3d} labeled →  avg acc = {acc:.4f}\n"
        print(line, end="")  # still prints to stdout
        f.write(line)  # writes the same line to the file

print(f"\nSaved results to {output_path}")

xs_, ys_ = [100, 150, 200, 250, 300], [0.238, 0.3406, 0.3974, 0.4327, 0.484]
# 5) (Optional) Plot
plt.figure()
xs, ys = zip(*avg_results)
plt.plot(xs, ys, marker='s', label='Avg over 5 seeds')
plt.plot(xs_, ys_, marker='s', label='Random over 5 seeds')
plt.xlabel("Number of labeled samples")
plt.ylabel("Validation accuracy")
plt.savefig('figure')
plt.title("Active Learning (average over 5 seeds)")
plt.legend()
plt.show()
