import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import random
import numpy as np

# Quick test configuration (fewer rounds/epochs for faster testing)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
initial_labeled = 100
query_size = 50
batch_size = 32
num_epochs = 5  # Reduced for quick testing
num_rounds = 5  # Reduced for quick testing


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Data & model definitions
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

    def forward(self, x):
        f = self.features(x).view(x.size(0), -1)
        return self.cls_head(f)


def compute_uncertainty_scores(model, unlabeled_loader):
    """Compute uncertainty scores for unlabeled samples."""
    model.eval()
    uncertainty_scores = []

    with torch.no_grad():
        for batch_idx, (x_batch, _) in enumerate(unlabeled_loader):
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = F.softmax(logits, dim=1)

            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            combined_uncertainty = entropy + (1.0 - margin)

            for i, score in enumerate(combined_uncertainty):
                # The index returned is the index *within the unlabeled pool*
                pool_idx = batch_idx * unlabeled_loader.batch_size + i
                uncertainty_scores.append((score.item(), pool_idx))

    return uncertainty_scores


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total


def run_uncertainty_active_learning_seed(seed):
    """Run uncertainty-based active learning for a given seed"""
    print(f"Running uncertainty strategy with seed: {seed}")
    set_seeds(seed)

    indices = list(range(len(full_train)))
    random.shuffle(indices)
    labeled = indices[:initial_labeled].copy()
    unlabeled = indices[initial_labeled:].copy()

    results = []

    ### FIX 1: Initialize model and optimizer ONCE, before the loop starts.
    # This ensures the model is fine-tuned, not retrained from scratch.
    model = ActiveLearningModel(num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for round_idx in range(num_rounds):
        print(f"  Round {round_idx + 1}/{num_rounds}, labeled: {len(labeled)}")

        train_loader = DataLoader(Subset(full_train, labeled),
                                  batch_size=batch_size, shuffle=True, num_workers=0)
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        acc = evaluate(model, val_loader)
        results.append((len(labeled), acc))
        print(f"    Accuracy: {acc:.4f}")

        # Query new points using uncertainty strategy
        if len(unlabeled) > 0:
            unlabeled_subset = Subset(full_train, unlabeled)
            unlabeled_loader = DataLoader(unlabeled_subset,
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=0)

            uncertainty_scores = compute_uncertainty_scores(model, unlabeled_loader)
            uncertainty_scores.sort(key=lambda x: x[0], reverse=True)

            selected_count = min(query_size, len(unlabeled))

            ### FIX 2: Correctly map pool indices back to original dataset indices.
            # 1. Get indices of the most uncertain samples *within the unlabeled pool*.
            query_pool_indices = [score_tuple[1] for score_tuple in uncertainty_scores[:selected_count]]
            # 2. Use these to get the *actual dataset indices* from the `unlabeled` list.
            selected_original_indices = [unlabeled[i] for i in query_pool_indices]

            # 3. Update labeled and unlabeled sets with the correct original indices.
            labeled.extend(selected_original_indices)
            unlabeled = [i for i in unlabeled if i not in selected_original_indices]

    return results


def run_random_active_learning_seed(seed):
    """Run random sampling active learning for a given seed"""
    print(f"Running random strategy with seed: {seed}")
    set_seeds(seed)

    indices = list(range(len(full_train)))
    random.shuffle(indices)
    labeled = indices[:initial_labeled].copy()
    unlabeled = indices[initial_labeled:].copy()

    results = []

    ### FIX 1: Initialize model and optimizer ONCE, for a fair comparison.
    model = ActiveLearningModel(num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for round_idx in range(num_rounds):
        print(f"  Round {round_idx + 1}/{num_rounds}, labeled: {len(labeled)}")

        train_loader = DataLoader(Subset(full_train, labeled),
                                  batch_size=batch_size, shuffle=True, num_workers=0)
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        acc = evaluate(model, val_loader)
        results.append((len(labeled), acc))
        print(f"    Accuracy: {acc:.4f}")

        # Query new points using random strategy
        if len(unlabeled) > 0:
            selected_count = min(query_size, len(unlabeled))
            selected_indices = random.sample(unlabeled, selected_count)

            labeled.extend(selected_indices)
            unlabeled = [i for i in unlabeled if i not in selected_indices]

    return results


def main():
    """Quick sanity check test"""
    print("=== QUICK SANITY CHECK TEST (CORRECTED) ===")

    seeds = [42, 123]
    print(f"Using seeds: {seeds}")

    print("\n1. Running Uncertainty-based Active Learning...")
    uncertainty_results = [run_uncertainty_active_learning_seed(seed) for seed in seeds]

    print("\n2. Running Random Active Learning...")
    random_results = [run_random_active_learning_seed(seed) for seed in seeds]

    print("\n3. Computing average results...")
    uncertainty_avg = []
    random_avg = []

    for round_idx in range(num_rounds):
        n_labeled = uncertainty_results[0][round_idx][0]
        unc_acc = np.mean([res[round_idx][1] for res in uncertainty_results])
        uncertainty_avg.append((n_labeled, unc_acc))
        rand_acc = np.mean([res[round_idx][1] for res in random_results])
        random_avg.append((n_labeled, rand_acc))

    print("\n=== QUICK TEST RESULTS ===")
    print("Labeled Samples | Uncertainty | Random   | Difference")
    print("-" * 50)

    total_uncertainty_win = 0
    for i, ((n_unc, acc_unc), (n_rand, acc_rand)) in enumerate(zip(uncertainty_avg, random_avg)):
        diff = acc_unc - acc_rand
        if diff > 0.001:
            total_uncertainty_win += 1
            status = "✓ UNC wins"
        elif diff < -0.001:
            status = "✗ RAND wins"
        else:
            status = "= TIE"
        print(f"{n_unc:14d} | {acc_unc:10.4f} | {acc_rand:7.4f} | {diff:+7.4f} {status}")

    print(f"\n=== QUICK TEST VERDICT ===")
    print(f"Uncertainty strategy wins: {total_uncertainty_win}/{len(uncertainty_avg)} rounds")

    if total_uncertainty_win >= len(uncertainty_avg) * 0.6:  # Win >60%
        print("✅ QUICK TEST PASSED: Uncertainty strategy is outperforming random.")
    else:
        print("❌ QUICK TEST FAILED: Random strategy is competitive or winning.")

    plt.figure(figsize=(8, 5))
    x_unc, y_unc = zip(*uncertainty_avg)
    x_rand, y_rand = zip(*random_avg)
    plt.plot(x_unc, y_unc, 'b-o', label='Uncertainty Strategy', linewidth=2)
    plt.plot(x_rand, y_rand, 'r-s', label='Random Strategy', linewidth=2)
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Validation Accuracy")
    plt.title("Quick Sanity Check (Corrected): Uncertainty vs Random")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()