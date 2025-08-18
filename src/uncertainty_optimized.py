import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import random
import numpy as np
import time
import os

# Configuration remains the same...
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = os.cpu_count() if device.type == 'cuda' else 0
pin_memory = device.type == 'cuda'
batch_size = 128
num_classes = 10
initial_labeled = 100
query_size = 50
num_epochs = 20
num_rounds = 20


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_train = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform)
val_loader = DataLoader(test_set, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers,
                        pin_memory=pin_memory)


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


@torch.no_grad()
def compute_uncertainty_scores(model, unlabeled_loader):
    model.eval()
    all_scores = []
    # --- FIX: Updated AMP syntax ---
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
        for x_batch, _ in unlabeled_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            logits = model(x_batch)
            probs = F.softmax(logits, dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            top_probs, _ = torch.topk(probs, 2, dim=1)
            margin = top_probs[:, 0] - top_probs[:, 1]
            combined_uncertainty = entropy - margin
            all_scores.append(combined_uncertainty)
    return torch.cat(all_scores)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    # --- FIX: Updated AMP syntax ---
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum()
            total += y.size(0)
    return (correct / total).item()


def run_active_learning_seed(seed, strategy):
    print(f"Running {strategy} strategy with seed: {seed}")
    set_seeds(seed)

    indices = list(range(len(full_train)))
    random.shuffle(indices)
    labeled_indices = set(indices[:initial_labeled])
    unlabeled_indices = set(indices[initial_labeled:])
    results = []

    model = ActiveLearningModel(num_classes).to(device)

    # --- FIX: Check for GPU compatibility before using torch.compile ---
    if device.type == 'cuda':
        major, _ = torch.cuda.get_device_capability()
        if major >= 7:
            try:
                model = torch.compile(model)
                print("Model compiled successfully (PyTorch 2.0+ on compatible GPU).")
            except Exception as e:
                print(f"Could not compile model despite compatible GPU, error: {e}")
        else:
            print(f"GPU (CUDA Capability {major}.x) is too old for torch.compile. Running without.")

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # --- FIX: Updated AMP syntax ---
    scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))

    for round_idx in range(num_rounds):
        print(
            f"  Round {round_idx + 1}/{num_rounds}, labeled: {len(labeled_indices)}, unlabeled: {len(unlabeled_indices)}")
        train_subset = Subset(full_train, list(labeled_indices))
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                  pin_memory=pin_memory)

        model.train()
        for epoch in range(num_epochs):
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device, non_blocking=True), y_batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)

                # --- FIX: Updated AMP syntax ---
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    logits = model(x_batch)
                    loss = criterion(logits, y_batch)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        acc = evaluate(model, val_loader)
        results.append((len(labeled_indices), acc))
        print(f"    Accuracy: {acc:.4f}")

        if not unlabeled_indices: break

        selected_count = min(query_size, len(unlabeled_indices))

        if strategy == 'uncertainty':
            unlabeled_list = list(unlabeled_indices)
            unlabeled_subset = Subset(full_train, unlabeled_list)
            unlabeled_loader = DataLoader(unlabeled_subset, batch_size=batch_size * 2, shuffle=False,
                                          num_workers=num_workers, pin_memory=pin_memory)
            uncertainty_scores = compute_uncertainty_scores(model, unlabeled_loader)
            _, top_k_indices_in_loader = torch.topk(uncertainty_scores, selected_count)
            selected_original_indices = {unlabeled_list[i] for i in top_k_indices_in_loader.cpu().numpy()}
        elif strategy == 'random':
            selected_original_indices = set(random.sample(list(unlabeled_indices), selected_count))
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        labeled_indices.update(selected_original_indices)
        unlabeled_indices.difference_update(selected_original_indices)

    return results


# The main function remains unchanged...
def main():
    """Main function to run sanity check comparing uncertainty vs random"""
    print("=== SANITY CHECK: Uncertainty vs Random Active Learning (OPTIMIZED & ROBUST) ===")

    seeds = [42, 123]
    print(f"Using seeds: {seeds}")

    print("\n1. Running Uncertainty-based Active Learning...")
    start_time = time.time()
    uncertainty_results = [run_active_learning_seed(seed, 'uncertainty') for seed in seeds]
    uncertainty_time = time.time() - start_time
    print(f"Uncertainty strategy took {uncertainty_time:.2f} seconds.")

    print("\n2. Running Random Active Learning...")
    start_time = time.time()
    random_results = [run_active_learning_seed(seed, 'random') for seed in seeds]
    random_time = time.time() - start_time
    print(f"Random strategy took {random_time:.2f} seconds.")

    # (Plotting and analysis code remains the same)
    # ...
    print("\n3. Computing average results...")
    uncertainty_avg = []
    random_avg = []

    for round_idx in range(num_rounds):
        n_labeled = uncertainty_results[0][round_idx][0]
        unc_acc = np.mean([res[round_idx][1] for res in uncertainty_results])
        uncertainty_avg.append((n_labeled, unc_acc))
        rand_acc = np.mean([res[round_idx][1] for res in random_results])
        random_avg.append((n_labeled, rand_acc))

    print("\n=== RESULTS COMPARISON ===")
    print("Labeled Samples | Uncertainty | Random   | Difference")
    print("-" * 50)
    # ... rest of main ...
    total_uncertainty_win = 0
    for i, ((n_unc, acc_unc), (n_rand, acc_rand)) in enumerate(zip(uncertainty_avg, random_avg)):
        diff = acc_unc - acc_rand
        if diff > 0.001:  # Use a small threshold to avoid calling ties wins
            total_uncertainty_win += 1
            status = "✓ UNC wins"
        elif diff < -0.001:
            status = "✗ RAND wins"
        else:
            status = "= TIE"

        print(f"{n_unc:14d} | {acc_unc:10.4f} | {acc_rand:7.4f} | {diff:+7.4f} {status}")

    print(f"\n=== SANITY CHECK VERDICT ===")
    print(f"Uncertainty strategy wins: {total_uncertainty_win}/{len(uncertainty_avg)} rounds")

    if total_uncertainty_win > len(uncertainty_avg) * 0.7:
        print("✅ SANITY CHECK PASSED: Uncertainty strategy consistently beats random!")
    else:
        print("❌ SANITY CHECK FAILED: Random strategy wins too many rounds!")

    plt.figure(figsize=(10, 6))
    x_unc, y_unc = zip(*uncertainty_avg)
    x_rand, y_rand = zip(*random_avg)
    plt.plot(x_unc, y_unc, 'b-o', label='Uncertainty Strategy')
    plt.plot(x_rand, y_rand, 'r-s', label='Random Strategy')
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Validation Accuracy")
    plt.title("Sanity Check: Uncertainty vs Random Active Learning (Optimized)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()