import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import random
import numpy as np

# TINY test configuration for ultra-fast verification
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 10
initial_labeled = 50  # Much smaller nitial set
query_size = 25  # Smaller query size
batch_size = 16  # Smaller batch size
num_epochs = 2  # Very few epochs
num_rounds = 3  # Very few rounds
max_train_samples = 1000  # Limit total training samples

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Data & model definitions
transform = transforms.Compose([
    transforms.Resize(64),  # Smaller images for speed
    transforms.ToTensor(),
])
full_train = datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform)
test_set = datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform)

# Take only a small subset for ultra-fast testing
train_indices = random.sample(range(len(full_train)), min(max_train_samples, len(full_train)))
full_train = Subset(full_train, train_indices)

val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

class TinyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Much smaller model for speed
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def compute_uncertainty_scores(model, unlabeled_loader):
    """Compute uncertainty scores for unlabeled samples using entropy and margin"""
    model.eval()
    uncertainty_scores = []
    
    with torch.no_grad():
        for batch_idx, (x_batch, _) in enumerate(unlabeled_loader):
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            probs = F.softmax(logits, dim=1)
            
            # Entropy-based uncertainty
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
            
            # Margin-based uncertainty (difference between top-2 predictions)
            sorted_probs, _ = torch.sort(probs, dim=1, descending=True)
            margin = sorted_probs[:, 0] - sorted_probs[:, 1]
            
            # Combined uncertainty score (higher = more uncertain)
            combined_uncertainty = entropy + (1.0 - margin)
            
            for i, score in enumerate(combined_uncertainty):
                uncertainty_scores.append((score.item(), batch_idx * batch_size + i))
    
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
    
    # Split data
    indices = list(range(len(full_train)))
    random.shuffle(indices)
    labeled = indices[:initial_labeled].copy()
    unlabeled = indices[initial_labeled:].copy()

    results = []
    for round_idx in range(num_rounds):
        print(f"  Round {round_idx + 1}/{num_rounds}, labeled: {len(labeled)}")
        
        # Train model from scratch on current labeled set
        model = TinyModel(num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam for faster convergence
        criterion = nn.CrossEntropyLoss()
        
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

        # Evaluate current model
        acc = evaluate(model, val_loader)
        results.append((len(labeled), acc))
        print(f"    Accuracy: {acc:.4f}")

        # Query new points using uncertainty strategy
        if len(unlabeled) > 0:
            # Create unlabeled dataloader
            unlabeled_subset = Subset(full_train, unlabeled)
            unlabeled_loader = DataLoader(unlabeled_subset, 
                                         batch_size=batch_size, 
                                         shuffle=False, 
                                         num_workers=0)
            
            # Compute uncertainty scores
            uncertainty_scores = compute_uncertainty_scores(model, unlabeled_loader)
            
            # Sort by uncertainty (highest first) and select top samples
            uncertainty_scores.sort(key=lambda x: x[0], reverse=True)
            
            # Select the most uncertain samples
            selected_count = min(query_size, len(unlabeled))
            selected_indices = [uncertainty_scores[i][1] for i in range(selected_count)]
            
            # Update labeled and unlabeled sets
            labeled.extend(selected_indices)
            unlabeled = [i for i in unlabeled if i not in selected_indices]

    return results

def run_random_active_learning_seed(seed):
    """Run random sampling active learning for a given seed"""
    print(f"Running random strategy with seed: {seed}")
    set_seeds(seed)
    
    # Split data
    indices = list(range(len(full_train)))
    random.shuffle(indices)
    labeled = indices[:initial_labeled].copy()
    unlabeled = indices[:initial_labeled:].copy()

    results = []
    for round_idx in range(num_rounds):
        print(f"  Round {round_idx + 1}/{num_rounds}, labeled: {len(labeled)}")
        
        # Train model from scratch on current labeled set
        model = TinyModel(num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam for faster convergence
        criterion = nn.CrossEntropyLoss()
        
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

        # Evaluate current model
        acc = evaluate(model, val_loader)
        results.append((len(labeled), acc))
        print(f"    Accuracy: {acc:.4f}")

        # Query new points using random strategy
        if len(unlabeled) > 0:
            selected_count = min(query_size, len(unlabeled))
            selected_indices = random.sample(unlabeled, selected_count)
            
            # Update labeled and unlabeled sets
            labeled.extend(selected_indices)
            unlabeled = [i for i in unlabeled if i not in selected_indices]

    return results

def main():
    """Ultra-fast sanity check test"""
    print("=== TINY SANITY CHECK TEST ===")
    print("Testing uncertainty vs random on small CIFAR-10 subset")
    print(f"Expected runtime: 2-5 minutes")
    print(f"Dataset: {max_train_samples} samples (vs 50,000 full CIFAR-10)")
    print(f"Model: Tiny CNN (vs ResNet18)")
    print(f"Training: {num_epochs} epochs, {num_rounds} rounds")
    
    # Use just 2 seeds for speed
    seeds = [42, 123]
    print(f"Using seeds: {seeds}")
    
    # Run both strategies
    print("\n1. Running Uncertainty-based Active Learning...")
    uncertainty_results = [run_uncertainty_active_learning_seed(seed) for seed in seeds]
    
    print("\n2. Running Random Active Learning...")
    random_results = [run_random_active_learning_seed(seed) for seed in seeds]
    
    # Compute average results
    print("\n3. Computing average results...")
    uncertainty_avg = []
    random_avg = []
    
    for round_idx in range(num_rounds):
        n_labeled = uncertainty_results[0][round_idx][0]
        
        # Average uncertainty results
        unc_acc = np.mean([res[round_idx][1] for res in uncertainty_results])
        uncertainty_avg.append((n_labeled, unc_acc))
        
        # Average random results
        rand_acc = np.mean([res[round_idx][1] for res in random_results])
        random_avg.append((n_labeled, rand_acc))
    
    # Print comparison
    print("\n=== QUICK TEST RESULTS ===")
    print("Labeled Samples | Uncertainty | Random   | Difference")
    print("-" * 50)
    
    total_uncertainty_win = 0
    for i, ((n_unc, acc_unc), (n_rand, acc_rand)) in enumerate(zip(uncertainty_avg, random_avg)):
        diff = acc_unc - acc_rand
        if diff > 0:
            total_uncertainty_win += 1
            status = "✓ UNC wins"
        elif diff < 0:
            status = "✗ RAND wins"
        else:
            status = "= TIE"
        
        print(f"{n_unc:14d} | {acc_unc:10.4f} | {acc_rand:7.4f} | {diff:+7.4f} {status}")
    
    # Final verdict
    print(f"\n=== QUICK TEST VERDICT ===")
    print(f"Uncertainty strategy wins: {total_uncertainty_win}/{len(uncertainty_avg)} rounds")
    
    if total_uncertainty_win > len(uncertainty_avg) * 0.7:
        print("✅ QUICK TEST PASSED: Uncertainty strategy consistently beats random!")
    elif total_uncertainty_win > len(uncertainty_avg) * 0.5:
        print("⚠️  QUICK TEST PARTIAL: Uncertainty strategy wins most rounds")
    else:
        print("❌ QUICK TEST FAILED: Random strategy wins too many rounds!")
    
    # Quick plot
    plt.figure(figsize=(8, 5))
    x_unc, y_unc = zip(*uncertainty_avg)
    x_rand, y_rand = zip(*random_avg)
    
    plt.plot(x_unc, y_unc, 'b-o', label='Uncertainty Strategy', linewidth=2)
    plt.plot(x_rand, y_rand, 'r-s', label='Random Strategy', linewidth=2)
    
    plt.xlabel("Number of Labeled Samples")
    plt.ylabel("Test Accuracy")
    plt.title("Tiny Sanity Check: Uncertainty vs Random")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('tiny_sanity_check.png', dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as 'tiny_sanity_check.png'")
    plt.show()
    
    print(f"\n=== SUMMARY ===")
    print(f"Dataset: {max_train_samples} CIFAR-10 samples (vs 50,000 full dataset)")
    print(f"Model: Tiny CNN (vs ResNet18)")
    print(f"Training: {num_epochs} epochs, {num_rounds} rounds")
    print(f"Runtime: ~2-5 minutes")
    print(f"Result: {'PASSED' if total_uncertainty_win > len(uncertainty_avg) * 0.7 else 'FAILED'}")

if __name__ == '__main__':
    main()
