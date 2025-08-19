import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, Subset
from scipy.stats import zscore

from .strategy import Strategy


# Defines the custom model required for the Minds strategy.
class ActiveLearningModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.cls_head = nn.Linear(self.feature_dim, num_classes)
        # Update prediction head
        self.up_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        f = self.features(x).view(x.size(0), -1)
        # Returns classification logits and the predicted update magnitude
        return self.cls_head(f), self.up_head(f).squeeze(1)


# Minds class implementing the active learning strategy.
class Minds(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(Minds, self).__init__(dataset, net, args_input, args_task)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = self.args_task['num_class']
        self.model = ActiveLearningModel(self.num_classes).to(self.device)

    def _compute_true_grad_norm(self, x, y, criterion):
        """Computes the true gradient norm for a single sample."""
        self.model.zero_grad()
        logits, _ = self.model(x.unsqueeze(0))
        loss = criterion(logits, y.unsqueeze(0))

        params = list(self.model.features.parameters()) + list(self.model.cls_head.parameters())
        grads = torch.autograd.grad(loss, params, retain_graph=False)

        # Calculate the squared sum of gradients and return the square root.
        return torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))

    def train(self):
        """Custom training loop for the Minds strategy."""
        # Hyperparameters from avg_our_method.py
        num_epochs = self.args_task.get('n_epoch', 20)
        batch_size = self.args_task['loader_tr_args']['batch_size']
        margin = 0.1
        lambda_rank = 0.01

        # Prepare data loader for labeled data
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        train_loader = DataLoader(labeled_data, batch_size=batch_size, shuffle=True, num_workers=0)

        # Setup optimizer and loss function
        optimizer = optim.SGD(self.model.parameters(), lr=self.args_task['optimizer_args']['lr'], momentum=0.9)
        criterion_cls = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(num_epochs):
            for x_batch, y_batch, _ in train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                logits, pred_update = self.model(x_batch)
                loss_cls = criterion_cls(logits, y_batch)

                # Compute true gradient norms for the batch
                true_norms = torch.stack([
                    self._compute_true_grad_norm(xi, yi, criterion_cls)
                    for xi, yi in zip(x_batch, y_batch)
                ])

                # Compute ranking loss
                idx_perm = torch.randperm(len(true_norms), device=self.device)

                if len(true_norms) > 1:
                    ranking_loss = sum(
                        F.relu(-torch.sign(true_norms[a] - true_norms[b]) * (pred_update[a] - pred_update[b]) + margin)
                        for a, b in zip(idx_perm[::2], idx_perm[1::2])
                    ) / (len(true_norms) // 2)
                else:
                    ranking_loss = 0  # Cannot compute ranking loss for a single item

                # Total loss
                loss = loss_cls + lambda_rank * ranking_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def query(self, n):
        """Query new points based on the predicted update magnitude."""
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        pool_loader = DataLoader(unlabeled_data, batch_size=self.args_task['loader_te_args']['batch_size'],
                                 shuffle=False, num_workers=0)

        self.model.eval()
        scores = []
        with torch.no_grad():
            for x_pool, _, idxs in pool_loader:
                x_pool = x_pool.to(self.device)
                _, preds = self.model(x_pool)
                scores.extend(zip(preds.cpu().numpy(), idxs.numpy()))

        # Selection strategy from avg_our_method.py
        scores.sort(key=lambda t: t[0])
        discard = int(0.1 * len(scores))
        candidates = scores[discard:]

        # Randomly sample from the top candidates
        selected_indices = [idx for _, idx in random.sample(candidates, k=min(n, len(candidates)))]

        return unlabeled_idxs[selected_indices]

    def predict(self, data):
        """Make predictions with the trained model."""
        loader = DataLoader(data, shuffle=False, **self.args_task['loader_te_args'])
        preds = torch.zeros(len(data), dtype=data.Y.dtype)

        self.model.eval()
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                logits, _ = self.model(x)
                pred = logits.argmax(dim=1)
                preds[idxs] = pred.cpu()

        return preds