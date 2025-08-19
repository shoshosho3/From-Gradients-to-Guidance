import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader
from tqdm import tqdm

from .strategy import Strategy


# This is the custom network architecture required for the Minds strategy.
# It has two heads: one for classification (cls_head) and one for predicting the update magnitude (up_head).
class Minds_Net_Backend(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Adapt the first layer for single-channel datasets like MNIST
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

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
        return self.cls_head(f), self.up_head(f).squeeze(dim=-1)


# This class integrates the Minds model and its custom training loop into the deepAL+ framework,
# similar to how Net_LPL or Net_WAAL are structured.
class Net_Minds:
    def __init__(self, args_task, device):
        self.params = args_task
        self.device = device
        self.num_classes = self.params['num_class']
        self.model = Minds_Net_Backend(self.num_classes, self.params['pretrained']).to(self.device)

    def train(self, data):
        """
        Optimized training loop for the Minds strategy.
        This is now the main training entry point, called by the Strategy class.
        """
        num_epochs = self.params.get('n_epoch', 20)
        optimizer = optim.SGD(self.model.parameters(), **self.params['optimizer_args'])
        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])

        # We need two loss functions.
        # One with `reduction='mean'` for the main classification task.
        criterion_cls = nn.CrossEntropyLoss(reduction='mean')
        # One with `reduction='none'` to get per-sample losses for the ranking task.
        criterion_cls_none = nn.CrossEntropyLoss(reduction='none')

        # Hyperparameters for the ranking loss
        margin = 0.1
        lambda_rank = 0.01

        self.model.train()
        for epoch in tqdm(range(num_epochs), desc="Training MindsNet"):
            for x_batch, y_batch, _ in loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                logits, pred_update = self.model(x_batch)

                # 1. Classification Loss (standard)
                loss_cls = criterion_cls(logits, y_batch)

                # 2. Ranking Loss (optimized)
                # We use the true per-sample loss as a fast and effective proxy for the gradient norm.
                with torch.no_grad():
                    true_update_proxy = criterion_cls_none(logits, y_batch)

                # Create random pairs for ranking
                idx_perm = torch.randperm(len(true_update_proxy), device=self.device)

                if len(true_update_proxy) > 1:
                    # Detach the proxy tensor as it's a target, not part of the computation graph for ranking loss
                    true_update_proxy = true_update_proxy.detach()

                    p1_proxy, p2_proxy = true_update_proxy[idx_perm[::2]], true_update_proxy[idx_perm[1::2]]
                    p1_pred, p2_pred = pred_update[idx_perm[::2]], pred_update[idx_perm[1::2]]

                    # Calculate ranking loss on the pairs
                    ranking_loss = F.relu(-torch.sign(p1_proxy - p2_proxy) * (p1_pred - p2_pred) + margin).mean()
                else:
                    ranking_loss = 0.0

                # 3. Total Loss
                loss = loss_cls + lambda_rank * ranking_loss
                loss.backward()
                optimizer.step()

    def predict_prob(self, data):
        """Predict class probabilities."""
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        self.model.eval()
        probs = torch.zeros([len(data), self.num_classes])
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                logits, _ = self.model(x)
                prob = F.softmax(logits, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_update_scores(self, data):
        """Predict the update magnitude scores for querying."""
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        self.model.eval()
        scores = torch.zeros(len(data))
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                _, pred_updates = self.model(x)
                scores[idxs] = pred_updates.cpu()
        return scores

    def get_model(self):
        return self.model


# This is the main Strategy class that will be used by demo.py
class Minds(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(Minds, self).__init__(dataset, net, args_input, args_task)

    def train(self):
        # The training is now delegated to the network class, following the framework's design.
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data)

    def query(self, n):
        """Query new points based on the predicted update magnitude."""
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Use the network to get scores
        pred_updates = self.net.predict_update_scores(unlabeled_data)
        scores = list(zip(pred_updates.numpy(), unlabeled_idxs))

        # Sort by score (ascending)
        scores.sort(key=lambda t: t[0])

        # Discard 10% of samples with the lowest scores
        discard = int(0.1 * len(scores))
        candidates = scores[discard:]

        # Randomly sample 'n' points from the remaining high-score candidates
        selected = random.sample(candidates, k=min(n, len(candidates)))

        return [idx for _, idx in selected]

    def predict(self, data):
        """Make predictions using the trained model."""
        probs = self.net.predict_prob(data)
        return torch.max(probs, 1)[1]