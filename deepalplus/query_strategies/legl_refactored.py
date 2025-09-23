import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .strategy import Strategy
from .base_handler import BaseNetHandler
from .common_nets import create_adapted_resnet18


class LEGL_Backend(nn.Module):
    """Two-headed model for LEGL: classification and gradient norm prediction."""

    def __init__(self, num_classes, pretrained=True, n_channels=3, dataset_name='cifar10'):
        super().__init__()
        self.features, self.feature_dim = create_adapted_resnet18(
            num_classes, pretrained, n_channels, dataset_name
        )
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.legl_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        embeddings = self.features(x).view(x.size(0), -1)
        logits = self.classifier(embeddings)
        predicted_grad_norm = self.legl_head(embeddings)
        return logits, predicted_grad_norm


class Net_LEGL(BaseNetHandler):
    """Network handler for the LEGL strategy."""

    def __init__(self, args_task, device, dataset_name, lambda_legl):
        super().__init__(args_task, device, dataset_name)
        # Pull hyperparameter from config instead of hard-coding
        self.lambda_legl = lambda_legl

    def _get_model_instance(self, n_channels):
        """Returns an instance of the custom two-headed LEGL model."""
        return LEGL_Backend(
            num_classes=self.params['num_class'],
            pretrained=self.params['pretrained'],
            n_channels=n_channels,
            dataset_name=self.dataset_name
        )

    def train(self, data):
        """Custom training loop for the combined classification and regression loss."""
        self._check_and_create_model(data)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), **self.params['optimizer_args'])
        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])

        for epoch in tqdm(range(1, self.params['n_epoch'] + 1), ncols=100, desc="Training LEGL"):
            for _, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                # Forward pass and classification loss
                logits, predicted_norms = self.model(x)
                loss_cls = F.cross_entropy(logits, y)

                # ground-truth gradient norms
                true_norms = self._calculate_true_grad_norms(logits, y)

                # regression loss
                loss_reg = F.mse_loss(predicted_norms.squeeze(), true_norms.detach())

                # Combine losses and backpropagate
                total_loss = loss_cls + self.lambda_legl * loss_reg
                total_loss.backward()
                optimizer.step()

    def _calculate_true_grad_norms(self, logits, y):
        """
        Calculates the true gradient norms for each sample in the batch.
        :param logits: the model's output logits
        :param y: the true labels
        :return: a tensor of gradient norms
        """
        true_norms = []
        final_layer_params = list(self.model.classifier.parameters())

        for i in range(len(logits)):
            sample_loss = F.cross_entropy(logits[i].unsqueeze(0), y[i].unsqueeze(0))
            grads = torch.autograd.grad(sample_loss, final_layer_params, retain_graph=True)
            grad_vec = torch.cat([g.view(-1) for g in grads])
            true_norms.append(grad_vec.norm())

        return torch.stack(true_norms).to(self.device)

    def predict_legl_scores(self, data):
        """Predicts informativeness scores using the trained LEGL head."""
        self._check_and_create_model(data)
        self.model.eval()
        scores = torch.zeros(len(data))
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                _, pred_scores = self.model(x)
                scores[idxs] = pred_scores.squeeze().cpu()
        return scores


class LEGL(Strategy):
    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        scores = self.net.predict_legl_scores(unlabeled_data)
        top_n_indices = scores.argsort(descending=True)[:n]
        return unlabeled_idxs[top_n_indices]


class RLEGL(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super().__init__(dataset, net, args_input, args_task)
        self.factor = self.args_input['strategy_args'].get('factor', 1.0)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        scores = self.net.predict_legl_scores(unlabeled_data)

        num_candidates = min(len(scores), int(self.factor * n))
        candidate_indices = scores.argsort(descending=True)[:num_candidates]

        selected_local_indices = np.random.choice(candidate_indices.cpu(), n, replace=False)
        return unlabeled_idxs[selected_local_indices]