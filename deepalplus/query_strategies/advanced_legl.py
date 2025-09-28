import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from .strategy import Strategy
from .base_handler import BaseNetHandler
from .common_nets import create_adapted_resnet18


class AdvancedLEGL_Backend(nn.Module):
    """
    Two-headed model for Advanced LEGL. The architecture is identical to the
    original LEGL_Backend, but it will be trained with an improved loss function.
    """

    def __init__(self, num_classes, pretrained=True, n_channels=3, dataset_name='cifar10'):
        super().__init__()

        # feature extractor
        self.features, self.feature_dim = create_adapted_resnet18(
            num_classes, pretrained, n_channels, dataset_name
        )

        # simple linear classifier head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        # regression head for predicting gradient norms
        self.legl_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):

        # embeddings from the feature extractor
        embeddings = self.features(x).view(x.size(0), -1)

        # getting the logits - scores for each class
        logits = self.classifier(embeddings)

        # getting the predicted gradient norms
        predicted_grad_norm = self.legl_head(embeddings)

        # returns embeddings as well for the efficient loss calculation
        return logits, predicted_grad_norm, embeddings


class Net_AdvancedLEGL(BaseNetHandler):
    """
    Network handler for the Advanced LEGL strategy.

    Supports two loss types for the regression head:
    - 'mse': The original Mean Squared Error loss.
    - 'rank': A more advanced pairwise Margin Ranking Loss.
    """

    def __init__(self, args_task, device, dataset_name, lambda_legl, loss_type='rank'):
        super().__init__(args_task, device, dataset_name)
        self.lambda_legl = lambda_legl
        if loss_type not in ['mse', 'rank']:
            raise ValueError("loss_type must be either 'mse' or 'rank'")
        self.loss_type = loss_type

    def _get_model_instance(self, n_channels):
        """Returns an instance of the custom two-headed LEGL model."""
        return AdvancedLEGL_Backend(
            num_classes=self.params['num_class'],
            pretrained=self.params['pretrained'],
            n_channels=n_channels,
            dataset_name=self.dataset_name
        )

    def train(self, data):
        """Custom training loop with selectable regression loss and uncertainty modulation."""
        self._check_and_create_model(data)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), **self.params['optimizer_args'])
        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])

        desc = f"Training Advanced LEGL (loss: {self.loss_type})"
        for epoch in tqdm(range(1, self.params['n_epoch'] + 1), ncols=100, desc=desc):
            for _, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                logits, predicted_scores, embeddings = self.model(x)
                loss_cls = F.cross_entropy(logits, y)

                # uses uncertainty-modulated norm as the regression target
                true_scores = self._calculate_uncertainty_modulated_norms(logits, y, embeddings)

                # uses a selectable, advanced loss function
                if self.loss_type == 'rank':
                    # pairs of samples from the batch
                    batch_size = len(y)
                    i = torch.randint(0, batch_size, (batch_size * 2,)).to(self.device)
                    j = torch.randint(0, batch_size, (batch_size * 2,)).to(self.device)

                    pred_i, pred_j = predicted_scores[i], predicted_scores[j]
                    true_i, true_j = true_scores[i], true_scores[j].detach()

                    # Target is 1 if score_i > score_j, -1 if score_i < score_j
                    target = torch.sign(true_i - true_j)
                    loss_reg = F.margin_ranking_loss(pred_i.squeeze(), pred_j.squeeze(), target, margin=0.1)
                else:  # 'mse'
                    loss_reg = F.mse_loss(predicted_scores.squeeze(), true_scores.detach())

                total_loss = loss_cls + self.lambda_legl * loss_reg
                total_loss.backward()
                optimizer.step()

    def _calculate_uncertainty_modulated_norms(self, logits, y, embeddings):
        """
        Calculates the true gradient norms and modulates them by predictive entropy.
        Samples that are uncertain (high entropy) will have their scores boosted.
        """
        # Calculate standard gradient norms
        nLab = self.params['num_class']
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(y, nLab)
        diff = probs - one_hot
        grad_embeddings = diff.unsqueeze(2) * embeddings.unsqueeze(1)
        true_norms = torch.norm(grad_embeddings.view(len(y), -1), dim=1)

        # Calculate predictive entropy for uncertainty
        log_probs = F.log_softmax(logits, dim=1)
        entropy = -torch.sum(probs * log_probs, dim=1)

        # Modulate the norm with entropy (add 1 to avoid multiplying by zero)
        modulated_norms = true_norms * (1 + entropy)
        return modulated_norms

    def predict(self, data):
        """Overrides base method to correctly handle the model's return signature."""
        self._check_and_create_model(data)
        self.model.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                logits, _, _ = self.model(x)
                pred = logits.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def predict_scores_and_embeddings(self, data):
        """
        Predicts both the informativeness scores and the feature embeddings for a dataset.
        This is a utility function required for the DiversityLEGL strategy.
        """
        self._check_and_create_model(data)
        self.model.eval()
        scores = torch.zeros(len(data))
        all_embeddings = torch.zeros(len(data), self.model.feature_dim)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                _, pred_scores, embeddings = self.model(x)
                scores[idxs] = pred_scores.squeeze().cpu()
                all_embeddings[idxs] = embeddings.cpu()
        return scores, all_embeddings


class AdvancedLEGL(Strategy):
    """
    Standard top-N query strategy using the advanced network.
    The "advancement" is in how the network was trained (e.g., with rank loss).
    """

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # We only need the scores for this strategy, not the embeddings
        scores, _ = self.net.predict_scores_and_embeddings(unlabeled_data)

        top_n_indices = scores.argsort(descending=True)[:n].cpu().numpy()
        return unlabeled_idxs[top_n_indices]


class DiversityLEGL(Strategy):
    """
    A diversity-aware query strategy.

    This strategy first uses the scores from the trained Advanced LEGL model to
    identify a pool of highly informative candidate samples. It then performs
    K-Means clustering on the feature embeddings of these candidates to select
    a final batch that is both informative and diverse.
    """

    def __init__(self, dataset, net, args_input, args_task):
        super().__init__(dataset, net, args_input, args_task)
        # Factor to determine the size of the candidate pool (e.g., 5*n)
        self.candidate_factor = getattr(self.args_input, "candidate_factor", 5.0)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # gets both scores and embeddings from the network
        scores, embeddings = self.net.predict_scores_and_embeddings(unlabeled_data)

        # filters for a candidate pool of the most informative samples
        num_candidates = min(len(scores), int(self.candidate_factor * n))
        candidate_local_indices = scores.argsort(descending=True)[:num_candidates]

        candidate_embeddings = embeddings[candidate_local_indices].cpu().numpy()
        candidate_global_idxs = unlabeled_idxs[candidate_local_indices.cpu().numpy()]

        # clusters the candidate embeddings to find n diverse representatives
        # uses n_init='auto' to avoid future warnings in scikit-learn
        kmeans = KMeans(n_clusters=n, n_init=10, random_state=0)
        kmeans.fit(candidate_embeddings)

        # selects the one sample from the candidate pool closest to each cluster centroid
        distances = euclidean_distances(kmeans.cluster_centers_, candidate_embeddings)

        # finds the index of the closest point in the candidate pool for each centroid
        final_candidate_indices = np.argmin(distances, axis=1)

        # returns the global indices of the selected diverse samples
        return candidate_global_idxs[final_candidate_indices]