import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

from .strategy import Strategy
from .base_handler import BaseNetHandler
from .common_nets import StandardResNet18


class Net_AdvancedEGL(BaseNetHandler):
    """
    Network handler for the Advanced EGL strategy.

    This version enhances the standard EGL score by modulating it with
    predictive uncertainty (entropy) and provides embeddings for diversity-based methods.
    """

    def _get_model_instance(self, n_channels):
        """Returns an instance of the standard, adapted ResNet18."""
        return StandardResNet18(
            num_classes=self.params['num_class'],
            pretrained=self.params['pretrained'],
            n_channels=n_channels,
            dataset_name=self.dataset_name
        )

    def train(self, data):
        """Standard classification training loop, same as for EGL."""
        self._check_and_create_model(data)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), **self.params['optimizer_args'])
        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])

        for epoch in tqdm(range(1, self.params['n_epoch'] + 1), ncols=100, desc="Training AdvancedEGL"):
            for _, (x, y, idxs) in enumerate(loader):

                # skips batches with 1 or 0 samples to avoid BatchNorm error during training.
                if x.shape[0] <= 1:
                    continue

                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

    def get_scores_and_embeddings(self, data):
        """
        Calculates uncertainty-modulated EGL scores and feature embeddings.

        The score is the L2 norm of the gradient embedding, multiplied by a
        factor related to the predictive entropy to boost uncertain samples.
        """
        self._check_and_create_model(data)
        self.model.eval()

        embDim = self.model.get_embedding_dim()
        nLab = self.params['num_class']

        scores = torch.zeros(len(data))
        all_embeddings = torch.zeros(len(data), embDim)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])

        with torch.no_grad():
            for x, y, idxs in loader:
                x = x.to(self.device)
                logits, embeddings = self.model(x)
                probs = F.softmax(logits, dim=1)

                # Calculate standard gradient norms using pseudo-labels
                pred_labels = torch.argmax(logits, dim=1)
                one_hot = F.one_hot(pred_labels, nLab)
                diff = probs - one_hot
                grad_emb = diff.unsqueeze(2) * embeddings.unsqueeze(1)
                norms = torch.norm(grad_emb.view(len(y), -1), dim=1)

                # Calculate predictive entropy for uncertainty
                log_probs = F.log_softmax(logits, dim=1)
                entropy = -torch.sum(probs * log_probs, dim=1)

                # Modulate the norm with entropy
                modulated_scores = norms * (1 + entropy)

                scores[idxs] = modulated_scores.cpu()
                all_embeddings[idxs] = embeddings.cpu()

        return scores.numpy(), all_embeddings.numpy()

    def predict(self, data):
        """Standard prediction method."""
        self._check_and_create_model(data)
        self.model.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                logits, _ = self.model(x)
                pred = logits.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds


class AdvancedEGL(Strategy):
    """
    Standard top-N query strategy using the uncertainty-modulated EGL scores.
    """

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # We only need the scores for this strategy, not the embeddings
        scores, _ = self.net.get_scores_and_embeddings(unlabeled_data)

        # Select the indices with the highest scores
        top_n_indices = scores.argsort()[-n:]
        return unlabeled_idxs[top_n_indices]


class DiversityEGL(Strategy):
    """
    A diversity-aware query strategy based on Advanced EGL.

    It first filters a pool of informative candidates using uncertainty-modulated
    EGL scores. It then performs K-Means clustering on the feature embeddings
    of these candidates to select a final batch that is both informative and diverse.
    """

    def __init__(self, dataset, net, args_input, args_task):
        super().__init__(dataset, net, args_input, args_task)
        # Factor to determine the size of the candidate pool (e.g., 5*n)
        self.candidate_factor = getattr(self.args_input, "candidate_factor", 5.0)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Gets both scores and embeddings from the network
        scores, embeddings = self.net.get_scores_and_embeddings(unlabeled_data)

        # Filters for a candidate pool of the most informative samples
        num_candidates = min(len(scores), int(self.candidate_factor * n))
        candidate_local_indices = scores.argsort()[-num_candidates:]

        candidate_embeddings = embeddings[candidate_local_indices]
        candidate_global_idxs = unlabeled_idxs[candidate_local_indices]

        # Clusters the candidate embeddings to find n diverse representatives
        # uses n_init='auto' to avoid future warnings in scikit-learn
        kmeans = KMeans(n_clusters=n, n_init=10, random_state=0)
        kmeans.fit(candidate_embeddings)

        # Selects the one sample from the candidate pool closest to each cluster centroid
        distances = euclidean_distances(kmeans.cluster_centers_, candidate_embeddings)

        # Finds the index of the closest point in the candidate pool for each centroid
        final_candidate_indices = np.argmin(distances, axis=1)

        # Returns the global indices of the selected diverse samples
        return candidate_global_idxs[final_candidate_indices]