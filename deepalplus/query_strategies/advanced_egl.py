import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .strategy import Strategy
from .base_egl_handler import BaseEGLNetHandler
from .egl_entropy_utils import modulate_scores_by_entropy
from .common_strategies import BaseDiversityStrategy


class Net_AdvancedEGL(BaseEGLNetHandler):
    """
    Network handler for the Advanced EGL strategy.

    This version enhances the standard EGL score by modulating it with
    predictive uncertainty (entropy) and provides embeddings for diversity-based methods.
    """

    def run_training_loop(self, loader, optimizer):
        """
        Overrides the base training loop to add a check for small batch sizes,
        which is specific to this strategy's original implementation to avoid
        BatchNorm errors.
        :param loader: DataLoader for the training data
        :param optimizer: Optimizer for model training
        """
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

    def calculate_advanced_egl_score_and_embedding(self, x, y, nLab):
        """
        Calculates uncertainty-modulated EGL scores and feature embeddings for a batch.
        :param x: Input batch of samples
        :param y: Labels for the input batch (used for getting batch size)
        :param nLab: Number of classes in the classification task
        :return: Tuple of (Tensor of modulated EGL scores, Tensor of embeddings)
        """
        norms, embeddings, logits, probs = self._calculate_base_egl_components(x, y, nLab)
        modulated_scores = modulate_scores_by_entropy(norms, logits)
        return modulated_scores, embeddings

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
                modulated_scores, embeddings = self.calculate_advanced_egl_score_and_embedding(x, y, nLab)
                scores[idxs] = modulated_scores.cpu()
                all_embeddings[idxs] = embeddings.cpu()

        return scores.numpy(), all_embeddings.numpy()


class AdvancedEGL(Strategy):
    """
    Standard top-N query strategy using the uncertainty-modulated EGL scores.
    """

    def query(self, n):
        """
        Queries the top-N samples with the highest uncertainty-modulated EGL scores.
        :param n: Number of samples to query
        :return: List of indices of the selected samples in the original dataset
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        scores, _ = self.net.get_scores_and_embeddings(unlabeled_data)
        top_n_indices = scores.argsort()[-n:]
        return unlabeled_idxs[top_n_indices]


class DiversityEGL(BaseDiversityStrategy):
    """
    A diversity-aware query strategy based on Advanced EGL.
    Inherits its core logic from BaseDiversityStrategy.
    """
    def _get_scores_and_embeddings(self, unlabeled_data):
        """Implements the required method to fetch scores and embeddings for EGL."""
        return self.net.get_scores_and_embeddings(unlabeled_data)