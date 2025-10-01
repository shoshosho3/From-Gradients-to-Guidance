import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .strategy import Strategy
from .base_handler import BaseNetHandler
from .common_nets import StandardResNet18


class Net_VanillaEGL(BaseNetHandler):
    """
    Network handler for the original Vanilla EGL strategy.

    This class computes the standard Expected Gradient Length (EGL) score, which
    is the L2 norm of the gradient of the loss with respect to the final layer's
    weights, using the model's own prediction as a pseudo-label.
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
        """Standard classification training loop."""
        self._check_and_create_model(data)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), **self.params['optimizer_args'])
        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])

        for epoch in tqdm(range(1, self.params['n_epoch'] + 1), ncols=100, desc="Training VanillaEGL"):
            for _, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

    def get_scores(self, data):
        """
        Calculates the original Expected Gradient Length (EGL) scores.

        The score is simply the L2 norm of the gradient embedding, without
        any additional modulation.
        """
        self._check_and_create_model(data)
        self.model.eval()

        nLab = self.params['num_class']
        scores = torch.zeros(len(data))
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

                # The score is the L2 norm of the gradient embedding
                norms = torch.norm(grad_emb.view(len(y), -1), dim=1)

                scores[idxs] = norms.cpu()

        return scores.numpy()

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


class VanillaEGL(Strategy):
    """
    Standard top-N query strategy using the original EGL scores.

    It selects the samples with the largest expected gradient length, which are
    considered the most informative for the model.
    """

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        scores = self.net.get_scores(unlabeled_data)

        # Select the indices with the highest scores
        top_n_indices = scores.argsort()[-n:]
        return unlabeled_idxs[top_n_indices]