import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .strategy import Strategy
from .base_handler import BaseNetHandler
from .common_nets import StandardResNet18


class Net_EGL(BaseNetHandler):
    """
    Network handler for the EGL strategy.
    This version uses a vectorized approach for superior performance.
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

        for epoch in tqdm(range(1, self.params['n_epoch'] + 1), ncols=100, desc="Training EGL"):
            for _, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

    def get_grad_embeddings(self, data):
        """
        Calculates gradient embeddings using pseudo-labels.
        This is a highly efficient, vectorized implementation of the EGL score calculation,
        mathematically equivalent to the loop-based version in the original egl.py.
        """
        self._check_and_create_model(data)
        self.model.eval()

        embDim = self.model.get_embedding_dim()
        nLab = self.params['num_class']
        grad_embeddings = np.zeros([len(data), embDim * nLab])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])

        with torch.no_grad():
            for x, y, idxs in loader:
                x = x.to(self.device)

                logits, embeddings = self.model(x)

                # Use argmax on logits for numerical stability and efficiency
                pred_labels = torch.argmax(logits, dim=1)
                probs = F.softmax(logits, dim=1)

                # EGL formula: G = flatten((P - Y_pseudo) ⨂ E)
                # where P are probabilities, Y_pseudo is one-hot pseudo-label, E is embedding.
                one_hot = F.one_hot(pred_labels, nLab)
                diff = probs - one_hot  # Shape: (batch, nLab)

                # Perform outer product for each sample in the batch via broadcasting
                # (batch, nLab, 1) * (batch, 1, embDim) -> (batch, nLab, embDim)
                grad = diff.unsqueeze(2) * embeddings.unsqueeze(1)

                # Flatten to (batch, nLab * embDim) and store
                grad_embeddings[idxs] = grad.view(len(y), -1).cpu().numpy()

        return grad_embeddings


class EGL(Strategy):
    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        grad_embeddings = self.net.get_grad_embeddings(unlabeled_data)
        # The score is the L2 norm of the gradient embedding vector.
        scores = np.linalg.norm(grad_embeddings, axis=1)
        top_n_indices = scores.argsort()[-n:]
        return unlabeled_idxs[top_n_indices]


class REGL(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super().__init__(dataset, net, args_input, args_task)
        self.factor = getattr(self.args_input.strategy_args, "factor", 1.0)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        grad_embeddings = self.net.get_grad_embeddings(unlabeled_data)
        scores = np.linalg.norm(grad_embeddings, axis=1)

        # Create a candidate pool from the top-scoring samples.
        num_candidates = min(len(scores), int(self.factor * n))
        candidate_indices = scores.argsort()[-num_candidates:]

        # Randomly select n samples from the candidate pool.
        selected_local_indices = np.random.choice(candidate_indices, n, replace=False)
        return unlabeled_idxs[selected_local_indices]