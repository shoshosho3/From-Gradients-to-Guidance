import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import random
from .strategy import Strategy

# Use CUDA if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Minds(Strategy):
    """
    Implements the MINDS (Margin-based Informativeness and Diversity Score) active learning strategy.

    This strategy is based on the provided `avg_our_method.py` script. It trains a model
    with a dual objective: a standard classification loss and a ranking loss. The ranking 
    loss encourages an auxiliary 'update prediction' head to estimate the true gradient norm 
    of each sample. The goal is for the model to learn to predict which samples will cause 
    the largest model updates.

    During querying, samples with high predicted update values (i.e., high expected gradient
    norms) are selected. A small fraction of the lowest-scoring samples are discarded, and
    the final selection is made by randomly sampling from the remaining high-scoring candidates.

    Assumptions:
    - The `net` object provided during initialization contains a PyTorch model accessible via `net.clf`.
    - This model (`net.clf`) must have an architecture similar to `ActiveLearningModel` from the script,
      specifically with `features`, `cls_head`, and `up_head` components. The forward pass must
      return two outputs: classification logits and the update prediction scalar.
    - The `dataset` object has a `self.dataset.labeled_idxs` boolean mask and provides access to the
      full underlying PyTorch dataset via `self.dataset.handler` for creating `Subset`s.
    """

    def __init__(self, dataset, net, args_input, args_task):
        super().__init__(dataset, net, args_input, args_task)

        # Hyperparameters from the provided script, fetched from args
        self.margin = 0.1
        self.lambda_rank = 0.01
        self.num_epochs = 20
        self.batch_size = 32
        self.lr = 1e-3
        self.momentum = 0.9
        self.discard_ratio = 0.1

    def _compute_true_grad_norm(self, x, y, criterion):
        """
        Computes the true L2 norm of the gradients for a single sample with respect
        to the feature extractor and classification head parameters.
        """
        model = self.net.clf
        model.to(device)
        x, y = x.to(device), y.to(device)

        model.zero_grad()
        logits, _ = model(x.unsqueeze(0))
        loss = criterion(logits, y.unsqueeze(0))

        # Parameters for which to compute gradients, as in the original script
        try:
            params = list(model.features.parameters()) + list(model.cls_head.parameters())
        except AttributeError:
            raise AttributeError("The model in 'net.clf' must have 'features' and 'cls_head' attributes.")

        grads = torch.autograd.grad(loss, params, retain_graph=False)
        grad_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
        return grad_norm

    def train(self, data=None, model_name=None):
        """
        Overrides the parent training method to implement the custom training loop
        with both a classification loss and a pairwise ranking loss.
        """
        model = self.net.clf
        model.to(device)

        labeled_idxs, labeled_data = self.dataset.get_labeled_data()

        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        criterion_cls = nn.CrossEntropyLoss()
        train_loader = DataLoader(labeled_data, batch_size=self.batch_size, shuffle=True)

        model.train()
        for epoch in range(self.num_epochs):
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

                # Forward pass
                logits, pred_update = model(x_batch)

                # 1. Classification loss
                loss_cls = criterion_cls(logits, y_batch)

                # 2. Ranking loss
                # Compute true gradient norms for each sample in the batch
                true_norms = torch.stack([
                    self._compute_true_grad_norm(xi, yi, criterion_cls)
                    for xi, yi in zip(x_batch, y_batch)
                ])

                ranking_loss = torch.tensor(0.0, device=device)
                if len(true_norms) > 1:
                    # Create random pairs for comparison
                    idx_perm = torch.randperm(len(true_norms), device=device)
                    pairs1 = idx_perm[::2]
                    pairs2 = idx_perm[1::2]

                    min_len = min(len(pairs1), len(pairs2))
                    if min_len > 0:
                        tn1 = true_norms[pairs1[:min_len]]
                        tn2 = true_norms[pairs2[:min_len]]
                        pu1 = pred_update[pairs1[:min_len]]
                        pu2 = pred_update[pairs2[:min_len]]

                        # Vectorized ranking loss (hinge loss)
                        ranking_loss = F.relu(-torch.sign(tn1 - tn2) * (pu1 - pu2) + self.margin).mean()

                # Total loss
                loss = loss_cls + self.lambda_rank * ranking_loss

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def query(self, n):
        """
        Selects a new batch of `n` samples from the unlabeled pool to be labeled.
        """
        unlabeled_idxs, unlabeled_subset = self.get_unlabeled_data()
        pool_loader = DataLoader(unlabeled_subset, batch_size=self.batch_size, shuffle=False)

        model = self.net.clf
        model.to(device)
        model.eval()


        scores = []
        with torch.no_grad():
            current_offset = 0
            for x_pool, _ in pool_loader:
                x_pool = x_pool.to(device)
                _, pred_update = model(x_pool)

                for j, score in enumerate(pred_update):
                    global_idx = unlabeled_idxs[current_offset + j]
                    scores.append((score.item(), global_idx))
                current_offset += len(x_pool)

        # Sort scores by predicted update value (ascending)
        scores.sort(key=lambda t: t[0])

        # Discard the bottom `discard_ratio` % of samples
        discard_count = int(self.discard_ratio * len(scores))
        candidates = scores[discard_count:]

        # Randomly sample `n` points from the remaining high-score candidates
        num_to_sample = min(n, len(candidates))
        if num_to_sample > 0:
            selected_tuples = random.sample(candidates, k=num_to_sample)
            queried_idxs = [idx for _, idx in selected_tuples]
        else:
            queried_idxs = []

        return np.array(queried_idxs)

    def get_unlabeled_data(self):
        """Helper to get unlabeled indices and a corresponding PyTorch Subset."""
        unlabeled_idxs = np.where(self.dataset.labeled_idxs == 0)[0]
        # Assumes the full dataset is accessible via self.dataset.handler
        unlabeled_subset = Subset(self.dataset.handler, unlabeled_idxs)
        return unlabeled_idxs, unlabeled_subset