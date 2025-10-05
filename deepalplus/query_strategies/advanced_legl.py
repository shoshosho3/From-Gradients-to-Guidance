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
from .common_strategies import BaseDiversityStrategy
from .egl_entropy_utils import modulate_scores_by_entropy


class AdvancedLEGL_Backend(nn.Module):
    """
    Two-headed model for Advanced LEGL. The architecture is identical to the
    original LEGL_Backend, but it will be trained with an improved loss function.
    """

    def __init__(self, num_classes, pretrained=True, n_channels=3, dataset_name='cifar10'):
        """
        Initializes the Advanced LEGL backend model with a feature extractor,
        :param num_classes: Number of classes in the classification task
        :param pretrained: Whether to use pretrained weights for the feature extractor
        :param n_channels: Number of input channels in the data
        :param dataset_name: Name of the dataset (for architecture adaptation)
        """

        super().__init__()

        # creating the shared feature extractor
        self.features, self.feature_dim = create_adapted_resnet18(
            num_classes, pretrained, n_channels, dataset_name
        )

        # creating the two heads
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        self.legl_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        """
        Forward pass through the shared feature extractor and both heads.
        :param x: Input batch of samples
        :return: Tuple of (logits, predicted gradient norm, embeddings)
        """
        embeddings = self.features(x).view(x.size(0), -1)
        logits = self.classifier(embeddings)
        predicted_grad_norm = self.legl_head(embeddings)
        return logits, predicted_grad_norm, embeddings


class Net_AdvancedLEGL(BaseNetHandler):
    """
    Network handler for the Advanced LEGL strategy.
    Supports 'mse' and 'rank' loss for the regression head.
    """
    def __init__(self, args_task, device, dataset_name, lambda_legl, loss_type='rank'):
        """
        Initializes the Advanced LEGL network handler with specified parameters.
        :param args_task: task-specific arguments
        :param device: computation device (CPU or GPU)
        :param dataset_name: name of the dataset
        :param lambda_legl: weighting factor for the LEGL loss component
        :param loss_type: type of regression loss ('mse' or 'rank')
        """

        super().__init__(args_task, device, dataset_name)
        self.lambda_legl = lambda_legl
        if loss_type not in ['mse', 'rank']:
            raise ValueError("loss_type must be either 'mse' or 'rank'")
        self.loss_type = loss_type

    def _get_model_instance(self, n_channels):
        """
        Returns an instance of the Advanced LEGL backend model.
        :param n_channels: Number of input channels in the data
        :return: An instance of AdvancedLEGL_Backend
        """
        return AdvancedLEGL_Backend(
            num_classes=self.params['num_class'],
            pretrained=self.params['pretrained'],
            n_channels=n_channels,
            dataset_name=self.dataset_name
        )

    def rank_loss(self, predicted_scores, true_scores, y):
        """
        Computes the pairwise ranking loss for the predicted and true scores.
        :param predicted_scores: the scores predicted by the model
        :param true_scores: the ground truth scores
        :param y:  labels for the input batch (used for getting batch size)
        :return: computed ranking loss
        """

        batch_size = len(y)
        i = torch.randint(0, batch_size, (batch_size * 2,)).to(self.device)
        j = torch.randint(0, batch_size, (batch_size * 2,)).to(self.device)
        pred_i, pred_j = predicted_scores[i], predicted_scores[j]
        true_i, true_j = true_scores[i], true_scores[j].detach()
        target = torch.sign(true_i - true_j)
        return F.margin_ranking_loss(pred_i.squeeze(), pred_j.squeeze(), target, margin=0.1)

    def calculate_loss_reg(self, logits, predicted_scores, y, embeddings):
        """
        Calculates the regression loss for the LEGL head, using either MSE or ranking loss.
        :param logits: Logits from the model
        :param predicted_scores: Scores predicted by the LEGL head
        :param y: True labels for the input batch
        :param embeddings: Feature embeddings from the model
        :return: Computed regression loss
        """
        true_scores = self._calculate_uncertainty_modulated_norms(logits, y, embeddings)
        if self.loss_type == 'rank':
            return self.rank_loss(predicted_scores, true_scores, y)
        return F.mse_loss(predicted_scores.squeeze(), true_scores.detach())

    def compute_total_loss(self, x, y):
        """
        Computes the total loss as the sum of classification loss and weighted regression loss.
        :param x: Input batch of samples
        :param y: True labels for the input batch
        :return: Computed total loss
        """
        logits, predicted_scores, embeddings = self.model(x)
        loss_cls = F.cross_entropy(logits, y)
        loss_reg = self.calculate_loss_reg(logits, predicted_scores, y, embeddings)
        return loss_cls + self.lambda_legl * loss_reg

    def run_training_loop(self, loader, optimizer, desc):
        """
        Custom training loop that incorporates the LEGL loss component.
        :param loader: DataLoader for the training data
        :param optimizer: Optimizer for model training
        :param desc: Description for the progress bar
        """

        for epoch in tqdm(range(1, self.params['n_epoch'] + 1), ncols=100, desc=desc):
            for _, (x, y, idxs) in enumerate(loader):
                # skips batches with 1 or 0 samples to avoid BatchNorm error during training.
                if x.shape[0] <= 1:
                    continue
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                total_loss = self.compute_total_loss(x, y)
                total_loss.backward()
                optimizer.step()

    def train(self, data):
        """
        Custom training loop with selectable regression loss and uncertainty modulation.
        :param data: training data
        """
        self._check_and_create_model(data)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), **self.params['optimizer_args'])
        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
        desc = f"Training Advanced LEGL (loss: {self.loss_type})"
        self.run_training_loop(loader, optimizer, desc)


    def _calculate_uncertainty_modulated_norms(self, logits, y, embeddings):
        """
        Calculates the true gradient norms and modulates them by predictive entropy.

        :param logits: Logits from the model
        :param y: True labels for the input batch
        :param embeddings: Feature embeddings from the model
        :return: Tensor of modulated gradient norms
        """
        nLab = self.params['num_class']
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(y, nLab)
        diff = probs - one_hot # the gradient part for cross-entropy loss
        grad_embeddings = diff.unsqueeze(2) * embeddings.unsqueeze(1) # the gradient embeddings
        true_norms = torch.norm(grad_embeddings.view(len(y), -1), dim=1) # the egl norms

        # using the centralized utility function for modulation
        modulated_norms = modulate_scores_by_entropy(true_norms, logits)
        return modulated_norms

    def predict(self, data):
        """
        Generic prediction logic for classification.
        :param data: Dataset for prediction
        :return: Tensor of predicted class labels
        """
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
        Predicts scores and embeddings for a dataset.

        :param data: Dataset for prediction
        :return: Tuple of (numpy array of predicted scores, numpy array of embeddings)
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
        return scores.numpy(), all_embeddings.numpy()


class AdvancedLEGL(Strategy):
    """
    Standard top-N query strategy using the advanced network's predicted scores.
    """

    def query(self, n):
        """
        Queries the top-N samples with the highest predicted scores.
        :param n: Number of samples to query
        :return: Indices of the queried samples
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        scores, _ = self.net.predict_scores_and_embeddings(unlabeled_data)
        top_n_indices = scores.argsort()[-n:]
        return unlabeled_idxs[top_n_indices]


class DiversityLEGL(BaseDiversityStrategy):
    """A diversity-aware query strategy that inherits its logic from BaseDiversityStrategy."""
    def _get_scores_and_embeddings(self, unlabeled_data):
        """Implements the required method to fetch scores and embeddings."""
        return self.net.predict_scores_and_embeddings(unlabeled_data)

    def _select_candidates(self, scores, num_candidates):
        """
        This function exists for reproducibility purposes, the sorting is slightly different
        """
        return (-scores).argsort()[:num_candidates]