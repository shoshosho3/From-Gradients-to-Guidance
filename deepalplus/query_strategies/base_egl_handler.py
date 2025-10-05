import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base_handler import BaseNetHandler
from .common_nets import StandardResNet18


class BaseEGLNetHandler(BaseNetHandler):
    """
    A base network handler for EGL-based strategies.

    This class contains the common components for model creation, training,
    prediction, and the core EGL norm calculation. Subclasses should
    implement their specific scoring logic.
    """

    def _get_model_instance(self, n_channels):
        """
        This method returns an instance of the standard, adapted ResNet18.
        :param n_channels: Number of input channels in the data
        :return: An instance of StandardResNet18
        """
        return StandardResNet18(
            num_classes=self.params['num_class'],
            pretrained=self.params['pretrained'],
            n_channels=n_channels,
            dataset_name=self.dataset_name
        )

    def run_training_loop(self, loader, optimizer):
        """
        Runs the standard training loop for a specified number of epochs.
        :param loader: DataLoader for the training data
        :param optimizer: Optimizer for model training
        """
        for epoch in tqdm(range(1, self.params['n_epoch'] + 1), ncols=100, desc=f"Training {type(self).__name__}"):
            for _, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

    def train(self, data):
        """
        Standard classification training setup.
        :param data: Labeled dataset for training
        """
        self._check_and_create_model(data)
        self.model.train()

        optimizer = optim.Adam(self.model.parameters(), **self.params['optimizer_args'])
        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])

        self.run_training_loop(loader, optimizer)

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
                logits, _ = self.model(x)
                pred = logits.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def _forward_and_get_outputs(self, x):
        """
        Forward pass to get logits, embeddings, and probabilities.
        :param x: Input data tensor
        :return: Tuple of (logits, embeddings, probabilities)
        """
        x = x.to(self.device)
        logits, embeddings = self.model(x)
        probs = F.softmax(logits, dim=1)
        return logits, embeddings, probs

    def _compute_gradient_norms(self, probs, logits, embeddings, y, nLab):
        """
        Computes the gradient norms for EGL.
        :param probs: Predicted class probabilities
        :param logits: Logits from the model
        :param embeddings: Embeddings from the model
        :param y: True labels
        :param nLab: Number of classes
        :return: Tensor of gradient norms
        """
        pred_labels = torch.argmax(logits, dim=1)
        one_hot = F.one_hot(pred_labels, nLab)
        diff = probs - one_hot
        grad_emb = diff.unsqueeze(2) * embeddings.unsqueeze(1)
        return torch.norm(grad_emb.view(len(y), -1), dim=1)

    def _calculate_base_egl_components(self, x, y, nLab):
        """
        Core EGL calculation logic to get gradient norms, embeddings, logits, and probabilities.
        :param x: Input data tensor
        :param y: True labels
        :param nLab: Number of classes
        :return: Tuple of (gradient norms, embeddings, logits, probabilities)
        """
        logits, embeddings, probs = self._forward_and_get_outputs(x)
        norms = self._compute_gradient_norms(probs, logits, embeddings, y, nLab)
        return norms, embeddings, logits, probs