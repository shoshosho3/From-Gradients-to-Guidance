import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

class BaseNetHandler(ABC):
    """
    Abstract Base Class for network handlers.
    Encapsulates common logic like model creation, prediction, and evaluation.
    """
    def __init__(self, args_task, device, dataset_name):
        self.params = args_task
        self.device = device
        self.dataset_name = dataset_name
        self.model = None

    def _get_model_instance(self, n_channels):
        """Subclasses must implement this to return their specific nn.Module."""
        raise NotImplementedError

    def _check_and_create_model(self, data):
        """Handles the boilerplate of model creation."""
        if self.model is None:
            first_x, _, _ = data[0]
            n_channels = first_x.shape[0]
            self.model = self._get_model_instance(n_channels).to(self.device)

    @abstractmethod
    def train(self, data):
        """Subclasses must implement their own training loop."""
        pass

    def predict(self, data):
        """Generic prediction logic for classification."""
        self._check_and_create_model(data)
        self.model.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                # Assumes the first return value is logits
                logits, _ = self.model(x)
                pred = logits.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def get_model(self):
        return self.model