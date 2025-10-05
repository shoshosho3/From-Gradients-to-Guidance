import torch
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

class BaseNetHandler(ABC):
    """
    Abstract Base Class for network handlers.
    Encapsulates common logic like model creation, prediction, and evaluation.
    """
    def __init__(self, args_task, device, dataset_name):
        """
        Initializes the BaseNetHandler.
        :param args_task: Task-specific arguments.
        :param device: Computation device
        :param dataset_name: Name of the dataset being used.
        """
        self.params = args_task
        self.device = device
        self.dataset_name = dataset_name
        self.model = None

    def _get_model_instance(self, n_channels):
        """Subclasses must implement this to return their specific nn.Module."""
        raise NotImplementedError

    def _check_and_create_model(self, data):
        """
        Checks if the model is instantiated; if not, creates it based on input data dimensions.
        :param data: The dataset to infer input dimensions from.
        """
        if self.model is None:
            first_x, _, _ = data[0]
            n_channels = first_x.shape[0]
            self.model = self._get_model_instance(n_channels).to(self.device)

    @abstractmethod
    def train(self, data):
        """Subclasses must implement their own training loop."""
        pass

    def predict(self, data):
        """
        Generic prediction logic for classification.
        :param data: The dataset to predict on.
        :return: Tensor of predicted class labels.
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

    def get_model(self):
        """
        Returns the instantiated model.
        :return: The model instance.
        """
        return self.model