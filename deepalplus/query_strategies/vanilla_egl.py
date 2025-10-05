import numpy as np
import torch
from torch.utils.data import DataLoader

from .strategy import Strategy
from .base_egl_handler import BaseEGLNetHandler


class Net_VanillaEGL(BaseEGLNetHandler):
    """
    Network handler for the original Vanilla EGL strategy.

    Inherits common functionality from BaseEGLNetHandler and implements the
    standard EGL score calculation.
    """

    def calculate_egl_score(self, x, y, nLab):
        """
        Calculates the standard EGL score for a batch of samples.
        It uses the common component calculation from the base class.
        :param x: Input batch of samples
        :param y: Labels for the input batch
        :param nLab: Number of classes in the classification task
        :return: Tensor of EGL scores for each sample in the batch
        """
        norms, _, _, _ = self._calculate_base_egl_components(x, y, nLab)
        return norms

    def get_scores(self, data):
        """
        Calculates the original Expected Gradient Length (EGL) scores.
        :param data: Unlabeled dataset for scoring
        :return: Numpy array of EGL scores for each sample in the dataset
        """
        self._check_and_create_model(data)
        self.model.eval()

        nLab = self.params['num_class']
        scores = torch.zeros(len(data))
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])

        with torch.no_grad():
            for x, y, idxs in loader:
                norms = self.calculate_egl_score(x, y, nLab)
                scores[idxs] = norms.cpu()

        return scores.numpy()


class VanillaEGL(Strategy):
    """
    Standard top-N query strategy using the original EGL scores.

    It selects the samples with the largest expected gradient length, which are
    considered the most informative for the model.
    """

    def query(self, n):
        """
        Queries the top-N samples with the highest EGL scores.
        :param n: Number of samples to query
        :return: Indices of the selected samples in the original dataset
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()
        scores = self.net.get_scores(unlabeled_data)
        top_n_indices = scores.argsort()[-n:]
        return unlabeled_idxs[top_n_indices]