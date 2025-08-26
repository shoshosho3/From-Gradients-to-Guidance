import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
import random

from .strategy import Strategy

# A standard ResNet18 backend for classification.
# It returns both logits and embeddings, which are needed for the EGL calculation.
class EGL_Backend(nn.Module):
    def __init__(self, num_classes, pretrained=True, n_channels=3):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        if n_channels != 3:
            self.channel_adapter = nn.Conv2d(n_channels, 3, kernel_size=1)
        else:
            self.channel_adapter = nn.Identity()
        
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.feature_dim = backbone.fc.in_features
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x):
        x = self.channel_adapter(x)
        embeddings = self.features(x).view(x.size(0), -1)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def get_embedding_dim(self):
        return self.feature_dim

# Network handler for the EGL strategy.
# Contains a standard training loop and the crucial `get_grad_embeddings` method.
class Net_EGL:
    def __init__(self, args_task, device):
        self.params = args_task
        self.device = device
        self.model = None

    def _create_model(self, data):
        if self.model is None:
            first_x, _, _ = data[0] 
            n_channels = first_x.shape[0]
            self.model = EGL_Backend(
                num_classes=self.params['num_class'],
                pretrained=self.params['pretrained'],
                n_channels=n_channels
            ).to(self.device)

    def train(self, data):
        """Standard classification training loop."""
        self._create_model(data)
        n_epoch = self.params['n_epoch']
        
        self.model.train()
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), **self.params['optimizer_args'])
        else:
            raise NotImplementedError

        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100, desc="Training EGL"):
            for _, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits, _ = self.model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
    
    def get_grad_embeddings(self, data):
        """
        Calculates the gradient of the loss with respect to the final layer's weights
        for each unlabeled sample, using the model's prediction as a pseudo-label.
        """
        self._create_model(data)
        self.model.eval()
        
        embDim = self.model.get_embedding_dim()
        nLab = self.params['num_class']
        grad_embeddings = np.zeros([len(data), embDim * nLab])

        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        
        with torch.no_grad():
            for x, y, idxs in loader:
                x = x.to(self.device)
                
                logits, embeddings = self.model(x)
                embeddings = embeddings.data.cpu().numpy()
                probs = F.softmax(logits, dim=1).data.cpu().numpy()
                pred_labels = np.argmax(probs, 1)

                for j in range(len(y)):
                    for c in range(nLab):
                        if c == pred_labels[j]:
                            grad_embeddings[idxs[j]][embDim*c:embDim*(c+1)] = embeddings[j] * (probs[j][c] - 1)
                        else:
                            grad_embeddings[idxs[j]][embDim*c:embDim*(c+1)] = embeddings[j] * probs[j][c]
        return grad_embeddings

    def predict(self, data):
        self._create_model(data)
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
        return self.model

# The main EGL Strategy class.
class EGL(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(EGL, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Calculate gradient embeddings at query time.
        grad_embeddings = self.net.get_grad_embeddings(unlabeled_data)

        # The score is the L2 norm of the gradient vector.
        scores = np.linalg.norm(grad_embeddings, axis=1)

        # Select the top n samples with the largest gradient lengths.
        top_n_indices = scores.argsort()[-n:]

        return unlabeled_idxs[top_n_indices]


# R-EGL Strategy. Introduces randomness
class REGL(Strategy):
    def __init__(self, dataset, net, args_input, args_task, factor=5):
        super(REGL, self).__init__(dataset, net, args_input, args_task)
        self.factor = factor

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Calculate gradient embeddings at query time.
        grad_embeddings = self.net.get_grad_embeddings(unlabeled_data)

        # The score is the L2 norm of the gradient vector.
        scores = np.linalg.norm(grad_embeddings, axis=1)

        # Sample n from the top factor*n samples with the largest gradient lengths.
        candidates = scores.argsort()[-1*self.factor*n:]
        selected = random.sample(candidates, n)

        return unlabeled_idxs[selected]