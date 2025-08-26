import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm

from .strategy import Strategy

# A versatile ResNet18 backend that can be adapted for different datasets.
# It serves as the core model for the Minds strategy.
class Minds_Backend(nn.Module):
    def __init__(self, num_classes, pretrained=True, n_channels=3):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)
        
        # --- FIX START ---
        # We use a more robust "adapter" pattern instead of modifying the ResNet's first layer.
        # This 1x1 convolution will map the input channels (e.g., 1 for MNIST) to the
        # 3 channels that the pretrained ResNet expects.
        if n_channels != 3:
            self.channel_adapter = nn.Conv2d(n_channels, 3, kernel_size=1)
        else:
            self.channel_adapter = nn.Identity() # If already 3 channels, do nothing.
        
        # We now use the standard ResNet backbone without modification.
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        # --- FIX END ---
        
        self.feature_dim = backbone.fc.in_features
        self.classifier = nn.Linear(self.feature_dim, num_classes)
        
    def forward(self, x):
        # --- FIX START ---
        # First, adapt the input channels to what ResNet expects.
        x = self.channel_adapter(x)
        # --- FIX END ---
        
        f = self.features(x)
        embeddings = f.view(f.size(0), -1)
        logits = self.classifier(embeddings)
        return logits, embeddings

    def get_embedding_dim(self):
        return self.feature_dim

# This class integrates the Minds model and its training logic into the framework,
# mimicking the structure of the `Net` class in `nets.py`.
class Net_Minds:
    def __init__(self, args_task, device):
        self.params = args_task
        self.device = device
        self.model = None

    def _create_model(self, data):
        """Lazy model instantiation based on the input data's shape."""
        if self.model is None:
            n_channels = data.X.shape[1]
            self.model = Minds_Backend(
                num_classes=self.params['num_class'],
                pretrained=self.params['pretrained'],
                n_channels=n_channels
            ).to(self.device)

    def train(self, data):
        self._create_model(data)  # Ensure model is created
        n_epoch = self.params['n_epoch']
        
        self.model.train()
        if self.params['optimizer'] == 'Adam':
            optimizer = optim.Adam(self.model.parameters(), **self.params['optimizer_args'])
        elif self.params['optimizer'] == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), **self.params['optimizer_args'])
        else:
            raise NotImplementedError

        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100, desc="Training Minds"):
            for _, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, _ = self.model(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
    
    def predict(self, data):
        """Predicts class labels."""
        self._create_model(data)
        self.model.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                out, _ = self.model(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def predict_prob(self, data):
        """Predicts class probabilities."""
        self._create_model(data)
        self.model.eval()
        probs = torch.zeros([len(data), self.params['num_class']])
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                out, _ = self.model(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs
    
    def get_grad_embeddings(self, data):
        """
        Calculates the gradient of the loss with respect to the last layer's weights
        for each unlabeled sample, using the model's prediction as a pseudo-label.
        This is a direct implementation of the logic from nets.py:Net.get_grad_embeddings.
        """
        self._create_model(data)
        self.model.eval()
        
        embDim = self.model.get_embedding_dim()
        nLab = self.params['num_class']
        embeddings = np.zeros([len(data), embDim * nLab])

        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                
                cout, out = self.model(x) # cout: logits, out: embeddings
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs, 1)

                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            # Gradient component for the predicted class: (p_c - 1) * embedding
                            embeddings[idxs[j]][embDim * c : embDim * (c + 1)] = out[j] * (batchProbs[j][c] - 1)
                        else:
                            # Gradient component for other classes: p_c * embedding
                            embeddings[idxs[j]][embDim * c : embDim * (c + 1)] = out[j] * batchProbs[j][c]
        return embeddings

    def get_model(self):
        return self.model

# This is the main Strategy class used by demo.py
class Minds(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(Minds, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        """
        Query new points by selecting those with the largest expected gradient length (EGL).
        The EGL is computed on the parameters of the last layer of the network.
        """
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Get the gradient embeddings for each unlabeled point.
        # This represents the gradient of the loss w.r.t the final layer's weights,
        # calculated using the model's prediction as a pseudo-label.
        grad_embeddings = self.net.get_grad_embeddings(unlabeled_data)

        # The uncertainty score is the L2 norm of this gradient vector.
        # A larger norm indicates a sample that would cause a larger model update,
        # implying it is more informative.
        scores = np.linalg.norm(grad_embeddings, axis=1)

        # Select the top 'n' samples with the highest scores.
        # argsort() gives indices for an ascending sort, so we take the last 'n' for the largest values.
        top_n_indices = scores.argsort()[-n:]

        return unlabeled_idxs[top_n_indices]