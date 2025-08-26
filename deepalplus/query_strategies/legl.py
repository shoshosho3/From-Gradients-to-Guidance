import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm

from .strategy import Strategy

# The LEGL backend has two heads: one for classification and one for predicting the gradient norm.
class LEGL_Backend(nn.Module):
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
        
        self.legl_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        x = self.channel_adapter(x)
        embeddings = self.features(x).view(x.size(0), -1)
        
        logits = self.classifier(embeddings)
        predicted_grad_norm = self.legl_head(embeddings)
        
        return logits, predicted_grad_norm

# Network handler for LEGL with a custom training loop.
class Net_LEGL:
    def __init__(self, args_task, device):
        self.params = args_task
        self.device = device
        self.model = None
        self.lambda_legl = 0.1 # Hyperparameter to balance the two losses

    def _create_model(self, data):
        if self.model is None:
            first_x, _, _ = data[0] 
            n_channels = first_x.shape[0]
            self.model = LEGL_Backend(
                num_classes=self.params['num_class'],
                pretrained=self.params['pretrained'],
                n_channels=n_channels
            ).to(self.device)

    def train(self, data):
        """Custom training loop to handle both classification and regression losses."""
        self._create_model(data)
        n_epoch = self.params['n_epoch']
        
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), **self.params['optimizer_args'])
        loader = DataLoader(data, shuffle=True, **self.params['loader_tr_args'])
        
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100, desc="Training LEGL"):
            for _, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                
                # --- FIX START: Perform a single forward pass for the entire batch ---
                logits, predicted_norms = self.model(x)
                loss_cls = F.cross_entropy(logits, y) # This is the mean classification loss

                # --- Step 2: Calculate ground-truth gradient norms (the meta-target) ---
                true_norms = []
                final_layer_params = list(self.model.classifier.parameters())
                
                # Loop over the results of the batch, not the model itself
                for i in range(len(x)):
                    # Calculate the loss for this one sample using its pre-computed logit
                    sample_logits = logits[i].unsqueeze(0)
                    sample_loss = F.cross_entropy(sample_logits, y[i].unsqueeze(0))
                    
                    # Compute gradients of this single sample's loss w.r.t the final layer
                    grads = torch.autograd.grad(sample_loss, final_layer_params, retain_graph=True)
                    
                    # Flatten all gradients into a single vector and compute its norm
                    grad_vec = torch.cat([g.view(-1) for g in grads])
                    true_norms.append(grad_vec.norm())
                
                # --- FIX END ---
                
                true_norms_tensor = torch.stack(true_norms).to(self.device)
                
                # Step 3: Calculate the regression loss
                loss_reg = F.mse_loss(predicted_norms.squeeze(), true_norms_tensor.detach())

                # Step 4: Combine losses and backpropagate
                total_loss = loss_cls + self.lambda_legl * loss_reg
                total_loss.backward()
                optimizer.step()

    def predict_legl_scores(self, data):
        """Predicts informativeness scores for querying using the trained LEGL head."""
        self._create_model(data)
        self.model.eval()
        scores = torch.zeros(len(data))
        loader = DataLoader(data, shuffle=False, **self.params['loader_te_args'])
        with torch.no_grad():
            for x, _, idxs in loader:
                x = x.to(self.device)
                _, pred_scores = self.model(x)
                scores[idxs] = pred_scores.squeeze().cpu()
        return scores
        
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

# The main LEGL Strategy class. Querying is simple and fast.
class LEGL(Strategy):
    def __init__(self, dataset, net, args_input, args_task):
        super(LEGL, self).__init__(dataset, net, args_input, args_task)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data()

        # Use the trained legl_head to predict informativeness scores.
        scores = self.net.predict_legl_scores(unlabeled_data)
        
        # Query the top n samples with the highest predicted scores.
        top_n_indices = scores.argsort(descending=True)[:n]
        
        return unlabeled_idxs[top_n_indices]