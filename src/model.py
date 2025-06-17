import torch
import torch.nn as nn

class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self, n_class):
        super(DinoVisionTransformerClassifier, self).__init__()
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.transformer = dinov2
        self.n_class = n_class
        self.classifier = nn.Sequential(
            nn.Linear(384, n_class))

    def forward(self, input_ids):
        x = input_ids
        x = self.transformer(x)
        x = self.classifier(x)
        return x