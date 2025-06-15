import torch.nn as nn
import torchvision.models as models

class MultiTaskFaceNet(nn.Module):
    def __init__(self, num_identities, backbone='resnet50', weights=None):
        super().__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if weights else None)
            feature_dim = 2048
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if weights else None)
            feature_dim = 512
        else:
            raise ValueError(f'Unsupported backbone: {backbone}')
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Gender classification head
        self.gender_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # Binary classification
        )
        
        # Identity classification head
        self.identity_head = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_identities)  # Multi-class classification
        )
    
    def forward(self, x):
        # Extract features through backbone
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten
        
        # Get predictions from both heads
        gender_out = self.gender_head(features)
        identity_out = self.identity_head(features)
        
        return {
            'gender': gender_out.squeeze(),
            'identity': identity_out
        }

def create_model(config, num_identities):
    model = MultiTaskFaceNet(
        num_identities=num_identities,
        backbone='resnet50',
        weights=True # Pass True to use default weights
    )
    model = model.to(config['device'])
    return model