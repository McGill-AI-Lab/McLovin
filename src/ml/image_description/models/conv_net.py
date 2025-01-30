import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNet(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        # More conv layers with batch normalization
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, stride=2), #64, 64, 64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), #64, 32, 32
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 256, 5, padding=1), #256, 28, 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), #256, 14, 14
            nn.BatchNorm2d(128),

            nn.Conv2d(256, 384, 3, padding=1), #384, 12, 12 
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 3, padding=1), #384, 10, 10
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), #256, 8, 8
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2) #256, 4, 4 -> this will become the input to the first linear
        )

        # Adaptive pooling to handle different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        # Dropout for regularization
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
