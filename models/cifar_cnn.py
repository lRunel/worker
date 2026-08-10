import torch.nn as nn
import torch.nn.functional as F
from . import register


@register("cifar_cnn", description="Standard CNN for CIFAR-10 (32x32 RGB)")
class CIFARCNN(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, **kwargs):
        super().__init__()
        # Input: (3, 32, 32)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # After two max pools: 32 -> 16 -> 8
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        
        # Block 2
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
