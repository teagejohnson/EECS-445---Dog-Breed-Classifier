import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt

class Challenge(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv_2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=2, padding=2)
        self.conv_3 = nn.Conv2d(in_channels=64, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool_2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(p=0.5)
        self.fc_1 = nn.Linear(in_features=32, out_features=2)

        self.init_weights()

    def init_weights(self):
        torch.manual_seed(42)

        for conv in [self.conv_1, self.conv_2, self.conv_3]:            
            nn.init.normal_(conv.weight, mean=0, std=sqrt(1 / (5 * 5 * conv.in_channels)))
            nn.init.zeros_(conv.bias)
        
        nn.init.normal_(self.fc_1.weight, mean=0, std=sqrt(1 / self.fc_1.in_features))
        nn.init.zeros_(self.fc_1.bias)
        

    def forward(self, x):
        x = F.relu(self.conv_1(x))
        x = self.dropout(x)
        x = self.pool_2(x)
        x = F.relu(self.conv_2(x))
        x = self.dropout(x)
        x = self.pool_2(x)
        x = F.relu(self.conv_3(x))
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)

        x = self.fc_1(x)

        return x
