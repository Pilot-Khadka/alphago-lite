import torch
from torch import nn


class SmallNetwork(nn.Module):
    def __init__(self, input_shape, num_classes=19 * 19):
        super().__init__()
        channels, height, width = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(channels, 48, kernel_size=7, stride=1, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
        )

        self.feature_size = self._calculate_output_size(input_shape)
        self.classifier = nn.Linear(self.feature_size, num_classes)

    def _calculate_output_size(self, input_shape):
        with torch.no_grad():
            x = torch.zeros(1, *input_shape)
            x = self.features(x)
            return x.numel()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
