import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class SmallGoPolicyNet(nn.Module):
    def __init__(self, input_channels, board_size=19, num_blocks=6):
        super().__init__()

        self.conv_in = nn.Conv2d(input_channels, 64, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(64)

        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(64, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
        )

        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)

        self._initialize_weights()

    def forward(self, x):
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.res_blocks(x)

        p = self.policy_head(x)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        return p

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
