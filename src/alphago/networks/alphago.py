import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential:
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class AlphaGo(nn.Module):
    BOARD_POSITIONS = None

    def __init__(
        self,
        is_policy_net: bool = True,
        board_size: int = 19,
        in_channels: int = 48,
        num_filters: int = 192,
        num_layers: int = 13,
        first_kernel_size: int = 5,
        other_kernel_size: int = 3,
    ):
        super().__init__()
        self.is_policy_net = is_policy_net
        self.board_size = board_size
        self.num_moves = board_size * board_size + 1

        layers = [_conv_block(in_channels, num_filters, first_kernel_size)]
        for _ in range(1, num_layers):
            layers.append(_conv_block(num_filters, num_filters, other_kernel_size))
        self.backbone = nn.Sequential(*layers)

        if is_policy_net:
            self.head = nn.Sequential(
                nn.Conv2d(num_filters, 2, kernel_size=1, bias=False),
                nn.BatchNorm2d(2),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(2 * board_size * board_size, self.num_moves),
            )
        else:
            self.head = nn.Sequential(
                nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(board_size * board_size, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 1),
                nn.Tanh(),
            )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        inputs:
            x: (B, in_channels, board_size, board_size)
        outputs:
            policy net: logits of shape (B, num_moves)
            value net:  win probability of shape (B, 1) in [-1, 1]
        """
        return self.head(self.backbone(x))

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        out = self(x)
        return torch.softmax(out, dim=-1) if self.is_policy_net else out
