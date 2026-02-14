import torch
import pytest
from alphago.networks import SmallResidualNetwork


@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("board_size", [9, 13, 19])
def test_small_residual_network_forward(batch_size, board_size):
    input_channels = 17  # typical Go features
    input_shape = (input_channels, board_size, board_size)

    x = torch.randn(batch_size, *input_shape)
    model = SmallResidualNetwork(channel_size=input_channels, board_size=board_size)

    output = model(x)

    expected_shape = (batch_size, board_size * board_size)
    assert output.shape == expected_shape, (
        f"Expected {expected_shape}, got {output.shape}"
    )

    assert torch.isfinite(output).all(), "Output contains non-finite values"


def test_residual_blocks_gradient():
    batch_size = 2
    board_size = 19
    input_channels = 17
    input_shape = (input_channels, board_size, board_size)

    x = torch.randn(batch_size, *input_shape, requires_grad=True)
    model = SmallResidualNetwork(channel_size=input_channels, board_size=board_size)

    output = model(x)
    loss = output.sum()
    loss.backward()

    grad_found = any(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert grad_found, "No gradients found in the model parameters"
