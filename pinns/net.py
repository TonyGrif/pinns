"""This module defines a general PyTorch network utility class"""

import torch
import torch.nn as nn


class Net(nn.Module):
    """Simple fully-connected MLP network"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        output_dim: int = 1,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        """Constructor for the MLP network

        Args:
            input_dim: Number of input features
            hidden_dim: Width of each hidden layer
            layers: Number of hidden layers
            output_dim: Number of output values
            device: Device to place the model on
        """
        super().__init__()

        stack = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]  # Input Layer
        for _ in range(layers - 1):
            stack += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]  # Hidden Layers
        stack.append(nn.Linear(hidden_dim, output_dim))  # Output Layer

        self.net = nn.Sequential(*stack)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP

        Args:
            x: Input tensor

        Returns:
            Tensor containing prediction
        """
        return self.net(x)
