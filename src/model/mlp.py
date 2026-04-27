import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Flexible multi-layer perceptron.

    Parameters
    ----------
    in_dim      : number of input features
    layer_sizes : list of hidden-layer widths, e.g. [128, 64, 32]
    dropout_rate: dropout probability applied after each hidden activation
    activation  : one of 'relu', 'leaky_relu', 'elu', 'gelu'
    use_bn      : whether to insert BatchNorm1d before each activation
    """

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }

    def __init__(
        """
        Constructor method
        """
        self,
        in_dim: int,
        layer_sizes: list,
        dropout_rate: float,
        activation: str,
        use_bn: bool,
    ):
        super().__init__()
        Act = self.ACTIVATIONS[activation]
        layers = []
        prev = in_dim

        for size in layer_sizes:
            layers.append(nn.Linear(prev, size))
            if use_bn:
                layers.append(nn.BatchNorm1d(size))
            layers.append(Act())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev = size

        layers.append(nn.Linear(prev, 1))  # binary output (BCEWithLogits)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)
