import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Shortcut connection
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.linear(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = out + self.shortcut(x)  # Residual addition
        return out

class PlayerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout=0.0, activation='relu'):
        super().__init__()
        layers = []
        last_dim = input_dim

        # Select activation
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'swish': nn.SiLU(),
        }
        act_fn = activations.get(activation.lower())
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {activation}")

        for dim in hidden_dims:
            if dropout > 0:
                layers.append(ResidualBlock(last_dim, dim, activation=act_fn, dropout=dropout))
            last_dim = dim
        layers.append(nn.Linear(last_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
