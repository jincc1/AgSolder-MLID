import torch
import torch.nn as nn
from . import weights_init

class WAE(nn.Module):
    """
    Defines the Wasserstein Autoencoder (WAE) model.
    """
    def __init__(self, input_size):
        super(WAE, self).__init__()
        self.input_size = input_size

        # Encoder
        self.encoder = nn.Sequential(
                        nn.Linear(self.input_size, 80),
                        nn.LayerNorm(80),
                        nn.ReLU(),
                        nn.Linear(80, 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Linear(64, 48),
                        nn.LayerNorm(48),
                        nn.ReLU(),
                        nn.Linear(48, 2),
                        )

        # Decoder
        self.decoder = nn.Sequential(
                        nn.Linear(2, 48),
                        nn.LayerNorm(48),
                        nn.ReLU(),
                        nn.Linear(48, 64),
                        nn.LayerNorm(64),
                        nn.ReLU(),
                        nn.Linear(64, 80),
                        nn.LayerNorm(80),
                        nn.ReLU(),
                        nn.Linear(80, self.input_size),
                        nn.Softmax(dim=1) # Softmax along dimension 1
                        )
        self.apply(weights_init)

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)