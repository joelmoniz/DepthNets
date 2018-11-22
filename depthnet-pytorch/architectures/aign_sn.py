import torch
from torch import nn
from .spectral_normalization import SpectralNorm
from .aign import Generator

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            SpectralNorm(nn.Linear(3*66, 512)),
            nn.ReLU(),
            
            SpectralNorm(nn.Linear(512,512)),
            nn.ReLU(),
            
            SpectralNorm(nn.Linear(512, 256)),
            nn.ReLU(),
            
            SpectralNorm(nn.Linear(256, 256)),
            nn.ReLU(),
            
            SpectralNorm(nn.Linear(256, 1))
        )
        
    def forward(self, x):
        return self.disc(x.view(-1, 3*66))

def get_network():
    return Generator(), Discriminator()
