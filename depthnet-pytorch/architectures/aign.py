#from architectures.generator import Generator
#from architectures.discriminator import Discriminator

#def get_network():
#    return Generator(), Discriminator()

import torch
from torch import nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(3*66, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.disc(x.view(-1, 3*66))

class Generator(nn.Module):
    def __init__(self, in_channels=66):
        super(Generator, self).__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        num_units = 8*8*256
        self.fc = nn.Sequential(
            nn.Linear(num_units, 512),
            nn.ReLU())
        self.rot_model = nn.Sequential(
            nn.Linear(512, 6),
            nn.Tanh())
        self.alpha_model = nn.Linear(512, 3*66)
        self.T_model = nn.Linear(512, 3)
        self.f_model = nn.Linear(512, 2) ## ??range??
        self.c_model = nn.Linear(512, 2)
        self.num_units = num_units
        
    def forward(self, x):
        features = self.extractor(x)
        features = features.view(-1, self.num_units)
        features = self.fc(features)
        return {
            'rot_model': self.rot_model(features),
            'alpha_model': self.alpha_model(features),
            'T_model': self.T_model(features),
            'f_model': self.f_model(features),
            'c_model': self.c_model(features)
        }

def get_network():
    return Generator(), Discriminator()
