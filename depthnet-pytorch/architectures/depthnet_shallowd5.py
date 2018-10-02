
import torch
from torch import nn
from torch.nn import init
from architectures.init import (weights_init_normal,
                                count_params)

class Generator(nn.Module):
    def __init__(self, nc=2, ni=512):
        """
        nc: how many sets of kpts we're conditioning on, s.t.
          the # of input units is nc*(2*num_keypts)
        ni = base number of units in hidden layer
        """
        super(Generator, self).__init__()
        self.num_units = 66
        self.nc = nc
        self.ni = 512
        self.fc = nn.Sequential(
            nn.Linear(self.nc*(2*self.num_units), ni),
            nn.BatchNorm1d(ni),
            nn.ReLU(),
            
            nn.Linear(ni, ni),
            nn.BatchNorm1d(ni),
            nn.ReLU(),

            nn.Linear(ni, ni),
            nn.BatchNorm1d(ni),
            nn.ReLU(),
            
            nn.Linear(ni, ni//2),
            nn.BatchNorm1d(ni//2),
            nn.ReLU(),

            nn.Linear(ni//2, ni//2),
            nn.BatchNorm1d(ni//2),
            nn.ReLU(),

            nn.Linear(ni//2, ni//2),
            nn.BatchNorm1d(ni//2),
            nn.ReLU(),
            
            nn.Linear(ni//2, ni//4),
            nn.BatchNorm1d(ni//4),
            nn.ReLU(),
            
            nn.Linear(ni//4, self.num_units)
        )

        self.fc.apply(weights_init_normal)
        
    def forward(self, x):
        # x is (bs,2,60)
        return self.fc(x.view(-1, (2*self.nc)*self.num_units))

'''
"""
So this discriminator was used in experiments where
it's not actually used, like the no_gan experiments.
So I am going to replace this with the discriminator
from depthnet_mlp.py instead.

"""
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Linear(3*66, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
                        
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        return self.disc(x.view(-1, 3*66))
'''

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

    
def get_network():
    g = Generator()
    d = Discriminator()
    
    print(g)
    print("# params:", count_params(g))
    print(d)
    print("# params:", count_params(d))
    
    return g, d

if __name__ == '__main__':
    get_network()
