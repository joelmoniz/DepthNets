
import torch
from torch import nn
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.num_units = 66
        self.extractor = nn.Sequential(
            nn.Linear(2*(2*self.num_units), 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.fc_z = nn.Linear(128, self.num_units)
        self.fc_m = nn.Linear(128, 8)
        
        self.extractor.apply(weights_init_normal)
        self.fc_z.apply(weights_init_normal)
        self.fc_m.apply(weights_init_normal)
        
    def forward(self, x):
        # x is (bs,2,60)
        features = self.extractor(x.view(-1, 4*self.num_units))
        pred_z = self.fc_z(features)
        pred_m = self.fc_m(features)
        return pred_z, pred_m

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
    print(d)
    return g, d
