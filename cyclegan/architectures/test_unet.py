from networks import define_G
import torch
import numpy as np
from torch.autograd import Variable

if __name__ == '__main__':
    net = define_G(input_nc=3, output_nc=3, ngf=64, which_model_netG='unet_80', norm='batch')

    xfake = np.random.normal(0,1,size=(2,3,80,80))
    xfake = Variable( torch.from_numpy(xfake).float() )
    dat = net(xfake)
    import pdb
    pdb.set_trace()

