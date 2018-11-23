"""
This contents of this module are derived from this CycleGAN
implementation:
https://raw.githubusercontent.com/togheppi/CycleGAN/master/model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, activation='relu', batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.batch_norm = batch_norm
        self.bn = torch.nn.InstanceNorm2d(output_size,
                                          track_running_stats=True)
        self.activation = activation
        self.relu = torch.nn.ReLU(True)
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
        self.tanh = torch.nn.Tanh()

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'no_act':
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=2, padding=1, output_padding=1, activation='relu', batch_norm=True):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, output_padding)
        self.batch_norm = batch_norm
        self.bn = torch.nn.InstanceNorm2d(output_size,
                                          track_running_stats=True)
        self.activation = activation
        self.relu = torch.nn.ReLU(True)

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation == 'relu':
            return self.relu(out)
        elif self.activation == 'lrelu':
            return self.lrelu(out)
        elif self.activation == 'tanh':
            return self.tanh(out)
        elif self.activation == 'no_act':
            return out


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter_in, num_filter, kernel_size=3, stride=1, padding=0):
        super(ResnetBlock, self).__init__()
        conv1 = torch.nn.Conv2d(num_filter_in, num_filter, kernel_size, stride, padding)
        conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding)
        bn = torch.nn.InstanceNorm2d(num_filter,
                                     track_running_stats=True)
        relu = torch.nn.ReLU(True)
        pad = torch.nn.ReflectionPad2d(1)

        self.resnet_block = torch.nn.Sequential(
            pad,
            conv1,
            bn,
            relu,
            pad,
            conv2,
            bn
        )

    def forward(self, x):
        out = self.resnet_block(x)
        return out

class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim, num_resnet, add_noise=False):
        super(Generator, self).__init__()

        # Reflection padding
        self.pad = torch.nn.ReflectionPad2d(3)
        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, kernel_size=7, stride=1, padding=0)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        # Resnet blocks
        self.resnet_blocks = []
        for i in range(num_resnet):
            self.resnet_blocks.append(ResnetBlock(num_filter*4, num_filter*4))
        self.resnet_blocks = torch.nn.Sequential(*self.resnet_blocks)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 4, num_filter * 2)
        self.deconv2 = DeconvBlock(num_filter * 2, num_filter)
        self.deconv3 = ConvBlock(num_filter, output_dim,
                                 kernel_size=7, stride=1, padding=0, activation='tanh', batch_norm=False)
        #self.add_noise = add_noise

    #def gaussian_noise(self, ins, mean=0, stddev=1):
    #    noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
    #    return ins + noise
        
    def forward(self, x):
        # Encoder
        enc1 = self.conv1(self.pad(x))
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        #if self.add_noise:
            #import pdb
            #pdb.set_trace()
        #    enc3 = self.gaussian_noise(enc3)
        # Resnet blocks
        res = self.resnet_blocks(enc3)
        # Decoder
        dec1 = self.deconv1(res)
        dec2 = self.deconv2(dec1)
        out = self.deconv3(self.pad(dec2))
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal(m.deconv.weight, mean, std)
            if isinstance(m, ResnetBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)
                torch.nn.init.constant(m.conv.bias, 0)


class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()

        conv1 = ConvBlock(input_dim, num_filter, kernel_size=4, stride=2, padding=1, activation='lrelu', batch_norm=False)
        conv2 = ConvBlock(num_filter, num_filter * 2, kernel_size=4, stride=2, padding=1, activation='lrelu')
        conv3 = ConvBlock(num_filter * 2, num_filter * 4, kernel_size=4, stride=2, padding=1, activation='lrelu')
        conv4 = ConvBlock(num_filter * 4, num_filter * 8, kernel_size=4, stride=1, padding=1, activation='lrelu')
        conv5 = ConvBlock(num_filter * 8, output_dim, kernel_size=4, stride=1, padding=1, activation='no_act', batch_norm=False)

        self.conv_blocks = torch.nn.Sequential(
            conv1,
            conv2,
            conv3,
            conv4,
            conv5
        )

    def forward(self, x):
        out = self.conv_blocks(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal(m.conv.weight, mean, std)



class GeneratorWithLongSkips(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim, num_resnet):
        super(GeneratorWithLongSkips, self).__init__()
        # Reflection padding
        self.pad = torch.nn.ReflectionPad2d(3)
        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, kernel_size=7, stride=1, padding=0) # (32, 80, 80)
        self.conv2 = ConvBlock(num_filter, num_filter * 2) # (64, 40, 40)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4) # (128, 20, 20)
        # Resnet blocks
        self.resnet_blocks = []
        for i in range(num_resnet):
            self.resnet_blocks.append(ResnetBlock(num_filter*4, num_filter*4)) # (128, 20, 20)
        self.resnet_blocks = torch.nn.Sequential(*self.resnet_blocks)
        # Decoder
        self.deconv1 = DeconvBlock( (num_filter * 4), num_filter * 2) # (64,40,40)
        self.deconv2 = DeconvBlock( (num_filter * 2) + (num_filter*2), num_filter) # (32, 80,80)
        self.deconv3 = ConvBlock(num_filter+num_filter, output_dim,
                                 kernel_size=7, stride=1, padding=0, activation='tanh', batch_norm=False) # (3,80,80)
        
    def forward(self, x):
        # Encoder
        enc1 = self.conv1(self.pad(x))
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        # Resnet blocks
        res = self.resnet_blocks(enc3)
        # Decoder
        dec1 = torch.cat((self.deconv1(res), enc2), dim=1)
        dec2 = torch.cat((self.deconv2(dec1), enc1), dim=1)
        out = self.deconv3(self.pad(dec2))
        return out




class GeneratorWithLongSkipsExtraConv(torch.nn.Module):
    """
    Add long skip connection between encoder/decoder and add option to
      perform extra convolution after each decode.
    """
    def __init__(self, input_dim, num_filter, output_dim, num_resnet):
        super(GeneratorWithLongSkipsExtraConv, self).__init__()
        # Reflection padding
        self.pad = torch.nn.ReflectionPad2d(3)
        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter, kernel_size=7, stride=1, padding=0) # (32, 80, 80)
        self.conv2 = ConvBlock(num_filter, num_filter * 2) # (64, 40, 40)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4) # (128, 20, 20)
        # Resnet blocks
        self.resnet_blocks = []
        for i in range(num_resnet):
            self.resnet_blocks.append(ResnetBlock(num_filter*4, num_filter*4)) # (128, 20, 20)
        self.resnet_blocks = torch.nn.Sequential(*self.resnet_blocks)
        # Decoder
        self.deconv1 = DeconvBlock( (num_filter * 4), num_filter * 2) # (64,40,40)
        self.conv_ad1 = ConvBlock( (num_filter*2)+(num_filter*2), num_filter*2, stride=1, padding=1)
        self.deconv2 = DeconvBlock( num_filter*2, num_filter) # (32, 80,80)
        self.conv_ad2 = ConvBlock( num_filter+num_filter, num_filter, stride=1, padding=1) 
        self.deconv3 = ConvBlock(num_filter, output_dim,
                                 kernel_size=7, stride=1, padding=0, activation='tanh', batch_norm=False) # (3,80,80)
        
    def forward(self, x):
        # Encoder
        enc1 = self.conv1(self.pad(x))
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        # Resnet blocks
        res = self.resnet_blocks(enc3)
        # Decoder
        dec1 = torch.cat((self.deconv1(res), enc2), dim=1)
        dec1 = self.conv_ad1(dec1)        
        dec2 = torch.cat((self.deconv2(dec1), enc1), dim=1)
        dec2 = self.conv_ad2(dec2)
        out = self.deconv3(self.pad(dec2))
        return out
