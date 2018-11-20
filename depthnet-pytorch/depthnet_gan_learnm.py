from __future__ import print_function
import time, os, pickle, sys, math
import numpy as np
import torch
from torch.autograd import grad
from depthnet_gan import DepthNetGAN

class DepthNetGAN_M(DepthNetGAN):

    def run_on_instance(self,
                        xy_keypts_src,
                        z_keypts_src,
                        xy_keypts_tgt,
                        z_keypts_tgt,
                        train,
                        **kwargs):
        """Train the network on a single example"""
        is_gan = not self.no_gan
        if train:
            self.optim['g'].zero_grad()
            self.optim['d'].zero_grad()
        bs = xy_keypts_src.shape[0]
        xy_keypts_src_torch, z_keypts_src_torch, xy_keypts_tgt_torch = \
            torch.from_numpy(xy_keypts_src).transpose(1,2).float(), \
            torch.from_numpy(z_keypts_src).float(), \
            torch.from_numpy(xy_keypts_tgt).transpose(1,2).float()
        if self.use_cuda:
            xy_keypts_src_torch = xy_keypts_src_torch.cuda()
            z_keypts_src_torch = z_keypts_src_torch.cuda()
            xy_keypts_tgt_torch = xy_keypts_tgt_torch.cuda()
        net_out = self.g(
            torch.cat((xy_keypts_src_torch,
                       xy_keypts_tgt_torch), dim=1))
        if not (type(net_out) == tuple and len(net_out) == 2):
            raise Exception("Output of g needs to be a tuple of two elements!")
        src_z_pred, m_pred = net_out
        src_z_pred = src_z_pred.unsqueeze(1)
        m_rshp = m_pred.view((bs, 2, 4))
        # Now we have to implement equation (4).
        # Let's compute the right-hand term which
        # multiplies m.
        ones = torch.ones((bs, 1, 66)).float()
        if self.use_cuda:
            ones = ones.cuda()
        if kwargs['use_gt_z']:
            # NOTE: ONLY USE FOR NON-MODEL VALIDATION
            # Use the ground truth src z's instead of the
            # predicted ones.
            rht = torch.cat( (xy_keypts_src_torch,
                              z_keypts_src_torch.unsqueeze(1),
                              ones), dim=1)
        else:
            rht = torch.cat( (xy_keypts_src_torch, src_z_pred, ones), dim=1)
        rhs = torch.matmul(m_rshp, rht)
        if not self.use_l1:
            l2_loss = torch.mean((xy_keypts_tgt_torch - rhs)**2)
        else:
            l2_loss = torch.mean(torch.abs(xy_keypts_tgt_torch - rhs))
        # Now do the adversarial losses.
        src_z_pred_given_inp = torch.cat(
            (src_z_pred, xy_keypts_src_torch), dim=1)
        g_loss = torch.FloatTensor([0.])
        if train:
            (self.lamb*l2_loss).backward(retain_graph=True)
            if is_gan:
                g_loss = -torch.mean(self.d(src_z_pred_given_inp))
                if (kwargs['iter']-1) % self.update_g_every == 0:
                    # Also update generator.
                    g_loss.backward()
            self.optim['g'].step()
        # Now do the discriminator
        d_loss_real = torch.FloatTensor([0.])
        d_loss_fake = torch.FloatTensor([0.])
        d_loss = torch.FloatTensor([0.])
        if is_gan:
            if train:
                self.optim['d'].zero_grad()
            src_z_gt_given_inp = torch.cat(
                (z_keypts_src_torch.unsqueeze(1),
                 xy_keypts_src_torch), dim=1)
            d_real = self.d(src_z_gt_given_inp)
            d_fake = self.d(src_z_pred_given_inp.detach())
            d_loss_real = torch.mean(d_real)
            d_loss_fake = torch.mean(d_fake)
            d_loss = -d_loss_real + d_loss_fake
            if train:
                d_loss.backward()
                self.optim['d'].step()
        # Grad norm
        g_norm_x = torch.FloatTensor([0.])
        if train and self.dnorm > 0.:
            d_real_inp = src_z_gt_given_inp.detach()
            d_real_inp.requires_grad = True
            d_real_ = self.d(d_real_inp)
            g_norm_x = self.grad_norm(
                d_real_, d_real_inp)
            self.optim['d'].zero_grad()
            (g_norm_x*self.dnorm).backward()
            self.optim['d'].step()
        losses = {
            'l2': l2_loss.data.item(),
        }
        if is_gan:
            losses['g_loss'] = g_loss.data.item()
            losses['d_loss'] = d_loss.data.item()
            losses['d_loss_real'] = d_loss_real.data.item()
            losses['d_loss_fake'] = d_loss_fake.data.item()
            if self.dnorm > 0:
                losses['dnorm_x'] = g_norm_x.data.item()
        outputs = {
            'src_z_pred': src_z_pred.detach(),
            'tgt_2d_pred': rhs.detach(),
            'affine': m_rshp.detach()
        }
        return losses, outputs
