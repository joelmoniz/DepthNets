from __future__ import print_function
import time, os, pickle, sys, math
import numpy as np
import torch
from torch.autograd import grad
from depthnet_gan import DepthNetGAN

"""
SDAM = source depth / average m. Basically, two things:
  1) The network only takes src depth, not src + target
  2) We create m based on random tgt keypoint samples.
"""
class DepthNetGAN_SDAM(DepthNetGAN):

    def run_on_instance(self,
                        xy_keypts_src,
                        z_keypts_src,
                        xy_keypts_tgt,
                        z_keypts_tgt,
                        train,
                        **kwargs):
        """
        Train the network on a single example.

        Parameters
        ----------
        xy_keypts_src: (bs, 66, 2)
        z_keypts_src: (bs, 66)
        xy_keypts_tgt: (bs, 66, 2)
        z_keypts_tgt: (bs, 66)
        """
        if train:
            self.optim['g'].zero_grad()
            self.optim['d'].zero_grad()
        bs = xy_keypts_src.shape[0]
        xy_keypts_src = xy_keypts_src.swapaxes(1,2) # (bs, 2, 66)
        xy_keypts_tgt = xy_keypts_tgt.swapaxes(1,2) # (bs, 2, 66)
        xy_keypts_src_torch, z_keypts_src_torch, xy_keypts_tgt_torch = \
            torch.from_numpy(xy_keypts_src).float(), \
            torch.from_numpy(z_keypts_src).float(), \
            torch.from_numpy(xy_keypts_tgt).float()
        if self.use_cuda:
            xy_keypts_src_torch = xy_keypts_src_torch.cuda()
            z_keypts_src_torch = z_keypts_src_torch.cuda()
            xy_keypts_tgt_torch = xy_keypts_tgt_torch.cuda()
        # Unlike the base model, we only pass in the source
        # keypts.
        src_z_pred = self.g(xy_keypts_src_torch.contiguous()).unsqueeze(1) # (bs,1,66)
        A = self.construct_A(xy_keypts_src,
                             src_z_pred)
        xt = xy_keypts_tgt.reshape((bs, 2*66), order='F') # (bs, 66*2)
        xt = torch.from_numpy(xt).float()
        if self.use_cuda:
            xt = xt.cuda()
        # Once we have computed A, we want to compute P times
        # the expression (A'A)^-1(A'xt_i), where i=1...P.
        # Once we have this expression, we compute the mean
        # m matrix.
        # (A'A)^-1
        X1 = [self.inv(mat) for mat in
              torch.matmul(A.transpose(2, 1), A)]
        X1 = torch.stack(X1)
        m_total = None
        P = 10
        for i in range(P):
            xt_perm = xt[ torch.randperm(xt.size(0)) ]
            # (A'xt)
            X2 = torch.bmm(A.transpose(2, 1), xt_perm.unsqueeze(2))
            # (A'A)^-1 . (A'xt)
            m = torch.bmm(X1, X2) # (bs,8,1)
            m_rshp = m.squeeze(2).view((bs, 2, 4))
            if m_total is None:
                m_total = m_rshp
            else:
                m_total += m_rshp
        m_total /= P
        # Now we have to implement equation (4).
        # Let's compute the right-hand term which
        # multiplies m.
        ones = torch.ones((bs, 1, 66)).float()
        if self.use_cuda:
            ones = ones.cuda()
        rht = torch.cat( (xy_keypts_src_torch, src_z_pred, ones), dim=1)
        rhs = torch.matmul(m_total, rht)
        if not self.use_l1:
            l2_loss = torch.mean((xy_keypts_tgt_torch - rhs)**2)
        else:
            l2_loss = torch.mean(torch.abs(xy_keypts_tgt_torch - rhs))
        if self.cheat:
            # Measure the L1 between the predicted z and the real z.
            # This is cheating!!!
            cheat_l1_loss = torch.mean(
                torch.abs(src_z_pred - z_keypts_src_torch.unsqueeze(1)))
        # Now do the adversarial losses.
        src_z_pred_given_inp = torch.cat(
            (src_z_pred, xy_keypts_src_torch), dim=1)
        if train:
            (self.lamb*l2_loss).backward(retain_graph=True)
            if self.cheat:
                cheat_l1_loss.backward(retain_graph=True)
            if not self.no_gan:
                g_loss = -torch.mean(self.d(src_z_pred_given_inp))
                if (kwargs['iter']-1) % self.update_g_every == 0:
                    # Also update generator.
                    g_loss.backward()
            self.optim['g'].step()
        # Now do the discriminator
        if train and not self.no_gan:
            self.optim['d'].zero_grad()
            src_z_gt_given_inp = torch.cat(
                (z_keypts_src_torch.unsqueeze(1),
                 xy_keypts_src_torch), dim=1)
            d_real = self.d(src_z_gt_given_inp)
            d_fake = self.d(src_z_pred_given_inp.detach())
            d_loss_real = torch.mean(d_real)
            d_loss_fake = torch.mean(d_fake)
            d_loss = -d_loss_real + d_loss_fake
            d_loss.backward()
            self.optim['d'].step()
        # Grad norm
        if train and not self.no_gan and self.dnorm > 0.:
            d_real_inp = src_z_gt_given_inp.detach()
            d_real_inp.requires_grad = True
            d_real_ = self.d(d_real_inp)
            g_norm_x = self.grad_norm(
                d_real_, d_real_inp)
            self.optim['d'].zero_grad()
            (g_norm_x*self.dnorm).backward()
            self.optim['d'].step()
        losses = {'l2': l2_loss.data.item()}
        if not self.no_gan:
            losses['g_loss'] = g_loss.data.item(),
            losses['d_loss'] = d_loss.data.item(),
            losses['d_loss_real'] = d_loss_real.data.item(),
            losses['d_loss_fake'] = d_loss_fake.data.item(),
            losses['dnorm_x'] = g_norm_x.data.item()
        if self.cheat:
            losses['cheat_l1_loss'] = cheat_l1_loss.data.item()
        outputs = {
            'src_z_pred': src_z_pred,
            'tgt_2d_pred': rhs
        }
        return losses, outputs
