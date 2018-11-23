from __future__ import print_function

import time, os, pickle, sys, math
import numpy as np
from tqdm import tqdm
import itertools
# torch imports
import torch
import torch.optim as optim
from torch.autograd import Variable, grad
# torchvision
from collections import OrderedDict
# local
from .base import BaseModel

def count_params(module, trainable_only=True):
    """
    Count the number of parameters in a module.
    """
    parameters = module.parameters()
    if trainable_only:
        parameters = filter(lambda p: p.requires_grad, parameters)
    num = sum([np.prod(p.size()) for p in parameters])
    return num

class CycleGAN(BaseModel):

    def __str__(self):
        g_summary = str(self.g_atob) + \
                    "\n# parameters for each G:" + str(self.num_parameters(self.g_atob))
        d_summary = str(self.d_a) + \
                    "\n# parameters for each D:" + str(self.num_parameters(self.d_a))
        return g_summary + "\n" + d_summary

    def __init__(self,
                 gen_atob_fn,
                 disc_a_fn,
                 gen_btoa_fn,
                 disc_b_fn,
                 opt_g=optim.Adam,
                 opt_d=optim.Adam,
                 opt_d_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 opt_g_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 loss='mse',
                 lamb=10.,
                 beta=5.,
                 dnorm=None,
                 handlers=[],
                 scheduler_fn=None,
                 scheduler_args={},
                 use_cuda='detect'):
        assert use_cuda in [True, False, 'detect']
        if use_cuda == 'detect':
            use_cuda = True if torch.cuda.is_available() else False
        self.lamb = lamb
        self.beta = beta
        self.dnorm = dnorm
        self.g_atob = gen_atob_fn
        self.g_btoa = gen_btoa_fn
        self.d_a = disc_a_fn
        self.d_b = disc_b_fn
        if loss == 'mse':
            self.loss = self.mse
        elif loss == 'bce':
            self.loss = self.bce
        optim_g = opt_g(
            itertools.chain(
                self.g_atob.parameters(),
                self.g_btoa.parameters()),
            **opt_g_args)
        optim_d_a = opt_d(filter(lambda p: p.requires_grad, self.d_a.parameters()), **opt_d_args)
        optim_d_b = opt_d(filter(lambda p: p.requires_grad, self.d_b.parameters()), **opt_d_args)
        self.optim = {
            'g': optim_g,
            'd_a': optim_d_a,
            'd_b': optim_d_b
        }
        self.scheduler = {}
        if scheduler_fn is not None:
            for key in self.optim:
                self.scheduler[key] = scheduler_fn(
                    self.optim[key], **scheduler_args)
        self.handlers = handlers
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.g_atob.cuda()
            self.g_btoa.cuda()
            self.d_a.cuda()
            self.d_b.cuda()

    def mse(self, prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.cuda()
        return torch.nn.MSELoss()(prediction, target)

    def bce(self, prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.cuda()
        return torch.nn.BCELoss()(prediction, target)

    def compute_g_losses_aba(self, A_real, atob, atob_btoa):
        """Return all the losses related to generation"""
        atob_gen_loss = self.loss(self.d_b(atob), 1)
        cycle_aba = torch.mean(torch.abs(A_real - atob_btoa))
        return atob_gen_loss, cycle_aba

    def compute_g_losses_bab(self, B_real, btoa, btoa_atob):
        """Return all the losses related to generation"""
        btoa_gen_loss = self.loss(self.d_a(btoa), 1)
        cycle_bab = torch.mean(torch.abs(B_real - btoa_atob))
        return btoa_gen_loss, cycle_bab

    def compute_d_losses(self, A_real, atob, B_real, btoa):
        """Return all losses related to discriminator"""
        fake_a = btoa.detach()
        fake_b = atob.detach()
        d_a_fake = self.d_a(fake_a)
        d_b_fake = self.d_b(fake_b)
        d_a_loss = 0.5*(self.loss(self.d_a(A_real), 1) +
                        self.loss(d_a_fake, 0))
        d_b_loss = 0.5*(self.loss(self.d_b(B_real), 1) +
                        self.loss(d_b_fake, 0))
        return d_a_loss, d_b_loss

    def compute_d_norms(self, A_real_, B_real_):
        A_real = Variable(A_real_.data, requires_grad=True)
        B_real = Variable(B_real_.data, requires_grad=True)
        d_a_real = self.d_a(A_real)
        d_b_real = self.d_b(B_real)
        this_ones_dafake = torch.ones(d_a_real.size())
        this_ones_dbfake = torch.ones(d_b_real.size())
        if self.use_cuda:
            this_ones_dafake = this_ones_dafake.cuda()
            this_ones_dbfake = this_ones_dbfake.cuda()
        gradients_da = grad(outputs=d_a_real,
                            inputs=A_real,
                            grad_outputs=this_ones_dafake,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]
        gradients_db = grad(outputs=d_b_real,
                            inputs=B_real,
                            grad_outputs=this_ones_dbfake,
                            create_graph=True,
                            retain_graph=True,
                            only_inputs=True)[0]
        gp_a = ((gradients_da.view(gradients_da.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        gp_b = ((gradients_db.view(gradients_db.size()[0], -1).norm(2, 1) - 1) ** 2).mean()
        return gp_a, gp_b
    
    def _train(self):
        self.g_atob.train()
        self.g_btoa.train()
        self.d_a.train()
        self.d_b.train()

    def _eval(self):
        self.g_atob.eval()
        self.g_btoa.eval()
        self.d_a.eval()
        self.d_b.eval()

    def train_on_instance(self, A_real, B_real):
        """Train the network on a single example"""
        self._train()
        atob = self.g_atob(A_real)
        atob_btoa = self.g_btoa(atob)
        atob_gen_loss, cycle_aba = self.compute_g_losses_aba(
            A_real, atob, atob_btoa)
        g_tot_loss = atob_gen_loss + self.lamb*cycle_aba
        if self.beta > 0:
            cycle_id_a = torch.mean(torch.abs(A_real - self.g_btoa(A_real)))
            g_tot_loss = g_tot_loss + self.beta*cycle_id_a
        self.optim['g'].zero_grad()
        g_tot_loss.backward()
        btoa = self.g_btoa(B_real)
        btoa_atob = self.g_atob(btoa)
        btoa_gen_loss, cycle_bab = self.compute_g_losses_bab(
            B_real, btoa, btoa_atob)
        g_tot_loss = btoa_gen_loss + self.lamb*cycle_bab
        if self.beta > 0:
            cycle_id_b = torch.mean(torch.abs(B_real - self.g_atob(B_real)))
            g_tot_loss = g_tot_loss + self.beta*cycle_id_b
        g_tot_loss.backward()
        d_a_loss, d_b_loss = self.compute_d_losses(A_real, atob, B_real, btoa)
        self.optim['d_a'].zero_grad()
        self.optim['d_b'].zero_grad()
        d_a_loss.backward()
        d_b_loss.backward()
        if self.dnorm is not None and self.dnorm > 0.:
            gp_a, gp_b = self.compute_d_norms(A_real, B_real)
            (gp_a*self.dnorm).backward(retain_graph=True)
            (gp_b*self.dnorm).backward(retain_graph=True)
        self.optim['g'].step()
        self.optim['d_a'].step()
        self.optim['d_b'].step()
        losses = {
            'atob_gen': atob_gen_loss.item(),
            'cycle_aba': cycle_aba.item(),
            'btoa_gen': btoa_gen_loss.item(),
            'cycle_bab': cycle_bab.item(),
            'd_a': d_a_loss.item(),
            'd_b': d_b_loss.item()
        }
        if self.beta > 0:
            losses['cycle_id_a'] = cycle_id_a.item()
            losses['cycle_id_b'] = cycle_id_b.item()
        if self.dnorm is not None and self.dnorm > 0.:
            losses['gp_a'] = gp_a.item()
            losses['gp_b'] = gp_b.item()
        outputs = {
            'atob': atob.detach(),
            'atob_btoa': atob_btoa.detach(),
            'btoa': btoa.detach(),
            'btoa_atob': btoa_atob.detach()
        }
        return losses, outputs

    def eval_on_instance(self, A_real, B_real):
        """Train the network on a single example"""
        self._eval()
        with torch.no_grad():
            atob = self.g_atob(A_real)
            atob_btoa = self.g_btoa(atob)
            atob_gen_loss, cycle_aba = self.compute_g_losses_aba(
                A_real, atob, atob_btoa)
            if self.beta > 0:
                cycle_id_a = torch.mean(torch.abs(A_real - self.g_btoa(A_real)))
            btoa = self.g_btoa(B_real)
            btoa_atob = self.g_atob(btoa)
            btoa_gen_loss, cycle_bab = self.compute_g_losses_bab(
                B_real, btoa, btoa_atob)
            if self.beta > 0:
                cycle_id_b = torch.mean(torch.abs(B_real - self.g_atob(B_real)))
            d_a_loss, d_b_loss = self.compute_d_losses(
                A_real, atob, B_real, btoa)
        losses = {
            'atob_gen': atob_gen_loss.item(),
            'cycle_aba': cycle_aba.item(),
            'btoa_gen': btoa_gen_loss.item(),
            'cycle_bab': cycle_bab.item(),
            'd_a': d_a_loss.item(),
            'd_b': d_b_loss.item()
        }
        if self.beta > 0:
            losses['cycle_id_a'] = cycle_id_a.item()
            losses['cycle_id_b'] = cycle_id_b.item()
        outputs = {
            'atob': atob.detach(),
            'atob_btoa': atob_btoa.detach(),
            'btoa': btoa.detach(),
            'btoa_atob': btoa_atob.detach()
        }
        return losses, outputs

    def _get_stats(self, dict_, mode):
        """
        From a dict of training/valid statistics, create a
          summarised dict for use with the progress bar.
        """
        allowed_keys = ['atob_gen', 'btoa_gen', 'd_a', 'd_b', 'gp_a', 'gp_b']
        allowed_keys = ['%s_%s' % (mode, key) for key in allowed_keys]
        stats = OrderedDict({})
        for key in dict_.keys():
            if key in allowed_keys:
                stats[key] = np.mean(dict_[key])
        return stats

    def save(self, filename):
        torch.save(
            (self.g_atob.state_dict(),
             self.g_btoa.state_dict(),
             self.d_a.state_dict(),
             self.d_b.state_dict()),
            filename)

    def load(self, filename):
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        g_atob, g_btoa, d_a, d_b = torch.load(
            filename, map_location=map_location)
        self.g_atob.load_state_dict(g_atob)
        self.g_btoa.load_state_dict(g_btoa)
        self.d_a.load_state_dict(d_a)
        self.d_b.load_state_dict(d_b)
