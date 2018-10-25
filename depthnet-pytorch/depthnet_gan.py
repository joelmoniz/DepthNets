from __future__ import print_function

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import numpy as np
# torch imports
import torch
import torch.optim as optim
from torch.autograd import grad
# torchvision
from collections import OrderedDict
from tqdm import tqdm
import util

def save_handler(results_dir):
    vis_dir = "%s/vis" % results_dir
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    def fn(losses, inputs, outputs, kwargs):
        xy_keypts_src, z_keypts_src, xy_keypts_tgt, z_keypts_tgt = \
            inputs # already in numpy
        xy_keypts_src = xy_keypts_src.swapaxes(1,2) # (bs,2,66)
        xy_keypts_tgt = xy_keypts_tgt.swapaxes(1,2)
        b = kwargs['iter']
        epoch = kwargs['epoch']
        mode = kwargs['mode']
        if b == 1:
            fig = plt.figure(figsize=(20,6))
            # Do the 2D ground truth keypts.
            ax = fig.add_subplot(161)
            # For now just save fig for first element in
            # the minibatch.
            ax.scatter(xy_keypts_src[0][0],
                       xy_keypts_src[0][1])
            ax.set_title('src 2D keypt')
            ax.invert_xaxis()
            ax.invert_yaxis()
            # Do the predicted 3D keypts.
            src_z_pred = outputs['src_z_pred'].data.cpu().numpy()
            ax = fig.add_subplot(162, projection='3d')
            ax.scatter(xy_keypts_src[0][0],
                       xy_keypts_src[0][1],
                       src_z_pred[0][0])
            ax.view_init(30, 30)
            ax.set_title('src 2D + pred z')
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.invert_zaxis()
            # Do the ground truth src 3d keypts.
            ax = fig.add_subplot(163, projection='3d')
            ax.scatter(xy_keypts_src[0][0],
                       xy_keypts_src[0][1],
                       z_keypts_src[0])
            ax.view_init(30, 30)
            ax.set_title('src 2D + real z')
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.invert_zaxis()
            # Do the ground truth tgt 3d keypts
            ax = fig.add_subplot(164, projection='3d')
            ax.scatter(xy_keypts_tgt[0][0],
                       xy_keypts_tgt[0][1],
                       z_keypts_tgt[0])
            ax.view_init(30, 30)
            ax.set_title('tgt 2D + real z')
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.invert_zaxis()
            # Do the predicted tgt 2d
            ax = fig.add_subplot(165)
            tgt_pred = outputs['tgt_2d_pred'].data.cpu().numpy()
            ax.scatter(tgt_pred[0][0],
                       tgt_pred[0][1])
            ax.set_title('tgt pred')
            ax.invert_xaxis()
            ax.invert_yaxis()
            # Do the ground truth tgt 2d
            ax = fig.add_subplot(166)
            ax.scatter(xy_keypts_tgt[0][0],
                       xy_keypts_tgt[0][1])
            ax.set_title('tgt 2D')
            ax.invert_xaxis()
            ax.invert_yaxis()
            
            fig.savefig('%s/vis/%s_%i.png' % (results_dir, mode, epoch))
    return fn


class DepthNetGAN():

    def __init__(self,
                 g_fn,
                 d_fn,
                 opt_g=optim.Adam,
                 opt_d=optim.Adam,
                 opt_d_args={'lr':0.0002, 'betas':(0.5, 0.999)},
                 opt_g_args={'lr':0.0002, 'betas':(0.5, 0.999)},
                 lamb=1.,
                 sigma=1.,
                 detach=False,
                 l2_decay=0.,
                 dnorm=0.,
                 use_l1=False,
                 no_gan=False,
                 cheat=False,
                 update_g_every=1.,
                 handlers=[],
                 scheduler_fn=None,
                 scheduler_args={},
                 use_cuda='detect'):
        """
        Parameters
        ----------
        lamb: keypt loss coefficient
        l2_decay: weight decay coefficient
        dnorm: gradient norm penalty for GAN (if enabled)
        use_l1: use L1 for keypt loss instead of L2
        cheat: regress the ground truth src depth (uses same
          coef as lambda)
        update_g_every: update the generator this many iterations.
          Only makes sense if GAN is enabled.
        """
        assert use_cuda in [True, False, 'detect']
        if dnorm > 0. and no_gan:
            raise Exception("dnorm cannot be > 0 if you're not using GAN")
        if use_cuda == 'detect':
            use_cuda = True if torch.cuda.is_available() else False
        self.lamb = lamb
        self.g = g_fn
        self.d = d_fn
        self.detach = detach
        self.sigma = sigma
        self.dnorm = dnorm
        self.use_l1 = use_l1
        self.cheat = cheat
        self.no_gan = no_gan
        self.update_g_every = update_g_every
        optim_g = opt_g(self.g.parameters(),
                        weight_decay=l2_decay, **opt_g_args)
        optim_d = opt_d(self.d.parameters(),
                        weight_decay=l2_decay, **opt_d_args)
        self.optim = {'g': optim_g, 'd': optim_d}
        self.scheduler = {}
        if scheduler_fn is not None:
            for key in self.optim:
                self.scheduler[key] = scheduler_fn(
                    self.optim[key], **scheduler_args)
        self.handlers = handlers
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.g.cuda()
            self.d.cuda()
        self.last_epoch = 0

    def mse(self, prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.cuda()
        return torch.nn.MSELoss()(prediction, target)
    
    def _train(self):
        self.g.train()
        self.d.train()

    def _eval(self):
        self.g.eval()
        self.d.eval()

    def construct_A(self, src_kps, src_z_pred):
        K = 66
        bs = src_kps.shape[0]
        # TODO: make more efficient
        A = np.zeros((bs, K*2, 8))
        for b in range(bs):
            c = 0
            for i in range(0, A.shape[1]-1, 2):
                A[b, i, 0] = src_kps[b, 0, c] # xi
                A[b, i, 1] = src_kps[b, 1, c] # yi
                #A[i,2] = z_pred[c] # zi
                A[b, i, -2] = 1.
                #
                A[b, i+1, 4] = src_kps[b, 0, c] # xi
                A[b, i+1, 5] =  src_kps[b, 1, c] # yi
                #A[i+1,6] = z_pred[c] # zi
                A[b, i+1, -1] = 1.
                c += 1
        A = torch.from_numpy(A).float()
        if self.use_cuda:
            A = A.cuda()
        for b in range(bs):
            c = 0
            for i in range(0, A.size(1)-1, 2):
                A[b, i, 2] = src_z_pred[b, 0, c] # zi
                A[b, i+1, 6] = src_z_pred[b, 0, c] # zi
                c += 1
        return A

    def grad_norm(self, d_out, x):
        ones = torch.ones(d_out.size())
        if self.use_cuda:
            ones = ones.cuda()
        grad_wrt_x = grad(outputs=d_out, inputs=x,
                          grad_outputs=ones,
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
        g_norm = (grad_wrt_x.view(
            grad_wrt_x.size()[0], -1).norm(2, 1)** 2).mean()
        return g_norm
    
    def inv(self, x):
        # https://github.com/pytorch/pytorch/pull/1670
        eye = torch.eye(8).float()
        if self.use_cuda:
            eye = eye.cuda()
        return torch.inverse( (self.sigma*eye) +x)
        
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
        is_gan = not self.no_gan
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
        src_z_pred = self.g(
            torch.cat((xy_keypts_src_torch,
                       xy_keypts_tgt_torch), dim=1)
        ).unsqueeze(1) # (bs,1,66)
        if kwargs['use_gt_z']:
            # NOTE: ONLY USE FOR NON-MODEL VALIDATION
            # Use the ground truth z's to construct the
            # A matrix instead of predicted depths.
            A = self.construct_A(xy_keypts_src,
                                 z_keypts_src_torch.unsqueeze(1))
        else:
            if self.detach:
                A = self.construct_A(xy_keypts_src,
                                     src_z_pred.detach())
            else:
                A = self.construct_A(xy_keypts_src,
                                     src_z_pred)
        xt = xy_keypts_tgt.reshape((bs, 2*66), order='F') # (bs, 66*2)
        xt = torch.from_numpy(xt).float()
        if self.use_cuda:
            xt = xt.cuda()
        X1 = [self.inv(mat) for mat in
              torch.matmul(A.transpose(2, 1), A)]
        X1 = torch.stack(X1)
        X2 = torch.bmm(A.transpose(2, 1), xt.unsqueeze(2))
        m = torch.bmm(X1, X2) # (bs,8,1)
        m_rshp = m.squeeze(2).view((bs, 2, 4))
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
        if self.cheat:
            cheat_l1_loss = torch.mean(
                torch.abs(src_z_pred - z_keypts_src_torch.unsqueeze(1)))
        # Now do the adversarial losses.
        src_z_pred_given_inp = torch.cat(
            (src_z_pred, xy_keypts_src_torch), dim=1)
        if train:
            (self.lamb*l2_loss).backward(retain_graph=True)
            if self.cheat:
                cheat_l1_loss.backward(retain_graph=True)
        if is_gan:
            g_loss = -torch.mean(self.d(src_z_pred_given_inp))
            if train:
                g_loss.backward()
        if train:
            if (kwargs['iter']-1) % self.update_g_every == 0:
                self.optim['g'].step()
        # Now do the discriminator
        if is_gan:
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
        if is_gan and self.dnorm > 0.:
            d_real_inp = src_z_gt_given_inp.detach()
            d_real_inp.requires_grad = True
            d_real_ = self.d(d_real_inp)
            g_norm_x = self.grad_norm(
                d_real_, d_real_inp)
            if train:
                self.optim['d'].zero_grad()
                (g_norm_x*self.dnorm).backward()
                self.optim['d'].step()
        losses = {'l2': l2_loss.data.item()}
        if is_gan:
            losses['g_loss'] = g_loss.data.item(),
            losses['d_loss'] = d_loss.data.item(),
            losses['d_loss_real'] = d_loss_real.data.item(),
            losses['d_loss_fake'] = d_loss_fake.data.item(),
            if self.dnorm > 0:
                losses['dnorm_x'] = g_norm_x.data.item()
        if self.cheat:
            losses['cheat_l1_loss'] = cheat_l1_loss.data.item()
        outputs = {
            'src_z_pred': src_z_pred,
            'tgt_2d_pred': rhs
        }
        return losses, outputs

    def train_on_instance(self,
                          xy_keypts_src,
                          z_keypts_src,
                          xy_keypts_tgt,
                          z_keypts_tgt,
                          **kwargs):
        self._train()
        return self.run_on_instance(
            xy_keypts_src,
            z_keypts_src,
            xy_keypts_tgt,
            z_keypts_tgt,
            train=True,
            **kwargs
        )
    
    def eval_on_instance(self,
                         xy_keypts_src,
                         z_keypts_src,
                         xy_keypts_tgt,
                         z_keypts_tgt,
                         **kwargs):
        self._eval()
        return self.run_on_instance(
            xy_keypts_src,
            z_keypts_src,
            xy_keypts_tgt,
            z_keypts_tgt,
            train=False,
            **kwargs
        )

    def eval_no_model(self, itr_valid):
        valid_dict = OrderedDict()
        for b, ((xy_keypts_src, z_keypts_src), (xy_keypts_tgt, z_keypts_tgt)) \
            in enumerate(itr_valid):
            # HACK: need to convert back to numpy here.
            # TODO: cleanup
            xy_keypts_src = xy_keypts_src.numpy()
            z_keypts_src = z_keypts_src.numpy()
            xy_keypts_tgt = xy_keypts_tgt.numpy()
            z_keypts_tgt = z_keypts_tgt.numpy()
            losses, outputs = self.eval_on_instance(
                xy_keypts_src, z_keypts_src,
                xy_keypts_tgt, z_keypts_tgt,
                iter=b+1,
                use_gt_z=True)
            for key in losses:
                this_key = 'valid_%s' % key
                if this_key not in valid_dict:
                    valid_dict[this_key] = []
                valid_dict[this_key].append(losses[key])
        for key in valid_dict:
            print("mean %s = %f" % (key, np.mean(valid_dict[key])))


    def _get_stats(self, dict_, mode):
        """
        From a dict of training/valid statistics, create a
          summarised dict for use with the progress bar.
        """
        stats = OrderedDict({})
        for key in dict_.keys():
            if 'epoch' not in key:
                stats[key] = np.mean(dict_[key])
        return stats

    def train(self,
              itr_train,
              itr_valid,
              epochs,
              model_dir,
              result_dir,
              append=False,
              save_every=1,
              scheduler_fn=None,
              scheduler_args={},
              verbose=True):
        for folder_name in [model_dir, result_dir]:
            if folder_name is not None and not os.path.exists(folder_name):
                os.makedirs(folder_name)
        f_mode = 'w' if not append else 'a'
        f = None
        if result_dir is not None:
            f = open("%s/results.txt" % result_dir, f_mode)
        for epoch in range(self.last_epoch, epochs):
            # Training
            epoch_start_time = time.time()
            if verbose:
                pbar = tqdm(total=len(itr_train))
            train_dict = OrderedDict({'epoch': epoch+1})
            for b, ((xy_keypts_src, z_keypts_src), (xy_keypts_tgt, z_keypts_tgt)) \
                in enumerate(itr_train):
                # HACK: need to convert back to numpy here.
                # TODO: cleanup
                xy_keypts_src = xy_keypts_src.numpy()
                z_keypts_src = z_keypts_src.numpy()
                xy_keypts_tgt = xy_keypts_tgt.numpy()
                z_keypts_tgt = z_keypts_tgt.numpy()
                losses, outputs = self.train_on_instance(
                    xy_keypts_src, z_keypts_src,
                    xy_keypts_tgt, z_keypts_tgt,
                    iter=b+1,
                    use_gt_z=False)
                for key in losses:
                    this_key = 'train_%s' % key
                    if this_key not in train_dict:
                        train_dict[this_key] = []
                    train_dict[this_key].append(losses[key])
                pbar.update(1)
                pbar.set_postfix(self._get_stats(train_dict, 'train'))
                # Process handlers.
                for handler_fn in self.handlers:
                    handler_fn(losses,
                               (xy_keypts_src, z_keypts_src, xy_keypts_tgt, z_keypts_tgt),
                               outputs,
                               {'epoch':epoch+1, 'iter':b+1, 'mode':'train'})
            if verbose:
                pbar.close()
                pbar = tqdm(total=len(itr_valid))
            valid_dict = {}
            for b, ((xy_keypts_src, z_keypts_src), (xy_keypts_tgt, z_keypts_tgt)) \
                in enumerate(itr_valid):
                # HACK: need to convert back to numpy here.
                # TODO: cleanup
                xy_keypts_src = xy_keypts_src.numpy()
                z_keypts_src = z_keypts_src.numpy()
                xy_keypts_tgt = xy_keypts_tgt.numpy()
                z_keypts_tgt = z_keypts_tgt.numpy()
                losses, outputs = self.eval_on_instance(
                    xy_keypts_src, z_keypts_src,
                    xy_keypts_tgt, z_keypts_tgt,
                    iter=b+1,
                    use_gt_z=False)
                for key in losses:
                    this_key = 'valid_%s' % key
                    if this_key not in valid_dict:
                        valid_dict[this_key] = []
                    valid_dict[this_key].append(losses[key])
                pbar.update(1)
                pbar.set_postfix(self._get_stats(valid_dict, 'valid'))
                # Process handlers.
                for handler_fn in self.handlers:
                    handler_fn(losses,
                               (xy_keypts_src, z_keypts_src, xy_keypts_tgt, z_keypts_tgt),
                               outputs,
                               {'epoch':epoch+1, 'iter':b+1, 'mode':'valid'})
            if verbose:
                pbar.close()
            # Step learning rates.
            for key in self.scheduler:
                self.scheduler[key].step()
            all_dict = train_dict
            all_dict.update(valid_dict)
            for key in all_dict:
                all_dict[key] = np.mean(all_dict[key])
            for key in self.optim:
                all_dict["lr_%s" % key] = \
                    self.optim[key].state_dict()['param_groups'][0]['lr']
            all_dict['time'] = \
                time.time() - epoch_start_time
            str_ = ",".join([str(all_dict[key]) for key in all_dict])
            print(str_)
            if f is not None:
                if (epoch+1) == 1 and not append:
                    # If we're not resuming, then write the header.
                    f.write(",".join(all_dict.keys()) + "\n")
                f.write(str_ + "\n")
                f.flush()
            if (epoch+1) % save_every == 0 and model_dir is not None:
                self.save(filename="%s/%i.pkl" % (model_dir, epoch+1),
                          epoch=epoch+1)
            
        if f is not None:
            f.close()

    def save(self, filename, epoch, legacy=False):
        if legacy:
            torch.save(
                (self.g.state_dict(),
                 self.d.state_dict()),
                filename)
        else:
            dd = {}
            dd['g'] = self.g.state_dict()
            dd['d'] = self.d.state_dict()
            for key in self.optim:
                dd['optim_' + key] = self.optim[key].state_dict()
            dd['epoch'] = epoch
            torch.save(dd, filename)

    def load(self, filename, legacy=False):
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        if legacy:
            g, d = torch.load(filename,
                              map_location=map_location)
            self.g.load_state_dict(g)
            self.d.load_state_dict(d)
        else:
            dd = torch.load(filename,
                            map_location=map_location)
            self.g.load_state_dict(dd['g'])
            self.d.load_state_dict(dd['d'])
            for key in self.optim:
                self.optim[key].load_state_dict(dd['optim_'+key])
            self.last_epoch = dd['epoch']

class zip_iter():
    def __init__(self, itr1, itr2):
        self.itr1 = itr1
        self.itr2 = itr2
        self.iter1 = None
        self.iter2 = None
        self.done = True
    def __len__(self):
        return min(len(self.itr1), len(self.itr2))
    def __iter__(self):
        return self
    def __next__(self):
        if self.done:
            self.iter1 = iter(self.itr1)
            self.iter2 = iter(self.itr2)
            self.done = False
        try:
            result1, result2 = \
                self.iter1.next(), self.iter2.next()
        except StopIteration:
            self.done = True
            raise StopIteration
        return result1, result2
        
if __name__ == '__main__':
    pass
