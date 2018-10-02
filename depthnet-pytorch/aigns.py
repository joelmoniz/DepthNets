import time
import os
import numpy as np
# torch imports
import torch
import torch.optim as optim
from torch.autograd import Variable
# torchvision
from collections import OrderedDict
# matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# misc
from tqdm import tqdm
import util

def save_handler(results_dir):
    vis_dir = "%s/vis" % results_dir
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    def fn(losses, inputs, outputs, kwargs):
        X_keypts, y_keypts, z_keypts = inputs
        y_keypts_t = y_keypts.transpose(1,2)
        b = kwargs['iter']
        epoch = kwargs['epoch']
        mode = kwargs['mode']
        if b == 1:
            # Do the 3D ground truth.
            fig = plt.figure(figsize=(20,6))
            ax = fig.add_subplot(141, projection='3d')
            #width, height = fig.get_size_inches()*fig.get_dpi()
            x_3d = torch.cat((y_keypts_t,
                              z_keypts.unsqueeze(1)), 1)
            # TODO: are these axes right??
            ax.scatter(x_3d[0][0].data.cpu().numpy(),
                       x_3d[0][1].data.cpu().numpy(),
                       x_3d[0][2].data.cpu().numpy())
            ax.set_title('3D keypt (GT)')
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.invert_zaxis()
            # Do the 2D keypts.
            ax = fig.add_subplot(142)
            #y_keypts = outputs['y_keypts']
            # For now just save fig for first element in
            # the minibatch.
            ax.scatter(y_keypts_t.data.cpu().numpy()[0][0],
                       y_keypts_t.data.cpu().numpy()[0][1])
            ax.set_title('2D keypt (GT)')
            ax.invert_xaxis()
            ax.invert_yaxis()
            # Do the predicted 3D keypts.
            x_3d_pred = outputs['x_3d'] # FIX NAME
            ax = fig.add_subplot(143, projection='3d')
            ax.scatter(x_3d_pred[0][0].data.cpu().numpy(),
                       x_3d_pred[0][1].data.cpu().numpy(),
                       x_3d_pred[0][2].data.cpu().numpy())
            ax.set_title('3D prediction')
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.invert_zaxis()
            # Do the cycle (projected back into 2D)
            ax = fig.add_subplot(144)
            x_2d_proj = outputs['x_2d_proj']
            ax.scatter(x_2d_proj.data.cpu().numpy()[0][0],
                       x_2d_proj.data.cpu().numpy()[0][1])
            ax.set_title('3D->2D projection')
            ax.invert_xaxis()
            ax.invert_yaxis()
            fig.savefig('%s/%s_%i.png' % (vis_dir, mode, epoch))
    return fn

def params_to_3d(dd, cuda=False):
    rots = dd['rot_model']
    bs = dd['alpha_model'].size()[0]
    SIN_THETA = 0
    COS_THETA = 1
    SIN_PHI = 2
    COS_PHI = 3
    SIN_PSI = 4
    COS_PSI = 5
    # Construct rotation matrix B
    rot_B = torch.zeros(bs, 3, 3).float()
    if cuda:
        rot_B = rot_B.cuda()
    rot_B[:, 0, 0] = rots[:, SIN_PSI]
    rot_B[:, 0, 1] = -rots[:, COS_PSI]
    rot_B[:, 0, 2] = 0.
    rot_B[:, 1, 0] = rots[:, COS_PSI]
    rot_B[:, 1, 1] = rots[:, SIN_PSI]
    rot_B[:, 1, 2] = 0.
    rot_B[:, 2, 0] = 0.
    rot_B[:, 2, 1] = 0.
    rot_B[:, 2, 2] = 1.
    # Construct rotation matrix C
    rot_C = torch.zeros(bs, 3, 3).float()
    if cuda:
        rot_C = rot_C.cuda()
    rot_C[:, 0, 0] = 1.
    rot_C[:, 0, 1] = 0.
    rot_C[:, 0, 2] = 0.
    rot_C[:, 1, 0] = 0.
    rot_C[:, 1, 1] = rots[:, COS_THETA]
    rot_C[:, 1, 2] = rots[:, SIN_THETA]
    rot_C[:, 2, 0] = 0.
    rot_C[:, 2, 1] = -rots[:, SIN_THETA]
    rot_C[:, 2, 2] = rots[:, COS_THETA]
    # Construct rotation matrix D.
    rot_D = torch.zeros(bs, 3, 3).float()
    if cuda:
        rot_D = rot_D.cuda()
    rot_D[:, 0, 0] = -rots[:, SIN_PHI]
    rot_D[:, 0, 1] = rots[:, COS_PHI]
    rot_D[:, 0, 2] = 0.
    rot_D[:, 1, 0] = -rots[:, COS_PHI]
    rot_D[:, 1, 1] = -rots[:, SIN_PHI]
    rot_D[:, 1, 2] = 0.
    rot_D[:, 2, 0] = 0.
    rot_D[:, 2, 1] = 0.
    rot_D[:, 2, 2] = 1.
    # R = BCD
    R_mat = torch.bmm(torch.bmm(rot_B, rot_C), rot_D)
    alpha = dd['alpha_model']
    alpha_reshp = alpha.view(-1, 3, 66)
    # x_3d = R * alpha + T
    new_alpha = torch.bmm(R_mat, alpha_reshp)
    T = dd['T_model'].unsqueeze(2).repeat(1, 1, 66)
    x_3d = (T + new_alpha)
    return x_3d

def project_3d_to_2d(dd, x_3d, cuda=False):
    bs = x_3d.size(0)
    ones = torch.ones(bs, 1, 66).float()
    if cuda:
        ones = ones.cuda()
    x_3d_row1 = torch.cat((x_3d, ones), dim=1)
    # Construct projection matrix P.
    P = torch.zeros(bs, 3, 4).float()
    if cuda:
        P = P.cuda()
    P[:, 0, 0] = dd['f_model'][:, 0]
    P[:, 1, 1] = dd['f_model'][:, 1]
    P[:, 2, 2] = 1.0
    # x_3d = P*x_3d + [cx, cy, 0]^T 
    proj = torch.bmm(P, x_3d_row1)
    C = dd['c_model'].unsqueeze(1).transpose(2, 1)
    col_zeros = torch.zeros(bs, 1, 1).float()
    if cuda:
        col_zeros = col_zeros.cuda()
    C1 = torch.cat((C, col_zeros), 1)
    proj_bias = proj + C1
    proj_2d = proj_bias[:, 0:2, :]
    return proj_2d
        
class AIGN():
    def __init__(self,
                 g_fn,
                 d_fn,
                 opt_g=optim.Adam,
                 opt_d=optim.Adam,
                 opt_d_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 opt_g_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
                 lamb=10.,
                 handlers=[],
                 scheduler_fn=None,
                 scheduler_args={},
                 use_cuda='detect'):
        assert use_cuda in [True, False, 'detect']
        if use_cuda == 'detect':
            use_cuda = True if torch.cuda.is_available() else False
        self.lamb = lamb
        self.g = g_fn
        self.d = d_fn
        optim_g = opt_g(self.g.parameters(), **opt_g_args)
        optim_d = opt_d(self.d.parameters(), **opt_d_args)
        self.optim = {
            'g': optim_g,
            'd': optim_d,
        }
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

    def mse(self, prediction, target):
        if not hasattr(target, '__len__'):
            target = torch.ones_like(prediction)*target
            if prediction.is_cuda:
                target = target.cuda()
            target = Variable(target)
        return torch.nn.MSELoss()(prediction, target)
    
    def _train(self):
        self.g.train()
        self.d.train()

    def _eval(self):
        self.g.eval()
        self.d.eval()

    def prepare_batch(self, y_keypts, z_keypts):
        # Construct X_keypts from y_keypts
        keypts_uint8 = y_keypts * 128.
        X_keypts = np.zeros((y_keypts.shape[0], 66, 128, 128))
        for b in range(len(keypts_uint8)):
            for i in range(66):
                X_keypts[b][i] = util.get_fm_for_xy(int(keypts_uint8[b][i, 0]),
                                                    int(keypts_uint8[b][i, 1]))
        X_keypts = torch.from_numpy(X_keypts).float()
        #X_keypts0 = X_keypts[0]
        #tmp = np.zeros((128,128))
        #for i in range(66):
        #    tmp += X_keypts0[i]
        if self.use_cuda:
            X_keypts = X_keypts.cuda()
            y_keypts = y_keypts.cuda()
            z_keypts = z_keypts.cuda()
        return X_keypts, y_keypts, z_keypts

    def train_on_instance(self, X_keypts, y_keypts, z_keypts, **kwargs):
        """Train the network on a single example"""
        self._train()
        self.optim['g'].zero_grad()
        self.optim['d'].zero_grad()
        # X_keypts = (bs, 66, 128, 128)
        # y_keypts = (bs, 66, 2)
        # z_keypts = (bs, 66)
        # y_keypts_t = (bs, 2, 66)
        # gt_3d_keypts = (bs, 3, 66)
        y_keypts_t = y_keypts.transpose(1, 2)
        gt_3d_keypts = torch.cat((y_keypts_t,
                                  z_keypts.unsqueeze(1)), 1)
        pred_params = self.g(X_keypts)
        x_3d = params_to_3d(pred_params, self.use_cuda)
        # Ok, try and fool the discriminator into thinking
        # this is real.
        g_loss = self.mse(self.d(x_3d), 1)
        # Ok, project back into 2D.
        x_2d_proj = project_3d_to_2d(
            pred_params, x_3d, self.use_cuda)
        # Now compute the L1 loss between this
        # and the actual 2d.
        l1_loss = torch.mean(torch.abs(x_2d_proj - y_keypts_t))
        g_tot_loss = g_loss + self.lamb*l1_loss
        g_tot_loss.backward()
        self.optim['g'].step()
        # Ok, now do the discriminator.
        # We need to concatenate the true y and true z
        self.optim['d'].zero_grad()
        d_loss_fake = self.mse(self.d(x_3d.detach()), 0)
        d_loss_real = self.mse(self.d(gt_3d_keypts), 1)
        d_loss = d_loss_fake + d_loss_real
        d_loss.backward()
        self.optim['d'].step()

        losses = {
            'g_loss': g_loss.data.item(),
            'l1_loss': l1_loss.data.item(),
            'd_loss': d_loss.data.item()
        }
        outputs = {
            'gt_3d': gt_3d_keypts,
            'x_3d': x_3d,
            'x_2d_proj': x_2d_proj
        }
        return losses, outputs

    def eval_on_instance(self, X_keypts, y_keypts, z_keypts, **kwargs):
        """Train the network on a single example"""
        self._eval()
        with torch.no_grad():
            y_keypts_t = y_keypts.transpose(1, 2)
            gt_3d_keypts = torch.cat((y_keypts_t,
                                      z_keypts.unsqueeze(1)), 1)
            pred_params = self.g(X_keypts)
            x_3d = params_to_3d(pred_params, self.use_cuda)
            # Ok, try and fool the discriminator into thinking
            # this is real.
            g_loss = self.mse(self.d(x_3d), 1)
            # Ok, project back into 2D.
            x_2d_proj = project_3d_to_2d(
                pred_params, x_3d, self.use_cuda)
            # Now compute the L1 loss between this
            # and the actual 2d.
            l1_loss = torch.mean(torch.abs(x_2d_proj - y_keypts_t))
            l2_loss = torch.mean((x_2d_proj - y_keypts_t)**2)
            g_tot_loss = g_loss + self.lamb*l1_loss
            # Ok, now do the discriminator.
            # We need to concatenate the true y and true z
            d_loss_fake = self.mse(self.d(x_3d.detach()), 0)
            d_loss_real = self.mse(self.d(gt_3d_keypts), 1)
            d_loss = d_loss_fake + d_loss_real
        losses = {
            'g_loss': g_loss.data.item(),
            'l1_loss': l1_loss.data.item(),
            'l2_loss': l2_loss.data.item(),
            'd_loss': d_loss.data.item()
        }
        outputs = {'x_3d': x_3d,
                   'x_2d_proj': x_2d_proj,
                   'gt_3d_keypts': gt_3d_keypts}
        return losses, outputs

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
        for epoch in range(epochs):
            # Training
            epoch_start_time = time.time()
            if verbose:
                pbar = tqdm(total=len(itr_train))
            train_dict = OrderedDict({'epoch': epoch+1})
            for b, (y_keypts, z_keypts) in enumerate(itr_train):
                X_keypts, y_keypts, z_keypts = self.prepare_batch(
                    y_keypts, z_keypts)
                losses, outputs = self.train_on_instance(
                    X_keypts, y_keypts, z_keypts, iter=b+1)
                for key in losses:
                    this_key = 'train_%s' % key
                    if this_key not in train_dict:
                        train_dict[this_key] = []
                    train_dict[this_key].append(losses[key])
                pbar.update(1)
                pbar.set_postfix(self._get_stats(train_dict, 'train'))
                # Process handlers.
                for handler_fn in self.handlers:
                    handler_fn(losses, (X_keypts, y_keypts, z_keypts), outputs,
                               {'epoch': epoch+1, 'iter': b+1, 'mode': 'train'})
            if verbose:
                pbar.close()
                pbar = tqdm(total=len(itr_valid))
            valid_dict = {}
            for b, (y_keypts, z_keypts) in enumerate(itr_valid):
                X_keypts, y_keypts, z_keypts = self.prepare_batch(
                    y_keypts, z_keypts)
                losses, outputs = self.eval_on_instance(
                    X_keypts, y_keypts, z_keypts, iter=b+1)
                for key in losses:
                    this_key = 'valid_%s' % key
                    if this_key not in valid_dict:
                        valid_dict[this_key] = []
                    valid_dict[this_key].append(losses[key])
                pbar.update(1)
                pbar.set_postfix(self._get_stats(valid_dict, 'valid'))
                # Process handlers.
                for handler_fn in self.handlers:
                    handler_fn(losses, (X_keypts, y_keypts, z_keypts), outputs,
                               {'epoch': epoch+1, 'iter': b+1, 'mode': 'valid'})
            if verbose:
                pbar.close()
            # Step learning rates.
            for key in self.scheduler:
                self.scheduler[key].step()
            all_dict = train_dict
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
                self.save(filename="%s/%i.pkl" % (model_dir, epoch+1))
            
        if f is not None:
            f.close()

    def save(self, filename):
        torch.save(
            (self.g.state_dict(),
             self.d.state_dict()),
            filename)

    def load(self, filename):
        if not self.use_cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None
        g, d = torch.load(filename, map_location=map_location)
        self.g.load_state_dict(g)
        self.d.load_state_dict(d)

if __name__ == '__main__':
    from architectures import generator, discriminator, shared
    g = generator.get_network()
    d = discriminator.get_network()
    xfake = Variable(torch.randn((2,66,128,128)))
    #print(g)
    #print(d)
    dd = g(xfake)
    #for key in dd:
    #    print(key, " ", dd[key].size())
    x_3d = shared.params_to_3d(dd)
    print(x_3d.shape)
    x_2d = shared.project_3d_to_2d(dd, x_3d)
    print(x_2d.shape)

    #######
    from iterators import iterator
    loader = iterator.get_iterator(32)
    #yy,zz = iter(loader).next()
    
    kpg = AIGN(g_fn=g,
                      d_fn=d,
                      handlers=[save_handler],
                      lamb=100.)
    #yy,zz = kpg.prepare_batch(yy,zz)

    #print(kpg.train_on_instance(xx,yy,zz))
    #print(kpg.eval_on_instance(xx,yy,zz))

    kpg.train(itr=loader, epochs=100, model_dir=None, result_dir=None)
    
    import pdb
    pdb.set_trace()
    
