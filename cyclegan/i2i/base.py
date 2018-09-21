import sys
import os
import time
from tqdm import tqdm
import numpy as np
from collections import OrderedDict

class BaseModel():

    def load(self, args, **kwargs):
        raise NotImplementedError()

    def save(self, args, **kwargs):
        raise NotImplementedError()
   
    def train_on_instance(self, x, y):
        raise NotImplementedError()

    def eval_on_instance(self, x, y):
        raise NotImplementedError()

    def _get_stats(self, dict_, mode):
        return {}

    def _zip(self, A, B):
        if sys.version[0] == '2':
            from itertools import izip
            return izip(A, B)
        else:
            return zip(A, B)

    def prepare_batch(self, A_real, B_real):
        if type(A_real) == list:
            if len(A_real) > 1:
                raise Exception("A_real should be either a tensor " +
                                "of a list of one element")
            if len(B_real) > 1:
                raise Exception("A_real should be either a tensor " +
                                "of a list of one element")
            A_real = A_real[0]
            B_real = B_real[0]
        A_real, B_real = A_real.float(), B_real.float()
        if self.use_cuda:
            A_real, B_real = A_real.cuda(), B_real.cuda()
        return A_real, B_real

    def train(self,
              itr_a_train, itr_b_train,
              itr_a_valid, itr_b_valid,
              epochs, model_dir, result_dir,
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
                n_iters = min(len(itr_a_train), len(itr_b_train))
                pbar = tqdm(total=n_iters)
            train_dict = OrderedDict({'epoch': epoch+1})
            for b, (A_real, B_real) in enumerate(
                    self._zip(itr_a_train, itr_b_train)):
                A_real, B_real = self.prepare_batch(A_real, B_real)
                losses, outputs = self.train_on_instance(A_real, B_real)
                for key in losses:
                    this_key = 'train_%s' % key
                    if this_key not in train_dict:
                        train_dict[this_key] = []
                    train_dict[this_key].append(losses[key])
                pbar.update(1)
                pbar.set_postfix(self._get_stats(train_dict, 'train'))
                # Process handlers.
                for handler_fn in self.handlers:
                    handler_fn(losses, (A_real, B_real), outputs,
                               {'epoch':epoch+1, 'iter':b+1, 'mode':'train'})
            if verbose:
                pbar.close()
            # Validation
            valid_dict = {}
            if itr_a_valid is not None and itr_b_valid is not None:
                if verbose:
                    n_iters = min(len(itr_a_valid), len(itr_b_valid))
                    pbar = tqdm(total=n_iters)
                for b, (A_real, B_real) in enumerate(
                        self._zip(itr_a_valid, itr_b_valid)):
                    A_real, B_real = self.prepare_batch(A_real, B_real)
                    losses, outputs = self.eval_on_instance(A_real, B_real)
                    for key in losses:
                        this_key = 'valid_%s' % key
                        if this_key not in valid_dict:
                            valid_dict[this_key] = []
                        valid_dict[this_key].append(losses[key])
                    pbar.update(1)
                    pbar.set_postfix(self._get_stats(valid_dict, 'valid'))
                    for handler_fn in self.handlers:
                        handler_fn(losses, (A_real, B_real), outputs,
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
                self.save(filename="%s/%i.pkl" % (model_dir, epoch+1))
        if f is not None:
            f.close()
