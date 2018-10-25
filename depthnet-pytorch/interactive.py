import torch
from iterators.iterator import (get_data_valid,
                                get_data_train,
                                get_data_test)
import numpy as np
from tqdm import tqdm
from scipy.stats.stats import pearsonr

'''
def measure_l2_pairwise(net, use_gt_z=False, dataset='valid'):
    net._eval()
    if dataset == 'valid':
        xy_keypts, z_keypts = get_data_valid()
    elif dataset == 'train':
        zy_keypts, z_keypts = get_data_train()
    else:
        raise Exception("todo")
    N = len(xy_keypts)
    pbar = tqdm(total=len(xy_keypts)**2)
    accum = []
    c = 0
    for i in range(len(xy_keypts)):
        for j in range(len(xy_keypts)):
            xy_keypts_src = xy_keypts[i][np.newaxis]
            z_keypts_src = z_keypts[i][np.newaxis]
            xy_keypts_tgt = xy_keypts[j][np.newaxis]
            z_keypts_tgt = z_keypts[j][np.newaxis]
            losses, _ = net.run_on_instance(xy_keypts_src=xy_keypts_src,
                                            z_keypts_src=z_keypts_src,
                                            xy_keypts_tgt=xy_keypts_tgt,
                                            z_keypts_tgt=z_keypts_tgt,
                                            train=False,
                                            use_gt_z=use_gt_z)
            pbar.update(1)
            accum.append(losses['l2'])
            c += 1
    print("mean / std l2: %, %ff" % (np.mean(accum), np.std(accum)))

def measure_l2_one_pass(net, use_gt_z=False, dataset='valid'):
    net._eval()
    if dataset == 'valid':
        xy_keypts, z_keypts = get_data_valid()
    elif dataset == 'train':
        xy_keypts, z_keypts = get_data_train()
    else:
        xy_keypts, z_keypts, _ = get_data_test()
    accum = []
    c = 0
    idxs = np.arange(0,len(xy_keypts))
    rnd_state = np.random.RandomState(0)
    rnd_state.shuffle(idxs)
    pbar = tqdm(total=len(xy_keypts))
    for i in range(len(xy_keypts)):
        xy_keypts_src = xy_keypts[i][np.newaxis]
        z_keypts_src = z_keypts[i][np.newaxis]
        xy_keypts_tgt = xy_keypts[idxs[i]][np.newaxis]
        z_keypts_tgt = z_keypts[idxs[i]][np.newaxis]
        losses, _ = net.run_on_instance(xy_keypts_src=xy_keypts_src,
                                        z_keypts_src=z_keypts_src,
                                        xy_keypts_tgt=xy_keypts_tgt,
                                        z_keypts_tgt=z_keypts_tgt,
                                        train=False,
                                        use_gt_z=use_gt_z)
        pbar.update(1)
        accum.append(losses['l2'])
        c += 1
    print("mean / std l2: %f, %f" % (np.mean(accum), np.std(accum)))

def measure_pearson_one_sweep(net, dataset='valid'):
    net._eval()
    if dataset == 'valid':
        y_keypts, z_keypts = get_data_valid()
    elif dataset == 'train':
        y_keypts, z_keypts = get_data_train()
        
    idxs = np.arange(0,len(y_keypts))
    rnd_state = np.random.RandomState(0)
    rnd_state.shuffle(idxs)
    preds = []
    actuals = []
    pearsons = []
    l2_losses = []
    for i in range(len(y_keypts)):
        if i == idxs[i]:
            continue
        print("%i / 5000" % i)
        src_keypt_xy = y_keypts[i][np.newaxis]
        src_keypt_z = z_keypts[i]
        tgt_keypt_xy = y_keypts[idxs[i]][np.newaxis]
        # Convert to Torch.
        src_keypt_xy = Variable(torch.from_numpy(src_keypt_xy))
        tgt_keypt_xy = Variable(torch.from_numpy(tgt_keypt_xy))
        
        net_out = net.g(
            torch.cat((src_keypt_xy.transpose(1,2),
                       tgt_keypt_xy.transpose(1,2)),dim=1)
        )

        if isinstance(net_out, tuple):
            # In case we're doing the learnm branch
            pred_src_z = net_out[0]
        else:
            pred_src_z = net_out

        pred_src_z = pred_src_z.data.cpu().numpy()[0]

        #plot_xy_and_z(xy_real=y_keypts[i].T,
        #              z_real=src_keypt_z,
        #              z_fake=pred_src_z,
        #              out_file="tmp/xyz_real_%i.png" % i)
        
        preds.append(pred_src_z)
        actuals.append(src_keypt_z)
        pearsons.append(
            pearsonr(pred_src_z.flatten(), src_keypt_z.flatten())[0])
        l2_losses.append( (pred_src_z.flatten() - src_keypt_z.flatten())**2 )
        
    preds = np.asarray(preds)
    actuals = np.asarray(actuals)
    # Subtract mean.
    preds = (preds - np.mean(preds, axis=0, keepdims=True))
    actuals = actuals - np.mean(actuals, axis=0, keepdims=True)
    #preds = actuals + np.random.normal(0,100,size=actuals.shape)
    covar = np.dot(preds.T, actuals) / (1.0*preds.shape[0])
    # We just want to know the magnitudes so square them.
    covar = covar**2
    result = np.sum(np.diag(covar)) / np.sum(covar)
    pearsons = np.asarray(pearsons)
    print("Result: sum(diag) / sum(all) = %f" % result)
    print("Pearson: %f +/- %f" % (np.mean(pearsons), np.std(pearsons)))
    print("R2: %f +/- %f" % (np.mean(pearsons**2), np.std(pearsons**2)))
    print("L2 distance: %f +/- %f" % (np.mean(l2_losses), np.std(l2_losses)))
    #np.savez(out_file, (preds, actuals))
    #open(out_file, 'w').write(vector_to_str_list(pearsons))
'''

def compute_covar(preds, actuals, n_kps=66):
    return np.sum(np.diag(np.abs(np.corrcoef(preds, actuals, rowvar=0)[0:n_kps,n_kps::])))
    
def measure_depth_test_pairwise(net, grid=True, dump_file=None, mode='test'):
    """
    Measure abs correlation matrix trace between predicted and real depths.

    Parameters
    ----------
    net:
    grid: if `True`, do this between pairwise combinations of left/right/center.
    dump_file: if `True`, dump the predicted depths to disk (only allowed if
      grid is `False`).
    mode: either 'valid' or 'test'. If set to 'valid', grid cannot be `True`.
      Furthermore, since the valid set is too big, when it is selected we
      only consider the first 225 examples.
    """

    if dump_file is not None and grid:
        raise Exception("Dumping npz only supported when grid=False!")

    net._eval()

    def fn(xy1_keypts, z1_keypts, xy2_keypts, z2_keypts,
           same=False, dump_file=None):
        pearsons = []
        l2_losses = []
        preds = []
        actuals = []
        map_ = None
        if dump_file is not None:
            n1 = len(xy1_keypts)
            n2 = len(xy2_keypts)
            map_ = np.zeros((n1,n2,66))
            
        for i in range(len(xy1_keypts)):
            for j in range(len(xy2_keypts)):
                if same and i==j:
                    continue
                src_keypt_xy = xy1_keypts[i][np.newaxis]
                src_keypt_z = z1_keypts[i]
                tgt_keypt_xy = xy2_keypts[j][np.newaxis]
                tgt_keypt_z = z2_keypts[j]
                # Convert to Torch.
                src_keypt_xy = torch.from_numpy(src_keypt_xy)
                tgt_keypt_xy = torch.from_numpy(tgt_keypt_xy)
                src_keypt_z = torch.from_numpy(src_keypt_z)
                tgt_keypt_z = torch.from_numpy(tgt_keypt_z)
                net_out = net.g(
                    torch.cat((src_keypt_xy.transpose(1,2),
                               tgt_keypt_xy.transpose(1,2)),dim=1)
                )
                if isinstance(net_out, tuple):
                    # In case we're doing the learnm branch
                    pred_src_z = net_out[0]
                else:
                    pred_src_z = net_out
                    
                pred_src_z = pred_src_z.data.cpu().numpy()[0]
                
                pearsons.append(
                    pearsonr(pred_src_z.flatten(), src_keypt_z.data.numpy().flatten())[0])
                l2_losses.append( (pred_src_z.flatten() - src_keypt_z.data.numpy().flatten())**2 )
                preds.append(pred_src_z.flatten())
                actuals.append(src_keypt_z.data.numpy().flatten())

                if map_ is not None:
                    map_[i][j] = pred_src_z.flatten()

        covar = compute_covar(preds, actuals)
        if map_ is not None:
            np.savez(dump_file, preds=map_)
        return {
            'pearsons': (np.mean(pearsons), np.std(pearsons)),
            'l2_losses': (np.mean(l2_losses), np.std(l2_losses)),
            'covar': covar
        }

    if mode == 'test':
        xy_keypts, z_keypts, orients = get_data_test()
    else:
        xy_keypts, z_keypts = get_data_valid()
        xy_keypts = xy_keypts[0:225]
        z_keypts = z_keypts[0:225]
    
    if not grid:

        all_all = fn(xy_keypts, z_keypts,
                     xy_keypts, z_keypts,
                     same=True, dump_file=dump_file)
        print( all_all['covar'] )

    else:

        if mode == 'valid':
            raise Exception("Cannot do left/center/right with valid set!")

        left = orients=='left'
        center = orients=='center'
        right = orients=='right'
        xy_keypts_left, z_keypts_left = xy_keypts[left], z_keypts[left]
        xy_keypts_center, z_keypts_center = xy_keypts[center], z_keypts[center]
        xy_keypts_right, z_keypts_right = xy_keypts[right], z_keypts[right]

        dd = dict()
        dd['left'] = {}
        dd['center'] = {}
        dd['right'] = {}
        dd['left']['xy'] = xy_keypts_left
        dd['left']['z'] = z_keypts_left
        dd['center']['xy'] = xy_keypts_center
        dd['center']['z'] = z_keypts_center
        dd['right']['xy'] = xy_keypts_right
        dd['right']['z'] = z_keypts_right

        results = {'left': {}, 'center':{}, 'right':{}}
        for dir1 in ['left', 'center', 'right']:
            for dir2 in ['left', 'center', 'right']:
                result = fn(dd[dir1]['xy'], dd[dir1]['z'],
                            dd[dir2]['xy'], dd[dir2]['z'],
                            same=dir1==dir2)
                results[dir1][dir2] = result

        print("\t%s\t\t%s\t\t%s" % ("left", "center", "right"))
        print("left\t%f\t%f\t%f" % (results['left']['left']['covar'],
                                    results['left']['center']['covar'],
                                    results['left']['right']['covar']))
        print("center\t%f\t%f\t%f" % (results['center']['left']['covar'],
                                      results['center']['center']['covar'],
                                      results['center']['right']['covar']))
        print("right\t%f\t%f\t%f" % (results['right']['left']['covar'],
                                     results['right']['center']['covar'],
                                     results['right']['right']['covar']))
    

def measure_kp_error_test_pairwise(net, grid=True, mode='test', use_gt_z=False):
    """
    Measure the keypoint error on the test set in a pairwise
      fashion.
    TODO: convert into batch version
    """

    net._eval()

    def fn(xy1_keypts, z1_keypts, xy2_keypts, z2_keypts, same=False):
        l2_losses = []
        for i in range(len(xy1_keypts)):
            for j in range(len(xy2_keypts)):
                if same and i==j:
                    continue
                xy_keypts_src = xy1_keypts[i][np.newaxis]
                z_keypts_src = z1_keypts[i][np.newaxis]
                xy_keypts_tgt = xy2_keypts[j][np.newaxis]
                z_keypts_tgt = z2_keypts[j][np.newaxis]
                losses, _ = net.run_on_instance(xy_keypts_src=xy_keypts_src,
                                                z_keypts_src=z_keypts_src,
                                                xy_keypts_tgt=xy_keypts_tgt,
                                                z_keypts_tgt=z_keypts_tgt,
                                                train=False,
                                                use_gt_z=use_gt_z)
                l2_losses.append(losses['l2'])
        return l2_losses

    if mode == 'test':
        xy_keypts, z_keypts, orients = get_data_test()
    else:
        xy_keypts, z_keypts = get_data_valid()
        xy_keypts = xy_keypts[0:225]
        z_keypts = z_keypts[0:225]
        
    if not grid:
        
        print("src: all, tgt: all")
        all_all = fn(xy_keypts, z_keypts,
                     xy_keypts, z_keypts, same=True)
        print(np.mean(all_all), np.std(all_all))

    else:

        if mode == 'valid':
            raise Exception("Cannot do left/center/right with valid set!")

        left = orients=='left'
        center = orients=='center'
        right = orients=='right'
        xy_keypts_left, z_keypts_left = xy_keypts[left], z_keypts[left]
        xy_keypts_center, z_keypts_center = xy_keypts[center], z_keypts[center]
        xy_keypts_right, z_keypts_right = xy_keypts[right], z_keypts[right]

        dd = dict()
        dd['left'] = {}
        dd['center'] = {}
        dd['right'] = {}
        dd['left']['xy'] = xy_keypts_left
        dd['left']['z'] = z_keypts_left
        dd['center']['xy'] = xy_keypts_center
        dd['center']['z'] = z_keypts_center
        dd['right']['xy'] = xy_keypts_right
        dd['right']['z'] = z_keypts_right
        
        for dir1 in ['left', 'center', 'right']:
            for dir2 in ['left', 'center', 'right']:
                print("src: %s, tgt: %s" % (dir1, dir2))
                result = fn(dd[dir1]['xy'], dd[dir1]['z'],
                            dd[dir2]['xy'], dd[dir2]['z'],
                            same=dir1==dir2)
                print(np.mean(result), " +/- ", np.std(result))
    
if __name__ == '__main__':
    pass
