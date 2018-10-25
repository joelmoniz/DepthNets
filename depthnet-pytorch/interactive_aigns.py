import torch
from iterators.iterator import (get_data_valid,
                                get_data_train,
                                get_data_test)
import numpy as np
from tqdm import tqdm
from scipy.stats.stats import pearsonr
from aigns import params_to_3d
import util


def compute_covar(preds, actuals):
    return np.sum(np.diag(np.abs(np.corrcoef(preds, actuals, rowvar=0)[0:66,66::])))

def measure_depth_test(net):
    """
    Measure depth metrics on the test set in a pairwise
      fashion.
    """

    net._eval()

    def fn(xy_keypts, z_keypts):
        pearsons = []
        l2_losses = []
        preds = []
        for i in range(len(xy_keypts)):
            
            keypt_xy = torch.from_numpy(xy_keypts[i][np.newaxis]).float()
            keypt_z_torch = torch.from_numpy(z_keypts[i]).float()
            keypt_z = z_keypts[i]
            
            # X_keypts is the feature map version of the 2d keypts
            # y_keypts is the matrix form of the 2d keypts
            # z_keypts is the ground truth z
            X_keypts, _, _ = net.prepare_batch(
                keypt_xy, keypt_z_torch)
            
            pred_params = net.g(X_keypts)
            x_3d = params_to_3d(pred_params, net.use_cuda) #(bs,3,66)
            
            #pred_src_z = pred_src_z.data.cpu().numpy()[0]
            pred_src_z = x_3d[:,-1,:].data.cpu().numpy() # last ch is the z's
            
            # TODO: clean this shit up
            pearsons.append(
                pearsonr(pred_src_z.flatten(), keypt_z)[0])
            l2_losses.append( (pred_src_z.flatten() - keypt_z)**2 )
            preds.append(pred_src_z.flatten())
        #print("len of array: %i" % len(pearsons))
        covar = compute_covar(preds, z_keypts)
        return {
            'pearsons': (np.mean(pearsons), np.std(pearsons)),
            'l2_losses': (np.mean(l2_losses), np.std(l2_losses)),
            'covar': covar
        }

    xy_keypts, z_keypts, orients = get_data_test()
    left = orients=='left'
    center = orients=='center'
    right = orients=='right'
    xy_keypts_left, z_keypts_left = xy_keypts[left], z_keypts[left]
    xy_keypts_center, z_keypts_center = xy_keypts[center], z_keypts[center]
    xy_keypts_right, z_keypts_right = xy_keypts[right], z_keypts[right]

    all_ = fn(xy_keypts, z_keypts)
    print( "all = ", all_['covar'])
    #print( "depth l2 = ", all_['l2_losses'] )

    left = fn(xy_keypts_left, z_keypts_left)
    print( "left = ", left['covar'])
    #print( "depth l2 = ", left['l2_losses'] )
    
    center = fn(xy_keypts_center, z_keypts_center)
    print("center = ", center['covar'])
    #print( "depth l2 = ", center['l2_losses'] )
    
    right = fn(xy_keypts_right, z_keypts_right)
    print("right = ", right['covar'])
    #print( "depth l2 = ", right['l2_losses'] )

    
def measure_kp_error_test(net, grid=True):
    """
    Measure the keypoint error on the test set.
    """

    net._eval()

    def fn(xy1_keypts, z1_keypts, xy2_keypts, z2_keypts, same=False):

        # First we're going to take the src and dest stuff,
        # batch it all together, and then run the AIGN model
        # to get the predicted outputs.
        xy_keypts_src_torch = torch.from_numpy(xy1_keypts).float()
        z_keypts_src_torch = torch.from_numpy(z1_keypts).float()
        X_keypts, _, _ = net.prepare_batch(xy_keypts_src_torch,
                                           z_keypts_src_torch)
        pred_params = net.g(X_keypts)
        x_3d = params_to_3d(pred_params, net.use_cuda)
        # This is the batch of predicted z values for
        # source.
        pred_src_zs = x_3d[:,-1,:].unsqueeze(1).cpu()
                
        l2_losses = []
        for i in range(len(xy1_keypts)):
            for j in range(len(xy2_keypts)):
                if same and i==j:
                    continue

                # Prepare the source and target keypts.
                xy_keypt_src = xy1_keypts[i][np.newaxis]
                xy_keypt_tgt = xy2_keypts[j][np.newaxis]
                xy_keypt_src_torch = torch.from_numpy(xy_keypt_src).transpose(1,2)
                xy_keypt_tgt_torch = torch.from_numpy(xy_keypt_tgt).transpose(1,2)
                # Extract the i'th z value from pred_src_zs.
                pred_src_z = pred_src_zs[i].unsqueeze(1)

                rhs = util.predict_tgt_kp_pseudoinv(xy_keypt_src,
                                                    pred_src_z,
                                                    xy_keypt_tgt)
                l2_loss = torch.mean((xy_keypt_tgt_torch - rhs)**2)
                l2_losses.append(l2_loss.data.item())
            
        return l2_losses

    xy_keypts, z_keypts, orients = get_data_test()
    left = orients=='left'
    center = orients=='center'
    right = orients=='right'
    xy_keypts_left, z_keypts_left = xy_keypts[left], z_keypts[left]
    xy_keypts_center, z_keypts_center = xy_keypts[center], z_keypts[center]
    xy_keypts_right, z_keypts_right = xy_keypts[right], z_keypts[right]

    if not grid:
    
        # all, all
        print("src: all, tgt: all")
        all_all = fn(xy_keypts, z_keypts,
                     xy_keypts, z_keypts, same=True)
        print(np.mean(all_all), np.std(all_all))

    else:

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
