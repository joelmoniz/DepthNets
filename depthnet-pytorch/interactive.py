import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import os
import shutil
from util import (compute_covar,
                  read_kpt_file,
                  convert_keypts_66_to_68,
                  convert_depth_66_to_68,
                  rot_matrix_x,
                  rot_matrix_y,
                  rot_matrix_z,
                  affine_matrix_x,
                  affine_matrix_y,
                  affine_matrix_z)
from skimage.io import imread, imsave
from skimage.transform import resize, rescale
from iterators.iterator import (get_data_valid,
                                get_data_train,
                                get_data_test)
import numpy as np
from tqdm import tqdm
from scipy.stats.stats import pearsonr


def measure_depth(net, grid=True, dump_file=None, mode='test'):
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
        print("DepthCorr:")
        print(all_all['covar'])

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

        print("DepthCorr:")
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
    

def measure_kp_error(net, grid=True, mode='test', use_gt_z=False):
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
        print("kp error:", np.mean(all_all), " +/- ", np.std(all_all))

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
                print("kp error:", np.mean(result), " +/- ", np.std(result))


def read_input(src_kpts_file,
               tgt_kpts_file,
               src_img_file,
               tgt_img_file,
               output_size,
               scale_depth,
               kpt_file_separator):
    
    # Read in the keypoint files.
    src_kpts_all = read_kpt_file(src_kpts_file,
                                 sep=kpt_file_separator)
    tgt_kpts_all = read_kpt_file(tgt_kpts_file,
                                 sep=kpt_file_separator)
    src_kpts = src_kpts_all[:, 0:2]
    if src_kpts_all.shape[1] > 2:
        src_depth = src_kpts_all[:, 2]
    else:
        src_depth = None
    tgt_kpts = tgt_kpts_all[:, 0:2]
    if tgt_kpts_all.shape[1] > 2:
        tgt_depth = tgt_kpts_all[:, 2]
    else:
        tgt_depth = None
    
    # If the images are tuple, then assume that this
    # denotes the (width, height), otherwise it's the
    # location of the image.
    
    src_img = imread(src_img_file)
    src_kpts[:, 0] /= float(src_img.shape[1]) # x's
    src_kpts[:, 1] /= float(src_img.shape[0]) # y's
    
    tgt_img = imread(tgt_img_file)
    tgt_kpts[:, 0] /= float(tgt_img.shape[1]) # x's
    tgt_kpts[:, 1] /= float(tgt_img.shape[0]) # y's

    if src_depth is not None and scale_depth is not None:
        if scale_depth < 0:
            raise Exception("scale_depth must be >= 0")
        elif scale_depth == 0:
            # do min-max normalisation
            this_min = np.min(src_depth)
            this_max = np.max(src_depth)
            src_depth = (src_depth - this_min) / (this_max - this_min)
        else:
            # scale by a constant
            src_depth /= scale_depth

    if tgt_depth is not None and scale_depth is not None:
        if scale_depth < 0:
            raise Exception("scale_depth must be >= 0")
        elif scale_depth == 0:
            # do min-max normalisation
            this_min = np.min(tgt_depth)
            this_max = np.max(tgt_depth)
            tgt_depth = (tgt_depth - this_min) / (this_max - this_min)
        else:
            # scale by a constant
            tgt_depth /= scale_depth
            
    if output_size == 'src':
        out_width = src_img.shape[1]
        out_height = src_img.shape[0]
    elif output_size == 'tgt':
        out_width = tgt_img.shape[1]
        out_height = tgt_img.shape[0]
    else:
        out_width = output_size
        out_height = output_size

    src_img_resized = resize(src_img, (out_height, out_width))
    tgt_img_resized = resize(tgt_img, (out_height, out_width))
        
    return {'src_kpts': src_kpts,
            'tgt_kpts': tgt_kpts,
            'src_depth': src_depth,
            'tgt_depth': tgt_depth,
            'src_img_resized': src_img_resized,
            'tgt_img_resized': tgt_img_resized,
            'out_width': out_width,
            'out_height': out_height}
        
def warp_to_rotated_target_face(net,
                                src_kpts_file,
                                tgt_kpts_file,
                                src_img_file,
                                tgt_img_file,
                                axis,
                                tgt_angle_1,
                                tgt_angle_2,
                                output_dir,
                                output_size='tgt',
                                scale_depth=-1,
                                norm_mean=0.,
                                norm_std=0.,
                                frame_multiplier=100.,
                                kpt_file_separator=',',
                                rotate_source=False):
    """
    Warp a source face to a continually rotated target face.
    For each rotation performed on the target face, we estimate
      depth / affine to map the source face to that particular
      target face.

    Parameters
    ----------
    net: the network
    src_kpts_file: path to the source keypoints file
    tgt_kpts_file: path to the target keypoints file. This
      file MUST contain depths.
    src_img_file: path to the source image
    tgt_img_file: path to the target image
    axis: which axis to rotate around ('x', 'y', or 'z')
    tgt_angle_1: the first destination target angle
      (in radians).
    tgt_angle_2: the final destination target angle
      (in radians)
    output_dir: output directory
    output_size: desired output size of the warp. This
      effectively scales (and re-saves) the source image,
      keypoints, and affine. It can either be an integer,
      or 'src' (for src width/height), or 'tgt' (for tgt
      width/height).
    scale_depth: the value by which we divide the target
      depths. (Obviously, this requires knowing the general
      scale of your keypoints, so the default value may or
      may not be appropriate.)
    norm_mean: shift the x/y coords of the target keypoints prior
      to rotating. After rotating, the result is de-normalised with
      this also.
    norm_std: scale the x/y coords of the target keypoints prior
      to rotating. After rotating, the result is de-normalised with
      this also.
    frame_multiplier: a multiplier for the number of rotations we do
      when we go from 0 to `tgt_angle_1` (or `tgt_angle_1` to `tgt_angle_2`).
    kpt_file_separator:
    rotate_source: if `True`, then we do not warp the source
      face to the target rotations. Instead, we rotate the
      source face directly by manually constructing an
      affine matrix which encodes the desired rotation
      (after inferring the depth of the source keypoints
      by using the target keypoints).
    """

    net._eval()

    folders = ['keypoints',
               'depth',
               'affine',
               'source',
               'expected_result',
               'plot_target',
               'plot_source']
    for folder in folders:
        if not os.path.exists("%s/%s" % (output_dir, folder)):
            os.makedirs("%s/%s" % (output_dir, folder))

    if rotate_source:
        print("NOTE: rotate_source enabled. This means that the " +
              "arguments `depth_const`, `norm_mean`, and `norm_std` " +
              "have no effect")

    dd = read_input(src_kpts_file=src_kpts_file,
                    tgt_kpts_file=tgt_kpts_file,
                    src_img_file=src_img_file,
                    tgt_img_file=tgt_img_file,
                    output_size=output_size,
                    scale_depth=scale_depth,
                    kpt_file_separator=kpt_file_separator)
    src_kpts = dd['src_kpts']
    tgt_kpts = dd['tgt_kpts']
    tgt_depth = dd['tgt_depth']
    src_img_resized = dd['src_img_resized']
    out_width = dd['out_width']
    out_height = dd['out_height']

    if tgt_depth is None:
        raise Exception("The depths of the target kpts must be known!")
    
    print("tgt_depth min max =", tgt_depth.min(), tgt_depth.max())

    fake_z = np.zeros((1, 66)).astype(src_kpts.dtype)
    
    # Let a 90 degree (0.78 rads) rotation correspond to 100 frames.
    # We want the # of frames to be proportional to how much we rotate. That is,
    # if we do a 180 degree rotation then it should use 200 frames.
    num_rotations_1 = int(abs(tgt_angle_1)*frame_multiplier)
    num_rotations_2 = int(frame_multiplier*abs(tgt_angle_2-tgt_angle_1))
    degs = np.hstack((np.linspace(0, tgt_angle_1, num=num_rotations_1),
                      np.linspace(tgt_angle_1, tgt_angle_2, num=num_rotations_2)))

    if axis == 'x':
        rot_matrix = rot_matrix_x
        rot_affine = affine_matrix_x
    elif axis == 'y':
        rot_matrix = rot_matrix_y
        rot_affine = affine_matrix_y
    elif axis == 'z':
        rot_matrix = rot_matrix_z
        rot_affine = affine_matrix_z

    imsave(arr=src_img_resized,
           fname="%s/source/src.png" % output_dir)

    for k in range(len(degs)):

        if not rotate_source:
        
            # Run the rotation matix on the target keypoints.
            result = np.dot(rot_matrix(degs[k]),
                            np.vstack(( (tgt_kpts.swapaxes(0, 1)-norm_mean)/norm_std, tgt_depth)))

            # Save plots of the target rotation
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            elev, azim = 30, 75
            ax.scatter(
                xs=result[0],
                zs=result[1],
                ys=tgt_depth,
                c=tgt_depth,
                linewidths=1.,
                edgecolors='black'
            )
            ax.view_init(elev, azim)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.invert_zaxis()
            #bounds = 3.0
            #ax.set_xlim(bounds, -bounds)
            #ax.set_ylim(bounds, -bounds)
            #ax.set_zlim(bounds, -bounds)
            fig.savefig("%s/plot_target/%06d.png" % (output_dir, k))
            plt.close(fig)

            tgt_warped = result.swapaxes(0, 1)[:, 0:2]*norm_std + norm_mean
            losses, outputs = net.run_on_instance(xy_keypts_src=src_kpts[np.newaxis],
                                                  z_keypts_src=fake_z,
                                                  xy_keypts_tgt=tgt_warped[np.newaxis],
                                                  z_keypts_tgt=fake_z,
                                                  train=False,
                                                  use_gt_z=False)
            pred_affine = outputs['affine'][0].cpu().numpy()
            pred_depths = outputs['src_z_pred'].cpu().numpy().flatten()
            
        else:
            
            losses, outputs = net.run_on_instance(xy_keypts_src=src_kpts[np.newaxis],
                                                  z_keypts_src=fake_z,
                                                  xy_keypts_tgt=tgt_kpts[np.newaxis],
                                                  z_keypts_tgt=fake_z,
                                                  train=False,
                                                  use_gt_z=False)
            pred_affine = torch.from_numpy(rot_affine(degs[k], norm_mean, norm_std))
            pred_depths = outputs['src_z_pred'].cpu().numpy().flatten() / 2.
            
        pred_affine[0, 2:] *= out_width
        pred_affine[1, 2:] *= out_height

        pred_depths_68 = convert_depth_66_to_68(pred_depths)

        src_kpts_68 = convert_keypts_66_to_68(src_kpts)
        src_kpts_68[:, 0] *= out_width # first col is x
        src_kpts_68[:, 1] *= out_height # second col is y
        src_kpts_68 = src_kpts_68.astype(np.int32)
        
        # Save plots of the source with inferred depth.
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        elev, azim = 30, 75
        ax.scatter(
            xs=src_kpts[:,0],
            zs=src_kpts[:,1],
            ys=pred_depths,
            c=pred_depths,
            linewidths=1.,
            edgecolors='black'
        )
        ax.view_init(elev, azim)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.invert_zaxis()
        ax.set_xlim(0.5, -0.5)
        ax.set_ylim(0.5, -0.5)
        ax.set_zlim(0.5, -0.5)
        fig.savefig("%s/plot_source/%06d.png" % (output_dir, k))
        plt.close(fig)
        
        # Save the keypoints.
        # TODO: HACKY, IT MAKES DUPLICATES
        with open("%s/keypoints/%06d.txt" % (output_dir, k), "w") as f:
            for tp in src_kpts_68:
                f.write("%i %i\n" % (tp[0], tp[1]))

        # Save the depths.
        with open("%s/depth/%06d.txt" % (output_dir, k), "w") as f:
            for elem in pred_depths_68:
                f.write("%f\n" % elem)

        # Save the affines.
        with open("%s/affine/%06d.txt" % (output_dir, k), "w") as f:
            for tp in pred_affine:
                f.write("%f %f %f %f\n" % (tp[0], tp[1], tp[2], tp[3]))        
        




'''
def warp_to_interpolated_target_face(net,
                                     src_kpts_file,
                                     tgt_kpts_file,
                                     src_img_file,
                                     tgt_img_file,
                                     output_dir,
                                     output_size=80.,
                                     kpt_file_separator=",",
                                     basename_prefix='src',
                                     num_frames=100):
    """
    """

    net._eval()

    folders = ['keypoints',
               'depth',
               'affine',
               'source',
               'expected_result',
               'tgt_images',
               'plots']
    for folder in folders:
        if not os.path.exists("%s/%s" % (output_dir, folder)):
            os.makedirs("%s/%s" % (output_dir, folder))

    if basename_prefix == 'src':
        basename = os.path.basename(src_img_file)
    else:
        basename = os.path.basename(tgt_img_file)
            
    # Read in the keypoint files.
    src_kpts_all = read_kpt_file(src_kpts_file, sep=kpt_file_separator)
    tgt_kpts_all = read_kpt_file(tgt_kpts_file, sep=kpt_file_separator)
    src_kpts = src_kpts_all[:, 0:2]
    tgt_kpts = tgt_kpts_all[:, 0:2]
    src_depth = src_kpts_all[:, 2]
    tgt_depth = tgt_kpts_all[:, 2]
    
    # If the images are tuple, then assume that this
    # denotes the (width, height), otherwise it's the
    # location of the image.
    
    src_img = imread(src_img_file)
    src_kpts[:, 0] /= float(src_img.shape[1]) # x's
    src_kpts[:, 1] /= float(src_img.shape[0]) # y's
    src_depth /= ((src_img.shape[0] + src_img.shape[1]) / 2.)
    
    tgt_img = imread(tgt_img_file)
    tgt_kpts[:, 0] /= float(tgt_img.shape[1]) # x's
    tgt_kpts[:, 1] /= float(tgt_img.shape[0]) # y's
    tgt_depth /= ((tgt_img.shape[0] + tgt_img.shape[1]) / 2.)

    if output_size == 'src':
        out_width = src_img.shape[1]
        out_height = src_img.shape[0]
    elif output_size == 'tgt':
        out_width = tgt_img.shape[1]
        out_height = tgt_img.shape[0]
    else:
        out_width = output_size
        out_height = output_size
    
    src_kpts_68 = convert_keypts_66_to_68(src_kpts)
    src_kpts_68[:, 0] *= out_width # first col is x
    src_kpts_68[:, 1] *= out_height # second col is y
    src_kpts_68 = src_kpts_68.astype(np.int32)
    
    fake_z = np.zeros((1, 66)).astype(src_kpts.dtype) # TODO: HACK

    print("Saving resized src image...")
    src_img_80 = resize(src_img, (out_height, out_width))
    imsave(arr=src_img_80,
           fname="%s/source/%s.png" % (output_dir, basename))

    print("Saving resized tgt image...")
    tgt_img_80 = resize(tgt_img, (out_height, out_width))
    imsave(arr=tgt_img_80,
           fname="%s/tgt_images/%s.png" % (output_dir, basename))


    for k, alpha in enumerate(np.linspace(0, 1, num_frames)):
        
        tgt_warped = (1.-alpha)*src_kpts + alpha*tgt_kpts
        losses, outputs = net.run_on_instance(xy_keypts_src=src_kpts[np.newaxis],
                                              z_keypts_src=fake_z,
                                              xy_keypts_tgt=tgt_warped[np.newaxis],
                                              z_keypts_tgt=fake_z,
                                              train=False,
                                              use_gt_z=False)
        pred_affine = outputs['affine'][0].cpu().numpy()
        pred_affine[0, 2:] *= out_width
        pred_affine[1, 2:] *= out_height

        pred_depths = outputs['src_z_pred'].cpu().numpy().flatten()
        pred_depths_68 = convert_depth_66_to_68(pred_depths)
       
        # Save the keypoints.
        # TODO: HACKY, IT MAKES DUPLICATES
        with open("%s/keypoints/%06d.txt" % (output_dir, k), "w") as f:
            for tp in src_kpts_68:
                f.write("%i %i\n" % (tp[0], tp[1]))

        # Save the depths.
        with open("%s/depth/%06d.txt" % (output_dir, k), "w") as f:
            for elem in pred_depths_68:
                f.write("%f\n" % elem)

        # Save the affines.
        with open("%s/affine/%06d.txt" % (output_dir, k), "w") as f:
            for tp in pred_affine:
                f.write("%f %f %f %f\n" % (tp[0], tp[1], tp[2], tp[3]))
'''








                
def warp_to_target_face(net,
                        src_kpts_file,
                        tgt_kpts_file,
                        src_img_file,
                        tgt_img_file,
                        output_dir,
                        output_size='tgt',
                        kpt_file_separator=",",
                        basename_prefix='src',
                        save_plots=False):
    """
    Warp a source face to a target face.

    Parameters
    ----------
    net: the network
    src_kpts_file: path to the source keypoints file
    tgt_kpts_file: path to the target keypoints file
    src_img_file: path to the source image
    tgt_img_file: path to the target image
    output_dir: output directory
    output_size: desired output size of the warp (this
      scales the source image, keypoints, and affine)
    kpt_file_separator:
    basename_prefix:
    plot_warp: only works if you have the depths!!!
    """

    net._eval()

    folders = ['keypoints',
               'depth',
               'affine',
               'source',
               'expected_result',
               'plots',
               'tgt_images',
               'plot_warp']
    
    for folder in folders:
        if not os.path.exists("%s/%s" % (output_dir, folder)):
            os.makedirs("%s/%s" % (output_dir, folder))

    if basename_prefix == 'src':
        basename = os.path.basename(src_img_file)
    else:
        basename = os.path.basename(tgt_img_file)

    dd = read_input(src_kpts_file=src_kpts_file,
                    tgt_kpts_file=tgt_kpts_file,
                    src_img_file=src_img_file,
                    tgt_img_file=tgt_img_file,
                    output_size=output_size,
                    scale_depth=None,
                    kpt_file_separator=kpt_file_separator)
    src_kpts = dd['src_kpts']
    tgt_kpts = dd['tgt_kpts']
    src_depth = dd['src_depth']
    if src_depth is None:
        src_depth = np.zeros((66,))
    tgt_depth = dd['tgt_depth']
    if tgt_depth is None:
        tgt_depth = np.zeros((66,))
    src_img_resized = dd['src_img_resized']
    tgt_img_resized = dd['tgt_img_resized']
    out_width = dd['out_width']
    out_height = dd['out_height']
    
    # If save_plots is set to True, then we will plot a 3d
    # linear interpolation between the source keypoints
    # and the target keypoints. This obviously assumes
    # that the keypoint file has depth included.
    if save_plots:
        for idx, alpha in enumerate(np.linspace(0, 1, num=100)):
            xy_interp = (1.-alpha)*src_kpts + alpha*tgt_kpts
            depth_interp = (1.-alpha)*src_depth + alpha*tgt_depth
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            elev, azim = 30, 85 # 30, 75
            ax.scatter(
                xs=xy_interp[:, 0],
                zs=xy_interp[:, 1],
                ys=depth_interp,
                c=depth_interp,
                linewidths=1.,
                edgecolors='black'
            )
            ax.view_init(elev, azim)
            ax.invert_xaxis()
            ax.invert_yaxis()
            ax.invert_zaxis()
            fig.savefig("%s/plot_warp/%06d.png" % (output_dir, idx))
            plt.close(fig)

    # Add batch axis.
    src_kpts_batch = src_kpts[np.newaxis]
    tgt_kpts_batch = tgt_kpts[np.newaxis]
    # HACK: this 'fake z' is just a placeholder and isn't
    # actually used in `run_on_instance`.
    fake_z = np.zeros((1, 66)).astype(src_kpts.dtype)
    
    losses, outputs = net.run_on_instance(xy_keypts_src=src_kpts_batch,
                                          z_keypts_src=fake_z,
                                          xy_keypts_tgt=tgt_kpts_batch,
                                          z_keypts_tgt=fake_z,
                                          train=False,
                                          use_gt_z=False)

    pred_depths = outputs['src_z_pred'].cpu().numpy().flatten()
    
    #print("pred depths shape", pred_depths.shape)
    #print("pred depths min / max", pred_depths.min(), pred_depths.max())
    pred_affine = outputs['affine'][0].cpu().numpy()

    #pred_affine[:, 2:] *= output_size # this will also modify outputs['affine']
    pred_affine[0, 2:] *= out_width
    pred_affine[1, 2:] *= out_height
    
    #print("pred affine:", pred_affine)

    #src_img_80 = resize(src_img, (out_height, out_width))
    imsave(arr=src_img_resized,
           fname="%s/source/%s.png" % (output_dir, basename))

    #tgt_img_80 = resize(tgt_img, (out_height, out_width))
    imsave(arr=tgt_img_resized,
           fname="%s/tgt_images/%s.png" % (output_dir, basename))

    if save_plots:
        # Plot the source kpts just in 2d.
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.scatter(src_kpts[:, 0],
                   src_kpts[:, 1],
                   color='black')
        ax.invert_yaxis()
        plt.savefig("%s/plots/src_kpts.png" % output_dir)
        plt.close(fig)
        # Plot the target kpts just in 2d.
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.scatter(tgt_kpts[:, 0],
                   tgt_kpts[:, 1],
                   color='black')
        ax.invert_yaxis()
        plt.savefig("%s/plots/tgt_kpts.png" % output_dir)
        plt.close(fig)
        # Plot the source 3d keypoints.
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        elev, azim = 30, 85
        ax.scatter(
            xs=src_kpts[:, 0],
            zs=src_kpts[:, 1],
            ys=src_depth,
            c=src_depth,
            linewidths=1.,
            edgecolors='black'
        )
        ax.view_init(elev, azim)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.invert_zaxis()
        fig.savefig("%s/plots/src_3d.png" % output_dir)
        plt.close(fig)
        # Plot the target 3d keypoints.
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        elev, azim = 30, 85
        ax.scatter(
            xs=tgt_kpts[:, 0],
            zs=tgt_kpts[:, 1],
            ys=tgt_depth,
            c=tgt_depth,
            linewidths=1.,
            edgecolors='black'
        )
        ax.view_init(elev, azim)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.invert_zaxis()
        fig.savefig("%s/plots/tgt_3d.png" % output_dir)
        plt.close(fig)
    
        # Plot the target kpts just in 2d.
        tgt_2d_pred = outputs['tgt_2d_pred']
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        ax.scatter(tgt_2d_pred[:, 0],
                   tgt_2d_pred[:, 1],
                   color='black')
        ax.invert_yaxis()
        plt.savefig("%s/plots/pred_kpts.png" % output_dir)
        plt.close(fig)

    src_kpts_68 = convert_keypts_66_to_68(src_kpts)
    src_kpts_68[:, 0] *= out_width # first col is x
    src_kpts_68[:, 1] *= out_height # second col is y
    src_kpts_68 = src_kpts_68.astype(np.int32)
    
    #src_kpts_68 = (convert_keypts_66_to_68(src_kpts)*output_size).astype(np.int32)
    pred_depths_68 = convert_depth_66_to_68(pred_depths)

    # Save the keypoints.
    with open("%s/keypoints/%s.txt" % (output_dir, basename), "w") as f:
        for tp in src_kpts_68:
            f.write("%i %i\n" % (tp[0], tp[1]))

    # Save the depths.
    with open("%s/depth/%s.txt" % (output_dir, basename), "w") as f:
        for elem in pred_depths_68:
            f.write("%f\n" % elem)

    # Save the affines.
    with open("%s/affine/%s.txt" % (output_dir, basename), "w") as f:
        for tp in pred_affine:
            f.write("%f %f %f %f\n" % (tp[0], tp[1], tp[2], tp[3]))
    
        
if __name__ == '__main__':
    pass
