"""
This module builds celebA dataset
"""

# Param for bad warps
BAD_MIN_CONTOUR_DIST_RATIO = 0.5
# (difference of distance from each side's contour to nose/sum of distance from each side's contour to nose)
# should be at least this value for image to be classified a bad warp

# Param for nearly frontal
FRONT_MAX_CONTOUR_DIST_RATIO = 0.2 # default was 0.1
FRONT_MAX_BRIDGE_TO_CHIN_DEV = 10 # max no. of pixels along y-axis that bridge should be away from chin. default was 5

import os
import numpy as np
#import cv2
import scipy.misc
from shutil import copyfile
from scipy import misc
from collections import OrderedDict
#import pickle
import h5py
from PIL import Image
from skimage.transform import resize

#IMAGE_FOLDER = './images/'
#IMAGE_FLIPPED_FOLDER = './images_flipped/'
KEYPOINTS_FOLDER = './keypoints/'

# mean square distance
def msd(p1, p2):
    return np.mean(np.sum((p1-p2)**2, axis=1))

def create_dir_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_contour_to_nose_dists(kp):
    dist1 = msd(kp[[16, 15, 14, 13], :], kp[[28, 30, 33, 62], :])
    dist2 = msd(kp[[0, 1, 2, 3], :], kp[[28, 30, 33, 62], :])
    return [dist1, dist2]

# returns a set of images that are front-facing, and another set that are profile and likely to get
# distorted
def get_frontal_and_distorted():
    frontal = []
    distorted = []
    distorted_contour_nose_ratio = []

    c = 0
    for (dp, dn, fn) in os.walk(KEYPOINTS_FOLDER):
        if dn == []:

            for f in fn:
                f_kp = dp + '/' + f

                assert os.path.exists(f_kp)

                with open(f_kp) as f:
                    kp = f.read()
                kp = kp.split('\n')
                kp = [x.split() for x in kp if x is not '']
                kp = np.asarray(kp, dtype='int32')
                assert kp.shape == (68, 2)

                [dist1, dist2] = get_contour_to_nose_dists(kp)
                contour_nose_ratio = abs(dist1-dist2)/(dist1+dist2)

                if contour_nose_ratio > BAD_MIN_CONTOUR_DIST_RATIO:
                    distorted += [f_kp]
                    distorted_contour_nose_ratio += [contour_nose_ratio]
                elif FRONT_MAX_CONTOUR_DIST_RATIO > contour_nose_ratio and FRONT_MAX_BRIDGE_TO_CHIN_DEV > abs(kp[27, 0] - kp[27, 1]):
                    frontal += [f_kp]
                    
                c += 1
                print(c)
                #if c > 100:
                #    return frontal, distorted, distorted_contour_nose_ratio

    return frontal, distorted, distorted_contour_nose_ratio

def read_and_convert_to_cv(img_path):
    im = Image.open(img_path).convert('RGB')
    img = np.array(im)
    img_cv = img[:, :, ::-1].copy()
    return img_cv

def do_sina_zoom(X_buf):
    sp = {'top':17, 'bottom':3, 'left':10, 'right':10}
    new_buf = np.zeros_like(X_buf)
    for i in range(X_buf.shape[0]):
        this_img_60 = (resize(X_buf[i], (60, 60))*255.).astype("uint8")
        # ok, now slap it on a canvas
        this_canvas = np.zeros_like(X_buf[i])
        this_canvas[ sp['top']:(sp['top']+60), sp['left']:sp['left']+60, : ] = this_img_60
        new_buf[i] = this_canvas
    return new_buf

if __name__ == '__main__':

    clean = lambda filename: filename.replace(KEYPOINTS_FOLDER, "").replace(".txt","")
    clean_id = lambda filename: filename.replace(KEYPOINTS_FOLDER, "").\
        replace(".txt","").split('/')[1].split('_')[0]

    # generated_flipped_images()
    print('getting frontal and non-frontal faces')
    [f, d, d_ratio] = get_frontal_and_distorted()

    src_GT = [] # source ground truth faces
    src_depthNet = [] # source faces frontalized with depthNet
    src_id = [] # image id for source
    tg_GT = [] # target ground truth faces
    tg_id = [] # image id for target

    # taking frontal faces (tg_GT)
    print('making dset')
    for line in f:
        # print clean(line) + "," + "frontal"
        src = './images/%s.png' % clean(line)
        if not os.path.exists(src):
            print("Cannot find: %s" % src)
            continue
        img = read_and_convert_to_cv(src)
        tg_GT.append(img)
        img_id = clean_id(line)
        tg_id.append(img_id)

    for line in d:
        #print clean(line) + "," + "warped"
        # taking non frontal GT faces (src_GT)
        src = './images/%s.png' % clean(line)
        if not os.path.exists(src):
            print("Cannot find: %s" % src)
            continue
        img = read_and_convert_to_cv(src)
        src_GT.append(img)

        # taking frontalized images by DepthNet (src_depthNet)
        src = './frontalized_images/%s.png' % clean(line)
        img = read_and_convert_to_cv(src)
        src_depthNet.append(img)

        img_id = clean_id(line)
        src_id.append(img_id)

    src_depthNet = np.array(src_depthNet).astype(np.uint8)
    src_GT = np.array(src_GT).astype(np.uint8)
    src_id = np.array(src_id)
    tg_GT = np.array(tg_GT).astype(np.uint8)
    tg_id = np.array(tg_id)

    # We zoom out the depthnet faces just a little bit with
    # a heuristic. We also flip the BGR to that it's RGB.
    
    src_depthNet = do_sina_zoom(src_depthNet)[:, :, :, ::-1]
    src_GT = src_GT[:, :, :, ::-1]
    tg_GT = tg_GT[:, :, :, ::-1]

    print('dumping data')    
    h5f = h5py.File('celebA.h5', 'w')
    h5f.create_dataset('src_GT', data=src_GT)
    h5f.create_dataset('src_depthNet', data=src_depthNet)
    h5f.create_dataset('tg_GT', data=tg_GT)
    h5f.create_dataset('src_id', data=src_id.astype("S6"))
    h5f.create_dataset('tg_id', data=tg_id.astype("S6"))
    h5f.close()

    print("total frontal is %s" % len(f))
    print("total non_frontal is %s" % len(d))

    print('done')
    import pdb; pdb.set_trace()
