"""
This module builds VGG dataset.
"""


# Param for bad warps
BAD_MIN_CONTOUR_DIST_RATIO = 0.5
# (difference of distance from each side's contour to nose/sum of distance from each side's contour to nose) 
# should be at least this value for image to be classified a bad warp

# Param for nearly frontal
FRONT_MAX_CONTOUR_DIST_RATIO = 0.18
FRONT_MAX_BRIDGE_TO_CHIN_DEV = 4 # max no. of pixels along y-axis that bridge should be away from chin
max_kpt_border_dif = 8 # the difference between distance of kpts #0 and #16 to their x border
# also chin kpt's distance to the border should be smaller than this threshold
max_chin_kpts_val = 76

import os
import numpy as np
import scipy.misc
from shutil import copyfile
from PIL import Image
from collections import OrderedDict
import h5py
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
    frontal_contour_nose_ratio = []

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

                if contour_nose_ratio > BAD_MIN_CONTOUR_DIST_RATIO and\
                    max_kpt_border_dif > abs((80 - kp[16, 0]) - kp[0, 0]) and \
                    max_kpt_border_dif > (80 - kp[8, 1]) and\
                    min(kp[7,1], kp[8,1], kp[9,1]) < max_chin_kpts_val:
                    distorted += [f_kp]
                    distorted_contour_nose_ratio += [contour_nose_ratio]
                elif FRONT_MAX_CONTOUR_DIST_RATIO > contour_nose_ratio and \
                    FRONT_MAX_BRIDGE_TO_CHIN_DEV > abs(kp[27, 0] - kp[8, 0]) and \
                    max_kpt_border_dif > abs((80 - kp[16, 0]) - kp[0, 0]) and \
                    max_kpt_border_dif > (80 - kp[8, 1]) and\
                    min(kp[7,1], kp[8,1], kp[9,1]) < max_chin_kpts_val:
                        frontal += [f_kp]
                        frontal_contour_nose_ratio += [contour_nose_ratio]
                c += 1
                #if c > 500:
                # return frontal, distorted, distorted_contour_nose_ratio, frontal_contour_nose_ratio

    return frontal, distorted, distorted_contour_nose_ratio, frontal_contour_nose_ratio


def make_dir(base_out_path):
    if not os.path.exists(base_out_path):
        os.makedirs(base_out_path)


def convert_np_to_PIL(np_img):
    img_rev = np_img[:, :, ::-1].copy()
    rescaled = (255.0 / img_rev.max() * (img_rev - img_rev.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    return im


def read_and_convert_to_cv(img_path):
    im = Image.open(img_path).convert('RGB')
    img = np.array(im)
    img_cv = img[:, :, ::-1].copy()
    return img_cv


def tile_images(img_orig, img_mask_bck, img_mask_face, img_frontalized_PIL,
                row_size, col_size):
    rows = 1 
    cols = 4

    gap_sz = 5
    gap_cols = (cols - 1) * gap_sz
    gap_rows = (rows - 1) * gap_sz
    index = 0

    new_im = Image.new('RGB', (cols*col_size + gap_cols,
                               rows*row_size + gap_rows), "white")
    for i in range(0, rows * row_size + gap_rows, row_size + gap_sz):
        for jj in range(0, cols * col_size + gap_cols, col_size + gap_sz):
            if jj == 0:
                new_im.paste(img_orig, (jj, i))
            elif jj == col_size + gap_sz:
                new_im.paste(img_mask_bck, (jj, i))
            elif jj == 2 * (col_size + gap_sz):
                new_im.paste(img_mask_face, (jj, i))
            else:
                new_im.paste(img_frontalized_PIL, (jj, i))
    return new_im


def plot_images(img_orig_cv, mask_bkg_cv, mask_face_cv, img_frontalized_cv,
                save_suffix):
    out_dir = '/storeSSD/jrantony/SiameseConvnetLasagne/vggnet_cropped_' +\
        'with_matlab_to_80x80/vis_test/VGG_dset_samples'
    row_size, col_size = 80, 80
    orig_img_PIL = convert_np_to_PIL(img_orig_cv)
    img_frontalized_PIL = convert_np_to_PIL(img_frontalized_cv)
    mask_bkg_PIL = convert_np_to_PIL(mask_bkg_cv)
    mask_face_PIL = convert_np_to_PIL(mask_face_cv)

    img_tiled = tile_images(orig_img_PIL, mask_bkg_PIL, mask_face_PIL,
                            img_frontalized_PIL, row_size, col_size)

    img_tiled.save('%s/%s.png' % (out_dir, save_suffix))

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

    row_size, col_size = 80, 80
    clean = lambda filename: filename.replace("./keypoints/", "").replace(".txt","")
    clean_save = lambda filename: filename.replace("./keypoints/", "").\
        replace(".txt","").replace('/', '_')
    criteria = lambda x: np.repeat((x == 0).reshape(row_size,col_size,1), 3, axis=2)

    print('getting frontal and non-frontal faces')
    [f, d, d_ratio, f_ratio] = get_frontal_and_distorted()

    src_GT = [] # source ground truth faces
    src_depthNet = [] # source faces frontalized with depthNet
    src_GT_mask_face = [] # source ground truth with masked faces
    src_id = [] # image id for source
    tg_GT = [] # target ground truth faces
    tg_id = [] # image id for target

    orig_img = "./images"
    frontalized_dir = "./depth_anaff_images"
    mask_bkg_dir = "./cropped_baseline_images"

    # taking frontal faces (tg_GT)
    print('making dset')
    counter = 0

    for line in f:
        # print clean(line) + "," + "frontal"
        orig_frontal = '%s/%s.png' % (orig_img, clean(line))
        img = read_and_convert_to_cv(orig_frontal)
        if img.shape != (row_size, col_size, 3):
            continue
        tg_GT.append(img)
        img_id = clean(line)
        tg_id.append(img_id)

    for line in d:
        #print clean(line) + "," + "warped"
        # taking non frontal GT faces (src_GT)
        orig_non_frontal = '%s/%s.png' % (orig_img, clean(line))
        img_orig = read_and_convert_to_cv(orig_non_frontal)

        # taking frontalized images by DepthNet (src_depthNet)
        frontalized = '%s/%s.png' % (frontalized_dir, clean(line))
        img_frontalized = read_and_convert_to_cv(frontalized)

        mask_bkg = '%s/%s.png' % (mask_bkg_dir, clean(line))
        mask_bkg = read_and_convert_to_cv(mask_bkg)
        # getting mask_face images

        # checking image resolution
        if img_orig.shape != (row_size, col_size, 3) or \
           img_frontalized.shape != (row_size, col_size, 3) or\
           mask_bkg.shape != (row_size, col_size, 3):
            continue

        # getting mask_face images
        sum_mask_bkg = np.sum(mask_bkg, axis=2)
        is_zero = criteria(sum_mask_bkg)
        mask_face = is_zero * img_orig

        #if counter < 100:
        #    plot_images(img_orig, mask_bkg, mask_face, img_frontalized,
        #                clean_save(line))
        #    counter += 1

        img_id = clean(line)
        src_GT.append(img_orig)
        src_GT_mask_face.append(mask_face)
        src_depthNet.append(img_frontalized)
        src_id.append(img_id)

    src_GT = np.array(src_GT).astype(np.uint8)
    src_GT_mask_face = np.array(src_GT_mask_face).astype(np.uint8)
    src_depthNet = np.array(src_depthNet).astype(np.uint8)
    src_id = np.array(src_id)
    tg_GT = np.array(tg_GT).astype(np.uint8)
    tg_id = np.array(tg_id)

    src_depthNet = do_sina_zoom(src_depthNet)[:, :, :, ::-1]
    src_GT_mask_face = src_GT_mask_face[:, :, :, ::-1]
    src_GT = src_GT[:, :, :, ::-1]
    tg_GT = tg_GT[:, :, :, ::-1]
    
    print('dumping data')
    
    h5f = h5py.File('vgg.h5', 'w')
    h5f.create_dataset('src_GT', data=src_GT)
    h5f.create_dataset('src_GT_mask_face', data=src_GT_mask_face)
    h5f.create_dataset('src_depthNet', data=src_depthNet)
    h5f.create_dataset('src_depthNet_and_mask',
                       data=np.concatenate((src_depthNet, src_GT_mask_face),
                                           axis=-1))
    h5f.create_dataset('tg_GT', data=tg_GT)
    h5f.create_dataset('src_id', data=src_id.astype("S50"))
    h5f.create_dataset('tg_id', data=tg_id.astype("S50"))
    h5f.close()

    print("total frontal is %s" % len(tg_id))
    print("total non_frontal is %s" % len(src_id))
