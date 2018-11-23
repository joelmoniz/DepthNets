"""
This module masks faces using kpts already detected 
"""

import numpy as np
import argparse
import cv2
#from RCN.preprocessing.tools import BGR2Gray
from PIL import Image
import h5py

def get_parsed_keypoints(path):
    with open(path) as f:
        x = f.read()
        y=x.split('\n')
        z=[[int(i) for i in k.split()] for k in y if k is not '']
    return np.array(z)


def read_kpts(kpts_dir, imgs_ids):
    kpts_list = []
    for img_id in imgs_ids:
        img_path = '%s/%s_crop.txt' % (kpts_dir, img_id)
        kpts = get_parsed_keypoints(img_path)
        kpts_list.append(kpts)
    return np.array(kpts_list)

def mask_out_face(imgs, pred_kpts):
    mask_imgs = []
    for img, kpts in zip(imgs, pred_kpts): 
        # mask_img = cv2.fillPoly(img, kpts)
        kpts = kpts.astype(np.int32)
        # reordering #1 to #17 kpts to form a polygon
        kpts_mask = np.concatenate((kpts[:17][::-1], kpts[17:27]), axis=0)
        img_mask = img.copy()
        #cv2.fillConvexPoly(img_mask, kpts_mask, 0)
        cv2.fillPoly(img_mask, kpts_mask.reshape(1,27,2), 0)
        mask_imgs.append(img_mask)

    return mask_imgs

def plot_cross(img, kpt, color, lnt=1):
    kpt = map(int, kpt)
    x, y = kpt
    cv2.line(img=img, pt1=(x-lnt, y-lnt), pt2=(x+lnt, y+lnt), color=color)
    cv2.line(img=img, pt1=(x-lnt, y+lnt), pt2=(x+lnt, y-lnt), color=color)
    return img

def draw_kpts(img, kpts, color):
    for kpt in kpts:
        x_i = int(kpt[0])
        y_i = int(kpt[1])
        img = plot_cross(img, kpt=(x_i, y_i), color=color)
    return img

def convert_np_to_PIL(np_img):
    img_rev = np_img[:, :, ::-1].copy()
    rescaled = (255.0 / img_rev.max() * (img_rev - img_rev.min())).astype(np.uint8)
    im = Image.fromarray(rescaled)
    return im

def tile_images(img, img_mask, img_depthNet, row_size, col_size):
    rows = 1
    cols = 3

    gap_sz = 5
    gap_cols = (cols - 1) * gap_sz
    gap_rows = (rows - 1) * gap_sz
    index = 0

    new_im = Image.new('RGB', (cols*col_size + gap_cols,
                               rows*row_size + gap_rows), "white")
    for i in xrange(0, rows * row_size + gap_rows, row_size + gap_sz):
        for jj in xrange(0, cols * col_size + gap_cols, col_size + gap_sz):
            if jj == 0:
                new_im.paste(img, (jj, i))
            elif jj == col_size + gap_sz:
                new_im.paste(img_mask, (jj, i))
            else:
                new_im.paste(img_depthNet, (jj, i))
    return new_im


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Getting keypoint prediction\
    #                                 using a trained model.')
    #parser.add_argument('--img_path', type=str, help='the complete path to the\
    #                    pickle file that contains pre-processed images',
    #                    required=True)

    #kpts_path = '/home/honari/libs/test_RCN/RCN/plotting/keypoints'
    kpts_path = "./keypoints"
    #args = parser.parse_args()
    #img_path = args.img_path
    imgs_path = 'celebA.h5'

    #fp = open(img_path, 'r')
    fp = h5py.File(imgs_path, 'a')
    #dset = pickle.load(fp)
    imgs = fp['src_GT']
    #imgs_depthNet = fp['src_depthNet']
    imgs_ids = fp['src_id'][:].astype("U6")

    print('getting kpts')
    #pred_kpts = get_kpts(imgs, path)
    pred_kpts = read_kpts(kpts_path, imgs_ids)

    print('getting masks')
    masked_face = mask_out_face(imgs, pred_kpts)

    """
    data_dict = OrderedDict()
    data_dict['img_orig'] = imgs
    data_dict['img_mask'] = masked_face
    pickle.dump('mask_faces.pickle', data_dict)
    """
    src_GT_mask_face = np.array(masked_face).astype(np.uint8)
    #img_path_out = img_path.split('.pickle')[0] + '_with_mask.pickle'
    #with open(img_path_out, 'wb') as fp:                                                        
    #    pickle.dump(dset, fp)
    fp.create_dataset('src_GT_mask_face', data=src_GT_mask_face)
    src_depthNet = fp['src_depthNet']
    fp.create_dataset('src_depthNet_and_mask',
                      data=np.concatenate((src_depthNet, src_GT_mask_face), axis=-1))

    '''
    print('plotting samples')
    n_sample = 50
    for img, img_mask, img_depthNet, img_id in \
        zip(imgs, masked_face, imgs_depthNet, np.arange(n_sample)):
        row_size, col_size, _ = img.shape
        img_PIL = convert_np_to_PIL(img)
        img_mask_PIL = convert_np_to_PIL(img_mask)
        img_depthNet_PIL = convert_np_to_PIL(img_depthNet)
        img_new = tile_images(img_PIL, img_mask_PIL, img_depthNet_PIL,
                              row_size, col_size)
        img_new.save('./sample_mask_imgs/img_%s.png' % (img_id))
    '''

    fp.close()

    print('done!')
