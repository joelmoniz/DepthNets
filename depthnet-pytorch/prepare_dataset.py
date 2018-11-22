from util import (get_data_from_id,
                  read_kpt_file)
import glob
import os
import numpy as np
from skimage.io import (imread,
                        imsave)
from skimage.transform import resize

root_dir = os.environ['DIR_3DFAW']

def prepare_train():
    ids = glob.glob("%s/train_img/*.jpg" % root_dir)
    ids = [os.path.basename(id_).replace(".jpg","") for id_ in ids ]
    y_keypts, z_keypts = get_keypts_from_ids(ids, "train")
    np.savez(file="%s/train" % root_dir,
             y_keypts=y_keypts,
             z_keypts=z_keypts)

def get_keypts_from_ids(ids, mode):
    y_keypts = []
    z_keypts = []
    x_keypts = []
    meta = []
    for k, id_ in enumerate(ids):
        print("%i / %i" % (k, len(ids)))
        _,b,c = get_data_from_id(root=root_dir, mode=mode, id_=id_)
        # a is f64, let's make it uint8 to save some space.
        #a = (a*256.).astype("uint8")
        #imgs.append(a)
        y_keypts.append(b.astype("float32"))
        z_keypts.append(c.astype("float32"))
    #imgs = np.asarray(imgs)
    y_keypts = np.asarray(y_keypts)
    z_keypts = np.asarray(z_keypts)
    return y_keypts, z_keypts

def prepare_valid():
    ids = []
    with open("%s/list_valid_test.txt" % root_dir) as f:
        for line in f:
            line = line.rstrip().split(",")
            if line[1] == "valid":
                ids.append(line[0])
    y_keypts, z_keypts = get_keypts_from_ids(ids, "valid")
    np.savez(file="%s/valid" % root_dir,
             y_keypts=y_keypts,
             z_keypts=z_keypts,
             ids=ids)
    
def prepare_test():
    ids = []
    orientations = []
    with open("%s/list_valid_test.txt" % root_dir) as f:
        for line in f:
            line = line.rstrip().split(",")
            if line[1] == "test":
                ids.append(line[0])
                orientations.append(line[2])
    y_keypts, z_keypts = get_keypts_from_ids(ids, "valid") # yes, valid
    np.savez(file="%s/test" % root_dir,
             y_keypts=y_keypts,
             z_keypts=z_keypts,
             ids=ids,
             orientations=orientations)

def prepare_valid_imgs_downsized():
    ids = glob.glob("%s/valid_img/*.jpg" % root_dir)
    ids = [os.path.basename(id_).replace(".jpg","") for id_ in ids]
    output_folder = "%s/valid_img_cropped_80x80" % root_dir
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for id_ in ids:
        kpts = read_kpt_file("%s/valid_lm/%s_lm.csv" % (root_dir, id_))
        img = imread("%s/valid_img/%s.jpg" % (root_dir, id_))
        img = img[ int(np.min(kpts[:,1])):int(np.max(kpts[:,1])),
                   int(np.min(kpts[:,0])):int(np.max(kpts[:,0]))]
        img = resize(img, (80, 80))
        imsave(arr=img, fname="%s/%s.jpg" % (output_folder, id_))
    
if __name__ == '__main__':
    prepare_train()
    prepare_valid()
    prepare_test()
    prepare_valid_imgs_downsized()
