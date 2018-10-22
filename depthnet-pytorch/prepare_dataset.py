from util import get_data_from_id
import glob
import os
import numpy as np
from skimage.io import imread, imsave

root_dir = "/data/milatmp1/beckhamc/tmp_data/3dfaw"

def test():
    img_downsized, y_keypts, z_keypts, x_keypts = get_data_from_id(
        root=root_dir, id_=ids[0])
    for i in range(len(y_keypts)):
        x_kp, y_kp = int(y_keypts[i][0]*128), int(y_keypts[i][1]*128)
        img_downsized[y_kp, x_kp] += 1.0
    img_downsized = np.clip(img_downsized, 0, 1)
    imsave(arr=img_downsized, fname="tmp/face.png")

    fm = np.zeros((128,128))
    for i in range(len(x_keypts)):
        fm += x_keypts[i]
    imsave(arr=fm, fname="tmp/fm.png")
    print(x_keypts.sum())

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
    

#prepare_train()
prepare_valid()
prepare_test()
