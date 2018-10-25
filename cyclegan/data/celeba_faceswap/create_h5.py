from skimage.io import imread
import numpy as np
import h5py
import glob

f = h5py.File('celeba_faceswap.h5', 'w')
imgs = []
for filename in glob.glob("extracted_images_pasted/*.png"):
    img = imread(filename)
    if len(img.shape) == 2:
        # If image is greyscale, add some channels
        img = np.asarray([img,img,img], dtype=img.dtype)
        img = img.swapaxes(0,1).swapaxes(1,2)
    elif len(img.shape) == 4:
        # If has alpha channel, remove it
        img = img[:, :, 0:3]
    imgs.append(img)

imgs = np.asarray(imgs)
f.create_dataset('imgs', data=imgs)
f.close()
