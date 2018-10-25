from skimage.io import imread, imsave
import numpy as np
import glob
import os
import sys

dest_folder = "extracted_images_pasted"
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

for filename in glob.glob("extracted_images/*.png"):
    print("Processing: %s" % filename)
    src_file = os.path.basename(filename).split("_")[0] + "_crop.png"
    tgt_file = os.path.basename(filename).split("_")[1].replace(".png","_crop.png")
    #print(src_file, tgt_file)
    src_img = imread(os.path.join("images", src_file))
    tgt_img = imread(os.path.join("images", tgt_file))
    normed_img = imread(filename)[:,:,0:3]

    final_img = np.copy(tgt_img)
    for i in range(final_img.shape[0]):
        for j in range(final_img.shape[1]):
            if normed_img[i,j].tolist() != [0,0,0]:
                final_img[i,j] = normed_img[i,j]

    imsave(arr=final_img,
           fname="%s/%s" % (dest_folder, os.path.basename(filename)))
    
