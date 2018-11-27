## Instructions

1. Download the CelebA dataset from here: [img_align_celeba.zip](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing).
   Extract these images into the folder: 'CelebA/images'
2. Download these [files](https://mega.nz/#!kLZzXKgT!DZKWca3rERDYQfQ7KTcdMW1yqtoNaO6M5tylY39cVI8) and extract them in this directory
   such that you have the folders 'keypoints' and 'depth_anaff_frontal'.
3. Run `python crop_celeba.py`, which will create 80x80 + face-cropped
   images and dump them into the folder called 'images'.
4. Modify `run_facewarp.py` such that it points to where FaceWarperServer
   was compiled. After this, run:
   `python run_facewarp.py --results_dir=frontalized_faces`
   This will frontalize the faces from the 'images' folder using the
   metadata provided by the folders 'keypoints' and 'depth_anaff_frontal'.
5. Run `python make_celebA_dataset.py`, which will create an HDF5 file
   based on the images required for CycleGAN.
6. Finally, run `python mask_face_region_with_avail_kpts.py` which
   will append some extra data to the same HDF5 file.
