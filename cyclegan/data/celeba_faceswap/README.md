### Instructions

1. Create symbolic links to the images and keypoint dirs of the
   CelebA folder:
   `ln -s ../celeba/keypoints keypoints`
   `ln -s ../celeba/images images`
2. Download this [file](https://mega.nz/#!8aI3WIqA!jfPmpgjS64krX-uXJLkP1EYE-Sp6WgiE167kmtZz83Y) and extract 'swap_samples' into this folder.
3. Modify `run_multiwarper.py` such that it points to where FaceWarperServer
   was compiled. After this, run `python run_multiwarper.py`. This will
   dump warped faces to the folder 'extracted_images'.
4. Run `paste_faces.py`, which will create a folder called
   `extracted_images_pasted`.
