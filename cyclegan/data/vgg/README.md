### Instructions

1. Modify `run_multiwarper.py` such that it points to where FaceWarperServer
   was compiled.
2. Download these [files](https://mega.nz/#!AORlAYxJ!N2V_YjOtK8U5lLaIKAlTGJJvpfQYy7sQwjD3T_CPpmU) and extract them here such that you have these
   folders: `images`, `depth_anaff_frontal`, and `keypoints`.
3. Run `python run_facewarp.py --results_dir=depth_anaff_images` and 
   `python run_facewarp.py --identity --results_dir=cropped_baseline_images`.
4. Run `make_VGG_dataset_sina.py`, which will generate `vgg.h5`. Done!!!
