from i2i.cyclegan import CycleGAN
from task_launcher_bgsynth import image_dump_handler

def get_experiment_depthnet_bg_vs_frontal():
    
    from architectures import block9_a6b3
    
    #name = "cg_baseline_sina_newblock9_inorm_sinazoom"
    name = "experiment_depthnet_bg_vs_frontal"
    
    gen_atob_fn, disc_a_fn, gen_btoa_fn, disc_b_fn = \
        block9_a6b3.get_network()
    net = CycleGAN(
        gen_atob_fn=gen_atob_fn,
        disc_a_fn=disc_a_fn,
        gen_btoa_fn=gen_btoa_fn,
        disc_b_fn=disc_b_fn,
        opt_d_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
        opt_g_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
        handlers=[image_dump_handler("results/%s" % name)],
        use_cuda='detect'
    )
    net.load("models/%s/48.pkl.bak2" % name) # this corresponds to 101st epoch
    return net.g_atob

def get_experiment_gt_vs_frontal_gt():
    
    from architectures import block9_a3b3
    
    #name = "cg_hard-baseline2_sina_newblock9_inorm_sinazoom"
    name = "experiment_gt_vs_frontal_gt"

    gen_atob_fn, disc_a_fn, gen_btoa_fn, disc_b_fn = \
        block9_a3b3.get_network()
    
    net = CycleGAN(
        gen_atob_fn=gen_atob_fn,
        disc_a_fn=disc_a_fn,
        gen_btoa_fn=gen_btoa_fn,
        disc_b_fn=disc_b_fn,
        opt_d_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
        opt_g_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
        handlers=[image_dump_handler("results/%s" % name)],
        use_cuda='detect'
    )
    net.load("models/%s/12.pkl.bak4" % name)
    return net.g_atob

def get_experiment_depthnet_gt_vs_frontal():

    from architectures import block9_a6b3
    
    #name = "cg_baseline_sina_newblock9_inorm_sinazoom_fullbg"
    name = "experiment_depthnet_gt_vs_frontal"

    gen_atob_fn, disc_a_fn, gen_btoa_fn, disc_b_fn = \
        block9_a6b3.get_network()

    net = CycleGAN(
        gen_atob_fn=gen_atob_fn,
        disc_a_fn=disc_a_fn,
        gen_btoa_fn=gen_btoa_fn,
        disc_b_fn=disc_b_fn,
        opt_d_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
        opt_g_args={'lr': 0.0002, 'betas': (0.5, 0.999)},
        handlers=[image_dump_handler("results/%s" % name)],
        use_cuda='detect'
    )
    net.load("models/%s/77.pkl.bak" % name)
    return net.g_atob


def put(draw, x, y, fill):
    """Convenience function to draw thick points on an image"""
    for j in [-1,0]:
        for k in [-1,0]:
            draw.point((x+j,y+k), fill)
            draw.point((x+j,y+k), fill)
            draw.point((x+j,y+k), fill)
            draw.point((x+j,y+k), fill)

if __name__ == '__main__':
    fn1 = get_experiment_depthnet_bg_vs_frontal()
    fn2 = get_experiment_gt_vs_frontal_gt()
    fn3 = get_experiment_depthnet_gt_vs_frontal()

            
'''
#
# TODO: fix this
#
def produce_neat_vis(mode):
    # prepare loading the data
    pkl_file = os.environ['PKL_SINA']
    with open(pkl_file, 'rb') as f:
        dat = pickle.load(f, encoding='latin1')
    src_GT, src_GT_mask_face, src_keypts, src_depthNet = \
            dat['src_GT'], dat['src_GT_mask_face'], dat['src_kpts'], dat['src_depthNet']
    src_depthNet = do_sina_zoom(src_depthNet)
    
    # load the models 
    g1 = get_cg_baseline_sina_newblock9_inorm_sinazoom() # depthnet + bg-only <-> frontal, ~ 100 epochs
    g2 = get_cg_hardbaseline2_sina_newblock9_inorm_sinazoom() # gt <-> frontal gt, ~ 90 epochs
    g3 = get_cg_baseline_sina_newblock9_inorm_sinazoom_fullbg() # depthnet + gt <-> frontal, ~ 77 epochs
    
    g1.eval()
    g2.eval()
    g3.eval()

    # indices chosen by sina
    indices = [5,6,11,12,17,23,54,59,64,67,68,78,79,1006,1018,1311,1319,1324,1328,1329,1397,
               434,446,451,464,474,485,497,1508,1509,1514,1530,1544,1569,1571,1586]
    
    # prepare iterators
    batch_size = len(indices)

    norm_tensor = lambda x: ((x.swapaxes(3,2).swapaxes(2,1) / 255.) - 0.5) / 0.5
    
    X_depthNet = src_depthNet[indices, :, :, ::-1]
    X_gt = src_GT[indices, :, :, ::-1]
    X_bg = src_GT_mask_face[indices, :, :, ::-1]
    X_bg_and_depthnet = np.concatenate((X_depthNet, X_bg), axis=-1) # [warped2front face, bg of orig]
    X_bg_and_depthnet = norm_tensor(X_bg_and_depthnet)
    X_gt_and_depthnet = np.concatenate((X_depthNet, X_gt), axis=-1) # [warped2front face, gt orig]
    X_gt_and_depthnet = norm_tensor(X_gt_and_depthnet)

    bg_and_depthnet = TensorDataset(X_bg_and_depthnet)
    loader_bg_and_depthnet = DataLoader(bg_and_depthnet, batch_size=batch_size, shuffle=False)
    
    X_gt = norm_tensor(X_gt)
    gt = TensorDataset(X_gt)
    loader_gt = DataLoader(gt, batch_size=batch_size, shuffle=False)
    loader_gt_iter = iter(loader_gt)

    loader_gt_and_depthnet = DataLoader(X_gt_and_depthnet, batch_size=batch_size, shuffle=False)
    loader_gt_and_depthnet_iter = iter(loader_gt_and_depthnet)
    
    # prepare grid
    N = len(indices)
    shp = 80
    xb = iter(loader_bg_and_depthnet).next().float().cuda()
    xb_src_gt = loader_gt_iter.next().float().cuda()
    xb_src_gt_and_depthnet = loader_gt_and_depthnet_iter.next().float().cuda()
    this_g1 = g1(xb).data.cpu().numpy()
    this_g2 = g2(xb_src_gt).data.cpu().numpy()
    this_g3 = g3(xb_src_gt_and_depthnet).data.cpu().numpy()
    for i in range(this_g1.shape[0]):
        row = np.zeros((shp, 6*shp, 3))
        # get the source gt image and add an index to it for
        # identification purposes
        #src_gt_img = Image.fromarray(X_gt[i])
        #d = ImageDraw.Draw(src_gt_img)
        #d.text((0,0), str(indices[i]), fill=(0,255,0))
        #src_gt_img = np.asarray(src_gt_img) / 255.
        src_gt_img = X_gt[i] / 255.
        row[:, 0:shp, :] = src_gt_img #######
        # get the source gt image and overlay keypoints on it
        # instead of showing the ugly bg-only image
        src_gt_img_keypts = Image.fromarray(X_gt[i])
        draw = ImageDraw.Draw(src_gt_img_keypts)
        this_keypts = src_keypts[ indices[i] ]
        for kp_tuple in this_keypts:
            put(draw, kp_tuple[0], kp_tuple[1], 'red')
        src_gt_img_keypts = np.asarray(src_gt_img_keypts) / 255.
        row[:, shp:(shp*2), :] = src_gt_img_keypts #######
        row[:, (shp*2):(shp*3), :] = X_depthNet[i] / 255.
        row[:, (shp*3):(shp*4), :] = convert_to_rgb(this_g1[i])
        row[:, (shp*4):(shp*5), :] = convert_to_rgb(this_g2[i])
        row[:, (shp*5):(shp*6), :] = convert_to_rgb(this_g3[i])
        imsave(arr=row, fname="vis_neat/%i.png" % indices[i])
'''
