import torch
from torch.autograd import Variable
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

def gkern(l=5, sig=1.):
    """
    Creates gaussian kernel with side length l and a sigma of sig.
    Acknowledgement: https://stackoverflow.com/users/6465762/clemisch
    """
    ax = np.arange(-l // 2 + 1., l // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sig**2))
    return kernel / np.sum(kernel)

def get_fm_for_xy(x,y):
    """
    Return a feature map corresponding to a keypt at
      location (x,y).
    """
    fm = np.zeros((128,128))
    gauss_len = 8
    gauss_std = 1 # 2 
    #x,y = 64, 64
    kern = gkern(l=gauss_len, sig=gauss_std)
    # The kernel may be bigger than the region
    # of the image it is applied to, so crop it
    # here if necessary.
    xh, xw = fm[y-(gauss_len//2):y+(gauss_len//2),
       x-(gauss_len//2):x+(gauss_len//2)].shape
    kern = kern[0:xh,0:xw]
    fm[y-(gauss_len//2):y+(gauss_len//2),
       x-(gauss_len//2):x+(gauss_len//2)] += kern
    return fm

def read_kpt_file(filename):
    """Return np array of keypts"""
    kpts = open(filename).read().split("\n")[0:-1]
    kpts = [ elem.split(",") for elem in kpts ]
    for entry in kpts:
        for i in range(3):
            entry[i] = float(entry[i])
    kpts = np.asarray(kpts)
    return kpts

def get_data_from_id(root, mode, id_):
    """
    Returns:
     - img_downsized: this is the image in 128px res.
     - y_keypts: the keypts in range [-1, 1]. To plot
       these, get them in [0,1], multiply by 128.,
       and overlay these on the img_downsized.
     - z_keypts: the z keypoints in unnormalised form.
     - x_keypts: the keypoints represented as a
       (n,128,128) input image using heatmap trick.
    """
    img = imread("%s/%s_img/%s.jpg" % (root,mode,id_))
    keypts = read_kpt_file("%s/%s_lm/%s_lm.csv" % (root,mode,id_))
    # We want the img + keypts in 128x128px img so preproc them
    # accordingly.
    img_downsized = resize(img, (128,128))
    # y_keypts are 2d keypts in [-1, 1], and
    # z_keypts are just the raw z's.
    #y_keypts = keypts[:,0:2] / float(img.shape[0])
    y_keypts = np.copy(keypts)[:,0:2]
    y_keypts[:,0] = y_keypts[:,0] / float(img.shape[1]) # x's
    y_keypts[:,1] = y_keypts[:,1] / float(img.shape[0]) # y's
    #y_keypts = (y_keypts - 0.5) / 0.5 # now in [-1, 1]
    avg_sz = (img.shape[0]+img.shape[1]) / 2.
    z_keypts = keypts[:,2] / avg_sz # what range??
    # x_keypts is the feature map version of the keypts.
    #x_keypts = []
    #keypts_uint8 = y_keypts * 128.
    #keypts_uint8 = keypts_uint8.astype("uint8")
    #for keypt_uint8 in keypts_uint8:
    #    x_keypts.append(get_fm_for_xy(keypt_uint8[0], keypt_uint8[1]))
    #x_keypts = np.asarray(x_keypts)
    return img_downsized, y_keypts, z_keypts

def construct_A(src_kps, src_z_pred, use_cuda=False):
    K = 66
    bs = src_kps.shape[0]
    A = np.zeros((bs, K*2, 8))
    for b in range(bs):
        c = 0
        for i in range(0, A.shape[1]-1, 2):
            A[b, i, 0] = src_kps[b, 0, c] # xi
            A[b, i, 1] = src_kps[b, 1, c] # yi
            #A[i,2] = z_pred[c] # zi
            A[b, i, -2] = 1.
            #
            A[b, i+1, 4] = src_kps[b, 0, c] # xi
            A[b, i+1, 5] =  src_kps[b, 1, c] # yi
            #A[i+1,6] = z_pred[c] # zi
            A[b, i+1, -1] = 1.
            c += 1
    # Ok, now turn it into a Variable and do
    # in-place ops on it which add the predicted
    # depths.
    A = torch.from_numpy(A).float()
    if use_cuda:
        A = A.cuda()
    for b in range(bs):
        c = 0
        for i in range(0, A.size(1)-1, 2):
            A[b, i, 2] = src_z_pred[b, 0, c] # zi
            A[b, i+1, 6] = src_z_pred[b, 0, c] # zi
            c += 1
    return A

def inv(x, use_cuda=False):
    # https://github.com/pytorch/pytorch/pull/1670
    eye = torch.eye(8).float()
    if use_cuda:
        eye = eye.cuda()
    return torch.inverse(eye+x)   

def predict_tgt_kp_pseudoinv(xy_keypt_src,
                             pred_src_z,
                             xy_keypt_tgt):
    """
    Given src keypts, predicted depths, and tgt keypts,
      construct a baseline estimate of the predicted 
      tgt keypoints through the pseudo-inverse (fixed m)
      formulation in the paper.
    xy_keypt_src: (bs, 66, 2) in numpy
    pred_src_z: (bs, 1, 66) in Torch
    xy_keypt_tgt: (bs, 66, 2) in numpy
    """
    # TODO
    assert xy_keypt_src.shape[0] == 1
    assert xy_keypt_tgt.shape[0] == 1
    # TODO
    A = construct_A(xy_keypt_src.swapaxes(1,2),
                    pred_src_z)
    tgt_kps_f = xy_keypt_tgt.swapaxes(1,2).reshape((1, 2*66), order='F')
    xt = torch.from_numpy(tgt_kps_f).float()
    X1 = [inv(mat) for mat in
          torch.matmul(A.transpose(2, 1), A)]
    X1 = torch.stack(X1)
    X2 = torch.bmm(A.transpose(2, 1), xt.unsqueeze(2))
    m = torch.bmm(X1, X2) # (bs,8,1)
    m_rshp = m.squeeze(2).view((1, 2, 4))
    ones = torch.ones((1, 1, 66)).float()
    xy_keypt_src_torch = torch.from_numpy(xy_keypt_src).float()
    xy_keypt_src_torch = xy_keypt_src_torch.transpose(1,2)
    rht = torch.cat((xy_keypt_src_torch,
                     pred_src_z,
                     ones), dim=1)
    rhs = torch.matmul(m_rshp, rht)
    return rhs
