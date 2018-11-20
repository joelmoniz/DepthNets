import torch
from torch.autograd import Variable
import numpy as np
from skimage.io import (imread,
                        imsave)
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

def read_kpt_file(filename, sep=","):
    """Return np array of keypts"""
    kpts = open(filename).read().split("\n")[0:-1]
    kpts = [ elem.split(sep) for elem in kpts ]
    num_cols = len(kpts[0])
    for entry in kpts:
        for i in range(num_cols):
            entry[i] = float(entry[i])
    kpts = np.asarray(kpts)
    return kpts

def get_data_from_id(root, mode, id_):
    """
    Returns:
     - img_downsized: this is the image in 128px res.
     - y_keypts: the keypts in range [0, 1]. To plot
       these, multiply by 128., and overlay these on 
       img_downsized.
     - z_keypts: the z keypoints normalised.
    """
    img = imread("%s/%s_img/%s.jpg" % (root,mode,id_))
    keypts = read_kpt_file("%s/%s_lm/%s_lm.csv" % (root,mode,id_))
    # We want the img + keypts in 128x128px img so preproc them
    # accordingly.
    img_downsized = resize(img, (128,128))
    y_keypts = np.copy(keypts)[:,0:2]
    y_keypts[:,0] = y_keypts[:,0] / float(img.shape[1]) # x's
    y_keypts[:,1] = y_keypts[:,1] / float(img.shape[0]) # y's
    avg_sz = (img.shape[0]+img.shape[1]) / 2.
    z_keypts = keypts[:,2] / avg_sz # what range??
    return img_downsized, y_keypts, z_keypts

def construct_A(src_kps, src_z_pred):
    K = 66
    bs = src_kps.shape[0]
    # TODO: make more efficient
    A = np.zeros((bs, K*2, 8))
    for b in range(bs):
        c = 0
        for i in range(0, A.shape[1]-1, 2):
            A[b, i, 0] = src_kps[b, 0, c] # xi
            A[b, i, 1] = src_kps[b, 1, c] # yi
            #A[i,2] = z_pred[c] # zi
            A[b, i, -2] = 1.
            #
            A[b, i+1, 3] = src_kps[b, 0, c] # xi
            A[b, i+1, 4] =  src_kps[b, 1, c] # yi
            #A[i+1,6] = z_pred[c] # zi
            A[b, i+1, -1] = 1.
            c += 1
    A = torch.from_numpy(A).float()
    if src_z_pred.is_cuda:
        A = A.cuda()
    for b in range(bs):
        c = 0
        for i in range(0, A.size(1)-1, 2):
            A[b, i, 2] = src_z_pred[b, 0, c] # zi
            A[b, i+1, 5] = src_z_pred[b, 0, c] # zi
            c += 1
    return A

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
    X1 = [torch.inverse(mat) for mat in
          torch.matmul(A.transpose(2, 1), A)]
    X1 = torch.stack(X1)
    X2 = torch.bmm(A.transpose(2, 1), xt.unsqueeze(2))
    m = torch.bmm(X1, X2) # (bs,8,1)
    bs = xy_keypt_src.shape[0]
    m_rshp = torch.cat((m[:, 0:6, :].reshape((bs, 2, 3)),
                        m[:, [6, 7], :].reshape((bs, 2, 1))),
                       dim=2)
    ones = torch.ones((1, 1, 66)).float()
    xy_keypt_src_torch = torch.from_numpy(xy_keypt_src).float()
    xy_keypt_src_torch = xy_keypt_src_torch.transpose(1,2)
    rht = torch.cat((xy_keypt_src_torch,
                     pred_src_z,
                     ones), dim=1)
    rhs = torch.matmul(m_rshp, rht)
    return rhs

def convert_keypts_66_to_68(arr):
    kps_68 = np.zeros((68, 2))
    kps_68[0:60] = arr[0:60] # kpts 1 to 60 is kypts 1 to 60
    kps_68[60] = (arr[60-1]+arr[50-1]) / 2. # kpt 61 is the avg of kpts 60 and 50
    kps_68[61] = arr[60] # kpt 62 is keypt 61
    kps_68[62] = arr[61] # kpt 63 is keypt 62
    kps_68[63] = arr[62] # kpt 64 is keypt 63
    kps_68[64] = (arr[54-1] + arr[56-1]) / 2. # kpt 65 is the avg of kpts 54 and 56
    kps_68[65] = arr[63] # kpt 66 is keypt 64
    kps_68[66] = arr[64] # kpt 67 is keypt 65
    kps_68[67] = arr[65] # kpt 68 is keypt 66
    return kps_68

def convert_depth_66_to_68(arr):
    d_68 = np.zeros((68,))
    d_68[0:60] = arr[0:60] # kpts 1 to 60 is kypts 1 to 60
    d_68[60] = (arr[60-1]+arr[50-1]) / 2. # kpt 61 is the avg of kpts 60 and 50
    d_68[61] = arr[60] # kpt 62 is keypt 61
    d_68[62] = arr[61] # kpt 63 is keypt 62
    d_68[63] = arr[62] # kpt 64 is keypt 63
    d_68[64] = (arr[54-1] + arr[56-1]) / 2. # kpt 65 is the avg of kpts 54 and 56
    d_68[65] = arr[63] # kpt 66 is keypt 64
    d_68[66] = arr[64] # kpt 67 is keypt 65
    d_68[67] = arr[65] # kpt 68 is keypt 66
    return d_68

def shift_matrix(shift):
    mat = np.eye(4)
    mat[0,-1] = shift
    mat[1,-1] = shift
    return mat

def scale_matrix(scale):
    mat = np.eye(4)
    mat[0,0] = scale
    mat[1,1] = scale
    return mat

def rot_matrix_x(theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3,3)).astype(np.float32)
    mat[0, 0] = 1.
    mat[1, 1] = np.cos(theta)
    mat[1, 2] = -np.sin(theta)
    mat[2, 1] = np.sin(theta)
    mat[2, 2] = np.cos(theta)
    return mat


def rot_matrix_y(theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3,3)).astype(np.float32)
    mat[0, 0] = np.cos(theta)
    mat[0, 2] = np.sin(theta)
    mat[1, 1] = 1.
    mat[2, 0] = -np.sin(theta)
    mat[2, 2] = np.cos(theta)
    return mat

def rot_matrix_z(theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3,3)).astype(np.float32)
    mat[0, 0] = np.cos(theta)
    mat[0, 1] = -np.sin(theta)
    mat[1, 0] = np.sin(theta)
    mat[1, 1] = np.cos(theta)
    mat[2, 2] = 1.
    return mat

def affine_matrix_and_rotation(theta, mean, std, rot_mat):
    """Construct an affine matrix of a rotation
    about the y axis"""

    shift1 = shift_matrix(-mean)
    scale1 = scale_matrix(1.0 / std)

    shift2 = shift_matrix(mean)
    scale2 = scale_matrix(std)
        
    rot_3x3 = rot_mat(theta) # 3x3
    rot = np.eye(4)
    rot[0:3,0:3] = rot_3x3
    
    result = np.dot(np.dot(np.dot(np.dot(shift1,scale1),rot),scale2),shift2)

    affine = np.hstack( (result[0:2,0:3],
                      np.zeros((2,1)) ) )
    
    return affine

def affine_matrix_x(theta, mean, std):
    return affine_matrix_and_rotation(theta, mean, std,
                                      rot_matrix_x)

def affine_matrix_y(theta, mean, std):
    return affine_matrix_and_rotation(theta, mean, std,
                                      rot_matrix_y)

def affine_matrix_z(theta, mean, std):
    return affine_matrix_and_rotation(theta, mean, std,
                                      rot_matrix_z)          

def compute_covar(preds, actuals, n_kps=66):
    return np.sum(np.diag(np.abs(np.corrcoef(preds, actuals, rowvar=0)[0:n_kps,n_kps::])))
