import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'jet'
#plt.rcParams['image.cmap'] = 'rainbow'
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import numpy as np
import argparse
from skimage.io import imread

if "DIR_3DFAW" not in os.environ:
    raise Exception("DIR_3DFAW env variable not found -- source env.sh")
DATA_DIR = os.environ["DIR_3DFAW"]

def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--depthnet_npz', type=str,
                        default="../tmp/depthnet_no_gan.npz")
    parser.add_argument('--depthnet_gan_npz', type=str,
                        default="../tmp/depthnet_with_gan.npz")
    parser.add_argument('--aign_npz', type=str,
                        default="../tmp/aigns.npz")
    args = parser.parse_args()
    return args

args = parse_args()

# Test set.
TEST_SET = "%s/test.npz" % DATA_DIR
test_set = np.load(TEST_SET)
test_xy = test_set['y_keypts'].swapaxes(1,2)
test_z = test_set['z_keypts']
test_set_ids = test_set['ids']

# DepthNet.
dn_preds = []
for filename in args.depthnet_npz.split(","):
    dn = np.load(filename)['preds']
    dn_preds.append(dn)
dn_preds = sum(dn_preds) / len(args.depthnet_npz.split(","))

# DepthNet with GAN.
dn_gan_preds = []
for filename in args.depthnet_gan_npz.split(","):
    dn_gan = np.load(filename)['preds']
    dn_gan_preds.append(dn_gan)
dn_gan_preds = sum(dn_gan_preds) / len(args.depthnet_gan_npz.split(","))

# AIGNs.
aign_preds = []
for filename in args.aign_npz.split(","):
    aign = np.load(filename)['preds']
    aign_preds.append(aign)
aign_preds = sum(aign_preds) / len(args.aign_npz.split(","))

# MOFA.
MOFA = "%s/mofa.npz" % DATA_DIR
mofa_dat = np.load(MOFA)
mofa_ids = mofa_dat['ids']
mofa_ids = [ x.decode('utf-8') for x in mofa_ids ]
mofa_z = mofa_dat['kps'][:,-1,:]
mofa_xy = mofa_dat['kps'][:,0:2,:]

IMAGE_DIR = "%s/valid_img_cropped_80x80" % DATA_DIR

def search_list(list_, query):
    for i in range(len(list_)):
        if list_[i] == query:
            return i
    return None

def vis(c, rnd_seed=0, out_file=None):
    rnd_state = np.random.RandomState(rnd_seed)
    mofa_c = search_list(mofa_ids, test_set_ids[c])
    if mofa_c is None:
        print("This index was not found in MOFA, returning...")
        return
    elev, azim = 30, 75 # 30,45 was old config
    # Show the image.
    fig = plt.figure(figsize=(6*5,5))
    ax = fig.add_subplot(161)
    img = imread("%s/%s.jpg" % (IMAGE_DIR, test_set_ids[c]))
    #img = np.zeros((80,80,3))
    ax.imshow(img)
    ax.axis('off')
    # Show the GT 3D.
    ax = fig.add_subplot(162, projection='3d')
    ax.scatter(
        xs=test_xy[c][0],
        zs=test_xy[c][1],
        ys=test_z[c],
        c=test_z[c],
        linewidths=1.,
        edgecolors='black'
    )
    ax.view_init(elev,azim)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.set_title("Ground truth")
    # Show the DepthNet NOGAN
    ax = fig.add_subplot(163, projection='3d')
    k = rnd_state.randint(0, len(test_z))
    ax.scatter(
        xs=test_xy[c][0],
        zs=test_xy[c][1],
        ys=dn_preds[c][k],
        c=dn_preds[c][k],
        linewidths=1.,
        edgecolors='black'
    )
    ax.view_init(elev,azim)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.set_title("DepthNet")
    # Show the DepthNet WITH GAN
    ax = fig.add_subplot(164, projection='3d')
    ax.scatter(
        xs=test_xy[c][0],
        zs=test_xy[c][1],
        ys=dn_gan_preds[c][k],
        c=dn_gan_preds[c][k],
        linewidths=1.,
        edgecolors='black'
    )
    ax.view_init(elev,azim)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.set_title("DepthNet + GAN")
    # Show the AIGN
    ax = fig.add_subplot(165, projection='3d')
    ax.scatter(
        xs=test_xy[c][0],
        zs=test_xy[c][1],
        ys=aign_preds[c],
        c=aign_preds[c],
        linewidths=1.,
        edgecolors='black'
    )
    ax.view_init(elev,azim)
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.invert_zaxis()
    ax.set_title("AIGN")
    ###################
    # Show the MOFA model
    ax = fig.add_subplot(166, projection='3d')
    # NOTE: the xy of MOFA and test
    # are highly correlated
    ax.scatter(
        #test_xy[c][0],
        #test_xy[c][1],
        xs=mofa_xy[mofa_c,0],
        zs=mofa_xy[mofa_c,1],
        ys=-mofa_z[mofa_c],
        c=-mofa_z[mofa_c],
        linewidths=1.,
        edgecolors='black'
    )
    ax.view_init(elev,azim)
    #ax.invert_xaxis()
    ax.invert_yaxis()
    #ax.invert_yaxis()
    #ax.invert_zaxis()
    ax.set_title("MOFA")
    if out_file is not None:
        fig.savefig(out_file, bbox_inches='tight', pad_inches=0.2)
    else:
        plt.show()
    plt.close(fig)

out_folder = "output"
if not os.path.exists(out_folder):
    os.makedirs(out_folder)

###########################
# Generate the depth rows #
###########################

for i in range(len(test_z)):
    print(i)
    vis(i, out_file="%s/%i.png" % (out_folder, i))

#########################
# Generate the heatmaps #
#########################

# GAN
x = []
y = []
for i in range(dn_preds.shape[0]):
    for j in range(dn_preds.shape[1]):
        x.append(test_z[i])
        y.append(dn_preds[i][j])
        
plt.hexbin(
    x=np.asarray(x).flatten(),
    y=np.asarray(y).flatten()
)
plt.xlabel("actual z")
plt.ylabel("pred z")
plt.title('DepthNet')
plt.ylim(ymin=-3, ymax=4)
plt.xlim(xmin=-0.2, xmax=0.2)
plt.tight_layout()
plt.savefig("%s/heatmap_dn.png" % out_folder)
# GAN + DEPTHNET
x = []
y = []
for i in range(dn_preds.shape[0]):
    for j in range(dn_preds.shape[1]):
        x.append(test_z[i])
        y.append(dn_gan_preds[i][j])
plt.hexbin(
    x=np.asarray(x).flatten(),
    y=np.asarray(y).flatten()
)
plt.xlabel("actual z")
plt.ylabel("pred z")
plt.title('DepthNet + GAN')
plt.ylim(ymin=np.asarray(y).min(), ymax=0.4)
plt.xlim(xmin=-0.2, xmax=0.2)
plt.tight_layout()
plt.savefig("%s/heatmap_dn_gan.png" % out_folder)
# AIGN
plt.hexbin(test_z.flatten(), aign_preds.flatten())
plt.xlabel("actual z")
plt.ylabel("pred z")
plt.title('AIGN')
#plt.ylim(ymin=aign_preds.min(), ymax=2)
#plt.xlim(xmin=-0.2, xmax=0.2)
plt.tight_layout()
plt.savefig("%s/heatmap_aign.png" % out_folder)
# MOFA
mids = []
for i in range(len(test_set_ids)):
    idx = search_list(mofa_ids, test_set_ids[i])
    if idx is not None:
        mids.append(idx)

plt.hexbin(test_z[mids].flatten(), mofa_z.flatten())
plt.xlabel("actual z")
plt.ylabel("pred z")
plt.title('MOFA')
plt.ylim(ymin=0., ymax=0.8)
plt.xlim(xmin=-0.2, xmax=0.2)
plt.tight_layout()
plt.savefig("%s/heatmap_mofa.png" % out_folder)
