import torch
from torch.autograd import Variable
from iterators.iterator import (get_data_valid,
                                get_data_train,
                                get_data_test)
import numpy as np
from tqdm import tqdm
from scipy.stats.stats import pearsonr

from interactive import compute_covar
import util

MOFA_FILE = "/data/milatmp1/beckhamc/tmp_data/3dfaw/mofa.npz"
TEST_FILE = "/data/milatmp1/beckhamc/tmp_data/3dfaw/test.npz"

mofa_dat = np.load(MOFA_FILE)
mofa_ids = mofa_dat['ids']
mofa_ids = [ x.decode('utf-8') for x in mofa_ids ]

test_data = np.load(TEST_FILE)
print(test_data.keys())
ids = test_data['ids']
orients = test_data['orientations']





# Be able to map ids to specific indexes
# for the test set.
id_to_index = {}
for i in range(len(ids)):
    id_to_index[ ids[i] ] = i

# This dictionary maps ids to orientations.
id_to_orient = {}
for i in range(len(ids)):
    if orients[i] == 'left':
        id_to_orient[ ids[i] ] = 'left'
    elif orients[i] == 'center':
        id_to_orient[ ids[i] ] = 'center'
    elif orients[i] == 'right':
        id_to_orient[ ids[i] ] = 'right'
    else:
        raise Exception("??")

# Create the test set that only comprises the
# MOFA IDS.
test_with_mofa_ids = []
for mofa_id in mofa_ids:
    test_with_mofa_ids.append(id_to_index[mofa_id])
test_mofa_xy = test_data['y_keypts'][test_with_mofa_ids]
test_mofa_z = test_data['z_keypts'][test_with_mofa_ids]
test_mofa_orients = test_data['orientations'][test_with_mofa_ids]
# Now split into orientation.
test_mofa_z_left = test_mofa_z[ test_mofa_orients == 'left' ]
test_mofa_z_center = test_mofa_z[ test_mofa_orients == 'center' ]
test_mofa_z_right = test_mofa_z[ test_mofa_orients == 'right' ]

mofa_z = mofa_dat['kps'][:,-1,:]
mofa_xy = mofa_dat['kps'][:,0:2,:]

'''
mofa_lefts = []
mofa_centers = []
mofa_rights = []
for mofa_id in mofa_ids:
    mofa_lefts.append( id_to_orient[mofa_id]=='left' )
    mofa_centers.append( id_to_orient[mofa_id]=='center' )
    mofa_rights.append( id_to_orient[mofa_id]=='right' )
mofa_z_left = mofa_z[mofa_lefts]
mofa_z_center = mofa_z[mofa_centers]
mofa_z_right = mofa_z[mofa_rights]
'''

mofa_z_left = mofa_z[ test_mofa_orients == 'left' ]
mofa_z_center = mofa_z[ test_mofa_orients == 'center' ]
mofa_z_right = mofa_z[ test_mofa_orients == 'right' ]

np.savez("tmp/interactive_mofa.npz",
         mofa_xy=mofa_xy,
         mofa_z=mofa_z,
         test_mofa_xy=test_mofa_xy,
         test_mofa_z=test_mofa_z)

def kp_fn(xy1_keypts, z1_keypts, xy2_keypts, z2_keypts, pred_z1_keypts, same=False):
    l2_losses = []
    for i in range(len(xy1_keypts)):
        for j in range(len(xy2_keypts)):
            if same and i==j:
                continue
            # Prepare the source and target keypts.
            xy_keypt_src = xy1_keypts[i][np.newaxis]
            xy_keypt_tgt = xy2_keypts[j][np.newaxis]
            xy_keypt_tgt_torch = torch.from_numpy(xy_keypt_tgt).transpose(1,2)
            # Extract the i'th z value from pred_src_zs.
            pred_src_z = torch.from_numpy(pred_z1_keypts[i][np.newaxis]).unsqueeze(1)
            rhs = util.predict_tgt_kp_pseudoinv(xy_keypt_src,
                                                pred_src_z,
                                                xy_keypt_tgt)
            l2_loss = torch.mean((xy_keypt_tgt_torch - rhs)**2)
            l2_losses.append(l2_loss.data.item())
    return l2_losses

'''
# Predict KP error on overall test set.
all_all = kp_fn(test_mofa_xy, test_mofa_z,
                test_mofa_xy, test_mofa_z,
                mofa_z,
                same=True)
print(np.mean(all_all), np.std(all_all))
'''

######################################################

# Ok, measure the correlation matrices.

from interactive import compute_covar


left_left = compute_covar(preds=mofa_z_left, actuals=test_mofa_z_left)
print(left_left)

center_center = compute_covar(preds=mofa_z_center, actuals=test_mofa_z_center)
print(center_center)

right_right = compute_covar(preds=mofa_z_right, actuals=test_mofa_z_right)
print(right_right)

