import imp
import glob
import os
import argparse
from aigns import (AIGN,
                   save_handler)
import interactive_aigns

'''
Process arguments.
'''
def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--name', type=str, default="deleteme")
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--lamb', type=float, default=10.)
    # Iterator returns (it_train_a, it_train_b, it_val_a, it_val_b)
    parser.add_argument('--iterator', type=str, default="iterators/mnist.py")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--interactive', type=str, default=None)
    parser.add_argument('--network', type=str, default="networks/mnist.py")
    parser.add_argument('--save_path', type=str, default='./results_aigns')
    parser.add_argument('--save_images_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
if args.interactive is not None:
    assert args.interactive in ['r2', 'non_model', 'free']
# Dynamically load network module.
net_module = imp.load_source('network', args.network)
gen_fn, disc_fn = getattr(net_module, 'get_network')()
# Dynamically load iterator module.
itr_module = imp.load_source('iterator', args.iterator)
itr_train, itr_val = getattr(itr_module, 'get_iterators')(args.batch_size)

gan_class = AIGN
gan_kwargs = {
    'g_fn': gen_fn,
    'd_fn': disc_fn,
    'opt_d_args': {'lr': args.lr, 'betas': (args.beta1, args.beta2)},
    'opt_g_args': {'lr': args.lr, 'betas': (args.beta1, args.beta2)},
    'lamb': args.lamb,
    'handlers': [save_handler("%s/%s" % (args.save_path, args.name))],
    'use_cuda': False if args.cpu else True
}
net = gan_class(**gan_kwargs)
print(net.g)
print(net.d)

if args.resume is not None:
    if args.resume == 'auto':
        # autoresume
        model_dir = "%s/%s/models" % (args.save_path, args.name)
        # List all the pkl files.
        files = glob.glob("%s/*.pkl" % model_dir)
        # Make them absolute paths.
        files = [os.path.abspath(key) for key in files]
        if len(files) > 0:
            # Get creation time and use that.
            latest_model = max(files, key=os.path.getctime)
            print("Auto-resume mode found latest model: %s" %
                  latest_model)
            net.load(latest_model)
    else:
        print("Loading model: %s" % args.resume)
        net.load(args.resume)
if args.interactive is not None:
    '''
    if args.interactive == 'r2':
        # Basically compute the R2 over
        # the entire validation set.
        #process_data_one_sweep("tmp/test.csv")
        interactive.measure_pearson_one_sweep(net)
    elif args.interactive == 'non_model':
        # Evaluate the non-model on the valid set.
        net.eval_on_iterator(itr_val_zipped, use_gt_z=True)
    else:
        import pdb; pdb.set_trace()
        #interactive.measure_pearson_test_pairwise(net)
    '''
    import pdb
    pdb.set_trace()
        
else:
    net.train(
        itr_train=itr_train,
        itr_valid=itr_val,
        epochs=args.epochs,
        model_dir="%s/%s/models" % (args.save_path, args.name),
        result_dir="%s/%s" % (args.save_path, args.name),
        save_every=args.save_every
    )
