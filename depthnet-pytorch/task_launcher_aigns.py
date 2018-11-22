#import imp
from importlib import import_module
import glob
import os
import argparse
from aigns import (AIGN,
                   save_handler)
from interactive_aigns import (measure_depth,
                               measure_kp_error)

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
    parser.add_argument('--dnorm', type=float, default=0.)
    # Iterator returns (it_train_a, it_train_b, it_val_a, it_val_b)
    parser.add_argument('--iterator', type=str, default="iterators/mnist.py")
    parser.add_argument('--resume', type=str, default=None)
    spec = parser.add_mutually_exclusive_group()
    spec.add_argument('--interactive', action='store_true')
    spec.add_argument('--compute_stats', action='store_true')
    spec.add_argument('--dump_depths', type=str)
    parser.add_argument('--update_g_every', type=int, default=1)
    parser.add_argument('--network', type=str, default="networks/mnist.py")
    parser.add_argument('--save_path', type=str, default='./results_aigns')
    parser.add_argument('--save_images_every', type=int, default=100)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    return args

args = parse_args()
# Dynamically load network module.
#net_module = imp.load_source('network', args.network)
net_module = import_module(args.network.replace("/", ".").\
                    replace(".py", ""))
gen_fn, disc_fn = getattr(net_module, 'get_network')()
# Dynamically load iterator module.
#itr_module = imp.load_source('iterator', args.iterator)
itr_module = import_module(args.iterator.replace("/", ".").\
                    replace(".py", ""))
itr_train, itr_val = getattr(itr_module, 'get_iterators')(args.batch_size)

gan_class = AIGN
gan_kwargs = {
    'g_fn': gen_fn,
    'd_fn': disc_fn,
    'opt_d_args': {'lr': args.lr, 'betas': (args.beta1, args.beta2)},
    'opt_g_args': {'lr': args.lr, 'betas': (args.beta1, args.beta2)},
    'lamb': args.lamb,
    'dnorm': args.dnorm,
    'update_g_every': args.update_g_every,
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
if args.interactive:
    import pdb
    pdb.set_trace()
elif args.compute_stats:
    print("Computing stats on validation set...")
    measure_depth(net, grid=False, mode='valid')
    measure_kp_error(net, grid=False, mode='valid')
    print("Computing stats on test set...")
    measure_depth(net, grid=True, mode='test')
    measure_kp_error(net, grid=False, mode='test')
elif args.dump_depths is not None:
    print("Dumping depths to file: %s" % args.dump_depths)
    measure_depth(net, grid=False, mode='test',
                  dump_file=args.dump_depths)
else:
    net.train(
        itr_train=itr_train,
        itr_valid=itr_val,
        epochs=args.epochs,
        model_dir="%s/%s/models" % (args.save_path, args.name),
        result_dir="%s/%s" % (args.save_path, args.name),
        save_every=args.save_every
    )
