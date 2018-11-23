from .shared.networks import define_G
from .shared.image2image_old import Discriminator

def get_network():
    gen_atob_fn = define_G(input_nc=3,
                           ngf=64,
                           output_nc=3,
                           which_model_netG='resnet_9blocks',
                           norm='instance')
    disc_a_fn = Discriminator(input_dim=3,
                              num_filter=64,
                              output_dim=1)
    gen_btoa_fn = define_G(input_nc=3,
                           ngf=64,
                           output_nc=3,
                           which_model_netG='resnet_9blocks',
                           norm='instance')
    disc_b_fn = Discriminator(input_dim=3,
                              num_filter=64,
                              output_dim=1)
    return (gen_atob_fn,
            disc_a_fn,
            gen_btoa_fn,
            disc_b_fn)
