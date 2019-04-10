#!/usr/bin/python3.7 python
"""

"""
__author__ = "Xavier Soria Poma, CVC-UAB"
__email__ = "xsoria@cvc.uab.es / xavysp@gmail.com"
__homepage__="www.cvc.uab.cat/people/xsoria"
__credits__=["tensorflow_tutorial","pix2pix_tesorflow",]
__copyright__   = "Copyright 2019, CIMI"

import argparse

from run_model import run_gan

parser = argparse.ArgumentParser(description='RGB2NIR parameters for feed the model')
parser.add_argument("--data_base_dir",default='/opt/dataset', help="path to folder containing images")
parser.add_argument("--data4train",default='EPFL', choices=["EPFL", "OMSIV",None])
parser.add_argument("--data4test",default='EPFL', choices=["EPFL", "OMSIV",None])
parser.add_argument('--train_list', default='train_pair.lst', type=str)  # SSMIHD: train_rgb_pair.lst, others train_pair.lst
parser.add_argument('--test_list', default='test_pair.lst', type=str)  # SSMIHD: train_rgb_pair.lst, others train_pair.lst
parser.add_argument("--model_state",default='train', choices=["train", "test", "export"])
parser.add_argument("--output_dir", default='/opt/results/rgb2nir', help="where to put output files")
parser.add_argument("--checkpoint_dir", default='checkpoints', help="directory with checkpoint to resume training from or use for testing")

parser.add_argument('--model_name', default='RGB2NIR', choices=['RGB2NIR'])
parser.add_argument('--is_rgb2nir', default=True, help='true for decolorization')
parser.add_argument("--max_epochs", type=int,default=1000, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--display_freq", type=int, default=10, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=250, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=8, help="number of images in batch")
parser.add_argument("--batch_normalization", type=bool, default=True, help=" use batch norm")
parser.add_argument("--image_height", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--image_width", type=int, default=256, help="scale images to this size before cropping to 256x256")
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

arg = parser.parse_args()
def main(args):
    model = run_gan(args=args)
    if args.model_state=='train':
        model.train()
    elif args.model_state =='test':
        model.test()
    else:
        raise NotImplementedError('Sorry you just can test or train the model, please set in '
                                  'args.model_state=train or test')

if __name__=='__main__':
    main(args=arg)