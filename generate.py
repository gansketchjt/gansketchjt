
import jittor as jt
from jittor import init
from jittor import nn
import os
import argparse
import random
import numpy as np
import tqdm
from torchvision import utils
import torch
import pickle
from training.networks.stylegan2 import Generator
jt.flags.use_cuda=1
np.random.seed(0)
def save_image_pytorch(img, name):
    'Helper function to save torch tensor into an image file.'
    img=torch.from_numpy(np.array(img))
    utils.save_image(
        img,
        name,
        nrow=1,
        padding=0,
        normalize=True,
        range=(-1, 1),
    )
    # jt.save_image(img, name, nrow=1, padding=0, normalize=False, range=( -1, 1))

def generate(args, netG, mean_latent):
    'Generates images from a generator.'
    if (args.w_shift is not None):
        w_shift = jt.array(np.load(args.w_shift))
        w_shift = w_shift[None, :]
        mean_latent = (mean_latent + w_shift)
    else:
        w_shift = jt.Var(0.0)
    ind = 0
    with jt.no_grad():
        netG.eval()
        if (args.fixed_z is not None):
            sample_z = (jt.load(args.fixed_z) + w_shift)
         
            for start in range(0, sample_z.shape[0], args.batch_size):
                end = min((start + args.batch_size), sample_z.shape[0])
                z_batch = sample_z[start:end]
                (sample, _) = netG([z_batch], truncation=args.truncation, truncation_latent=mean_latent)
                for s in sample:
                    save_image_pytorch(s, f'{args.save_dir}/{str(ind).zfill(6)}after.png')
                    ind += 1
            if args.origin_ckpt is None:
                return
            ind=0
            checkpoint = jt.load(args.origin_ckpt)
            netG.load_parameters(checkpoint)
            sample_z = (jt.load(args.fixed_z) + w_shift)
         
            for start in range(0, sample_z.shape[0], args.batch_size):
                end = min((start + args.batch_size), sample_z.shape[0])
                z_batch = sample_z[start:end]
                (sample, _) = netG([z_batch], truncation=args.truncation, truncation_latent=mean_latent)
                for s in sample:
                    save_image_pytorch(s, f'{args.save_dir}/{str(ind).zfill(6)}before.png')
                    ind += 1

            return
        zs=[]
        for start in range(0, args.samples, args.batch_size):
            end = min((start + args.batch_size), args.samples)
            batch_sz = (end - start)
            sample_z = (jt.randn(batch_sz, 512) + w_shift)
            zs.append(sample_z)     
            (sample, _) = netG([sample_z], truncation=args.truncation, truncation_latent=mean_latent)
            for s in sample:
                save_image_pytorch(s, f'{args.save_dir}/{str(ind).zfill(6)}after.png')
                ind += 1
        if args.origin_ckpt is None:
                return
        ind=0
        checkpoint = jt.load(args.origin_ckpt)
        netG.load_parameters(checkpoint)
        zid=0
        for start in range(0, args.samples, args.batch_size):
            end = min((start + args.batch_size), args.samples)
            batch_sz = (end - start)
            # sample_z = (jt.randn(batch_sz, 512) + w_shift)       
            sample_z=zs[zid]
            zid=zid+1
            (sample, _) = netG([sample_z], truncation=args.truncation, truncation_latent=mean_latent)
            for s in sample:
                save_image_pytorch(s, f'{args.save_dir}/{str(ind).zfill(6)}before.png')
                ind += 1
if (__name__ == '__main__'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./output', help='place to save the output')
    parser.add_argument('--ckpt', type=str, default=None, help='checkpoint file for the generator')
    parser.add_argument('--size', type=int, default=256, help='output size of the generator')
    parser.add_argument('--fixed_z', type=str, default=None, help='expect a .pth file. If given, will use this file as the input noise for the output')
    parser.add_argument('--origin_ckpt',type=str,default=None)
    parser.add_argument('--w_shift', type=str, default=None, help='expect a .pth file. Apply a w-latent shift to the generator')
    parser.add_argument('--batch_size', type=int, default=50, help='batch size used to generate outputs')
    parser.add_argument('--samples', type=int, default=50, help='number of samples to generate, will be overridden if --fixed_z is given')
    parser.add_argument('--truncation', type=float, default=0.5, help='strength of truncation')
    parser.add_argument('--truncation_mean', type=int, default=4096, help='number of samples to calculate the mean latent for truncation')
    parser.add_argument('--seed', type=int, default=None, help='if specified, use a fixed random seed')
    args = parser.parse_args()
    if (args.seed is not None):
        random.seed(args.seed)
        jt.set_global_seed(args.seed)
    if (not os.path.exists(args.save_dir)):
        os.makedirs(args.save_dir)
    netG = Generator(args.size, 512, 8)
    checkpoint = jt.load(args.ckpt)
    netG.load_parameters(checkpoint)
    if (args.truncation < 1):
        with jt.no_grad():
            mean_latent = netG.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, netG, mean_latent)
