"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th

th.manual_seed(20)

import torch.distributed as dist
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion import dist_util as _dist_util
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

import pdb

def save_output(out_path, arr0, label_arr=None, save_npz=True):
    Image.fromarray(arr0).save(out_path+".png")
    if save_npz:
        if label_arr is not None:
            np.savez(out_path+".npz", arr0[None,...], label_arr[None,...])
        else:
            np.savez(out_path+".npz", arr0[None,...])

out_path = logger.get_dir()
if not os.path.exists(f"{out_path}/output"):
    os.mkdir(f"{out_path}/output")

def main():

    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    if args.class_cond:
        with open("imagenet.txt", "r") as f:
            idx2label =  f.readlines()
        idx2label = [n.strip("\n") for n in idx2label]
        idx2label = np.array(idx2label)

    logger.log("creating model and diffusion...")
    main_dict = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(
        **main_dict
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    N_sample = 0
    while N_sample < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            #classes = np.random.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,))
            classes = np.arange(N_sample, N_sample+args.batch_size)
            batch_text = idx2label[classes]
            classes = th.LongTensor(classes).to(dist_util.dev())
            #classes = th.randint(
            #    low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            #)
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        with th.no_grad():
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        sample = sample.cpu().numpy()
        if args.class_cond:
            classes = classes.cpu().numpy()

        #logger.log(f"created {len(all_images) * args.batch_size} samples")

        out_path = logger.get_dir()
        logger.log(f"saving to {out_path}")
        for i in range(len(sample)):
            #for i, text in enumerate(batch_text):
            if args.class_cond:
                text = batch_text[i]
            else:
                #text = str(N_sample)
                text = f"gdcat_{(N_sample+args.idx):02d}_1"
            out_path_i = os.path.join(out_path, "output", text)
            if args.class_cond:
                save_output(out_path_i, sample[i], classes[i])
                #np.savez(out_path, arr, label_arr)
            else:
                save_output(out_path_i, sample[i])
                #np.savez(out_path, arr)
            logger.log(f"saving to {out_path_i}")
            N_sample += 1

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        idx=0,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
