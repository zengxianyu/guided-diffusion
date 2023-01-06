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
from guided_diffusion.image_datasets import load_data
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

    logger.log("loading data...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        return_name=True,
        return_prefix=True,
        deterministic=True,
    )

    logger.log("sampling...")
    bdata = next(data, None)
    while bdata is not None:
        pdb.set_trace()
        img, cond = bdata
        img = img.to(dist_util.dev())
        filename = cond.pop("filename")
        prefix = cond.pop("prefix")
        bdata = next(data)
        model_kwargs = {}
        if args.class_cond:
            model_kwargs["y"] = cond["y"].to(dist_util.dev())
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

        #logger.log(f"created {len(all_images) * args.batch_size} samples")

        out_path = logger.get_dir()
        logger.log(f"saving to {out_path}")
        for i in range(len(sample)):
            out_path_i = os.path.join(out_path, "output", prefix[i])
            if not os.path.exists(out_path_i):
                os.mkdir(out_path_i)
            out_path_i = os.path.join(out_path_i, filename[i])
            save_output(out_path_i, sample[i])
            logger.log(f"saving to {out_path_i}")

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=16,
        use_ddim=False,
        model_path="",
        data_dir=""
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
