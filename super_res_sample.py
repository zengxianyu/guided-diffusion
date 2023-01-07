"""
Generate a large batch of samples from a super resolution model, given a batch
of samples from a regular model from image_sample.py.
"""

import torch.nn.functional as F
from PIL import Image
import pdb
import argparse
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)

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

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
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
        image_size=args.small_size,
        class_cond=True,
        return_name=True,
        return_prefix=True,
        deterministic=True,
    )

    logger.log("creating samples...")
    bdata = next(data, None)
    while bdata is not None:
        img, model_kwargs = bdata
        img = img.to(dist_util.dev())
        filename = model_kwargs.pop("filename")
        prefix = model_kwargs.pop("prefix")
        #model_kwargs["low_res"] = F.interpolate(img, args.small_size, mode="area")
        model_kwargs["low_res"] = img
        bdata = next(data)
        model_kwargs = {k: v.to(dist_util.dev()) for k, v in model_kwargs.items()}
        sample = diffusion.p_sample_loop(
            model,
            (args.batch_size, 3, args.large_size, args.large_size),
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
        small_size=64,
        large_size=256,
        use_ddim=False,
        model_path="",
        data_dir=""
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
