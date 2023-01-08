"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
from PIL import Image

import numpy as np
import torch as th

th.manual_seed(20)

import torch.distributed as dist
import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
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

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("loading data...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        return_name=True,
        return_prefix=True,
        deterministic=True,
        n_split=args.n_split,
        i_split=args.i_split,
        imagenet=args.imagenet,
        return_loader=True
    )
    data = iter(data)

    logger.log("sampling...")
    bdata = next(data, None)
    while bdata is not None:
        img, cond = bdata
        img = img.to(dist_util.dev())
        filename = cond.pop("filename")
        prefix = cond.pop("prefix")
        bdata = next(data)
        model_kwargs = {}
        model_kwargs["y"] = cond["y"].to(dist_util.dev())
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (cond["y"].size(0), 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
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
        data_dir="",
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        i_split=0,
        n_split=1,
        imagenet=False
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
