from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator, PartialState
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel, DPMSolverMultistepScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import reward_opt.prompts
import reward_opt.rewards
from reward_opt.stat_tracking import PerPromptStatTracker
from reward_opt.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from reward_opt.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from argparse import ArgumentParser
from torchvision import transforms

def main():
    args = ArgumentParser()
    args.add_argument("--num_samples", type=int, default=16)
    args.add_argument("--batch_size_per_device", type=int, default=1)
    args.add_argument("--model", type=str, default="/mnt/workspace/workgroup/tangzhiwei.tzw/sdv1-5-full-diffuser")
    args.add_argument("--lora_rank", type=int, default=4)
    args.add_argument("--lora_path", default="/mnt/workspace/workgroup/tangzhiwei.tzw/reward_optimization/logs/w_ddpo_2023.12.19_21.31.10_unet_80.pt")
    args.add_argument("--sampler", type=str, default="DDIM", choices = ["DDIM", "DPMSolver"])
    args.add_argument("--steps", type=int, default=50)
    args.add_argument("--eta", type=float, default=1)
    args.add_argument("--guidance_scale", type=float, default=5)
    args.add_argument("--prompt_fn", type=str, default="simple_animals")
    args.add_argument("--reward_fn", type=str, default="jpeg_compressibility")
    args.add_argument("--output_dir", type=str, default="ddim_data")


    args = args.parse_args()

    # set up accelerator
    accelerator = Accelerator()
    pipeline = StableDiffusionPipeline.from_pretrained(args.model, revision="fp16", torch_dtype=torch.float16)
    # disable safety checker
    pipeline.safety_checker = None

    if args.sampler == "DDIM":
        pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    elif args.sampler == "DPMSolver":
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    if (args.lora_path is not None) and (args.lora_path != ""):
        lora_attn_procs = {}
        for name in pipeline.unet.attn_processors.keys():
            cross_attention_dim = (
                None if name.endswith("attn1.processor") else pipeline.unet.config.cross_attention_dim
            )
            if name.startswith("mid_block"):
                hidden_size = pipeline.unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(pipeline.unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = pipeline.unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=4)
        pipeline.unet.set_attn_processor(lora_attn_procs)

        pipeline.to(accelerator.device)

        pipeline(prompt=["cat"]).images[0] # do some renaming

        lora_state_dict = torch.load(args.lora_path)
        pipeline.unet.load_state_dict(lora_state_dict)

    else:

        pipeline.to(accelerator.device)

    # prepare prompt and reward fn
    prompt_fn = getattr(reward_opt.prompts, args.prompt_fn)
    reward_fn = getattr(reward_opt.rewards, args.reward_fn)()

    to_tensor = transforms.ToTensor()

    total_rounds = args.num_samples // (args.batch_size_per_device * accelerator.num_processes)

    # prepare output dir, if it exists, delete it
    if accelerator.is_local_main_process:
        if os.path.exists(args.output_dir):
            import shutil

            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

    # print info, including, prompt_fn, reward_fn total numsamples, numbers of gpu, total rounds
    if accelerator.is_local_main_process:
        print(f"Prompt function: {args.prompt_fn}")
        print(f"Total number of samples: {args.num_samples}")
        print(f"Number of GPUs: {accelerator.num_processes}")
        print(f"Total rounds: {total_rounds}")

    # start sampling
    for round in tqdm.trange(total_rounds, desc="generation round",
            disable=not accelerator.is_local_main_process):
            
        # generate prompts
        prompts, prompt_metadata = zip(
            *[prompt_fn() for _ in range(args.batch_size_per_device)]
        )

        prompts = list(prompts)

        with torch.no_grad():
            images = pipeline(prompt=prompts,eta=args.eta, num_inference_steps = args.steps, guidance_scale = args.guidance_scale).images
            

        # save images with rewards and prompts in file name
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            image_tensor = to_tensor(image)
            score = reward_fn(image_tensor.unsqueeze(0), prompt, None)[0][0]
            image.save(
                os.path.join(
                    args.output_dir,
                    f"{round}-{accelerator.process_index}-{i}_{prompt}_{score}.png",
                )
            )
        

if __name__ == "__main__":
    main()