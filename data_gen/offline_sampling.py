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
from diffusers import StableDiffusionPipeline, DDIMScheduler, UNet2DConditionModel
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

def main():
    args = ArgumentParser()
    args.add_argument("--num_samples", type=int, default=10)
    args.add_argument("--batch_size_per_device", type=int, default=1)
    args.add_argument("--model", type=str, default="/home/zhiweitang/sdv1-5-full-diffuser")
    args.add_argument("--prompt_fn", type=str, default="simple_animals")
    args.add_argument("--output_dir", type=str, default="simple_animals_data")


    args = args.parse_args()

    # set up accelerator
    accelerator = Accelerator()
    pipeline = StableDiffusionPipeline.from_pretrained(args.model, revision="fp16", torch_dtype=torch.float16)
    # disable safety checker
    pipeline.safety_checker = None

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline.to(accelerator.device)

    # prepare prompt and reward fn
    prompt_fn = getattr(reward_opt.prompts, args.prompt_fn)

    # generate negative prompt embeddings
    neg_prompt_embed = pipeline.text_encoder(
        pipeline.tokenizer(
            [""],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
    )[0]
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(args.batch_size_per_device, 1, 1)

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

        # encode prompts
        prompt_ids = pipeline.tokenizer(
            prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=pipeline.tokenizer.model_max_length,
        ).input_ids.to(accelerator.device)
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0]

        # sample
        with torch.no_grad():
            images, _, _, _ = pipeline_with_logprob(
                pipeline,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=sample_neg_prompt_embeds,
                num_inference_steps=50,
                guidance_scale=5,
                eta=1,
                output_type="pt",
            )

        # save images with rewards and prompts in file name
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            image = image.cpu().numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(
                os.path.join(
                    args.output_dir,
                    f"{round}-{accelerator.process_index}-{i}_{prompt}.png",
                )
            )
        

if __name__ == "__main__":
    main()