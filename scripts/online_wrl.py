from collections import defaultdict
import contextlib
import os, shutil
import datetime
from concurrent import futures
import time
from absl import app, flags
from ml_collections import config_flags
from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
import numpy as np
import reward_opt.prompts
import reward_opt.rewards
from reward_opt.datasets import ImageRewardDataset, online_data_generation
from reward_opt.stat_tracking import PerPromptStatTracker
from reward_opt.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob
from reward_opt.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import torch.nn.functional as F
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from copy import deepcopy
from reward_opt.global_path import *

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/owrl_config.py", "Training configuration.")

logger = get_logger(__name__)

### MODIFY THIS: Decide how to compute weights using reward, here is one choice
def reward2weight(rewards, config):
    # give 1 if reward > -70, otherwise give 0
    temperatures = config.train.temperatures
    weights = torch.exp((rewards - 1) / temperatures)
    return weights


def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from)))
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="reward_opt-pytorch", config=config.to_dict(), init_kwargs={"wandb": {"name": f"{config.run_name}_{config.prompt_fn}_{config.reward_fn}_{config.lora_rank}"}}
        )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(config.pretrained.model, revision=config.pretrained.revision)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)
    # disable safety checker
    pipeline.safety_checker = None
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )
    # switch to DDIM scheduler
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    
    # noise scheduler for training
    noise_scheduler = DDPMScheduler.from_pretrained(config.pretrained.model, subfolder="scheduler")

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    # clone the original unet so that we can compute the kl loss
    pipeline.unet_orig = deepcopy(pipeline.unet)

    if config.use_lora:
        # Set correct lora layers
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

            lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank = config.lora_rank)
        pipeline.unet.set_attn_processor(lora_attn_procs)

        # this is a hack to synchronize gradients properly. the module that registers the parameters we care about (in
        # this case, AttnProcsLayers) needs to also be used for the forward pass. AttnProcsLayers doesn't have a
        # `forward` method, so we wrap it to add one and capture the rest of the unet parameters using a closure.
        class _Wrapper(AttnProcsLayers):
            def forward(self, *args, **kwargs):
                return pipeline.unet(*args, **kwargs)

        unet = _Wrapper(pipeline.unet.attn_processors)
    else:
        unet = pipeline.unet

    # set up diffusers-friendly checkpoint saving with Accelerate

    def save_model_hook(models, weights, output_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            pipeline.unet.save_attn_procs(output_dir)
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            models[0].save_pretrained(os.path.join(output_dir, "unet"))
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        weights.pop()  # ensures that accelerate doesn't try to handle saving of the model

    def load_model_hook(models, input_dir):
        assert len(models) == 1
        if config.use_lora and isinstance(models[0], AttnProcsLayers):
            # pipeline.unet.load_attn_procs(input_dir)
            tmp_unet = UNet2DConditionModel.from_pretrained(
                config.pretrained.model, revision=config.pretrained.revision, subfolder="unet"
            )
            tmp_unet.load_attn_procs(input_dir)
            models[0].load_state_dict(AttnProcsLayers(tmp_unet.attn_processors).state_dict())
            del tmp_unet
        elif not config.use_lora and isinstance(models[0], UNet2DConditionModel):
            load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
            models[0].register_to_config(**load_model.config)
            models[0].load_state_dict(load_model.state_dict())
            del load_model
        else:
            raise ValueError(f"Unknown model type {type(models[0])}")
        models.pop()  # ensures that accelerate doesn't try to handle loading of the model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        unet.parameters(),
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    prompt_fn = getattr(reward_opt.prompts, config.prompt_fn)
    reward_fn = getattr(reward_opt.rewards, config.reward_fn)()

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
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # autocast = accelerator.autocast

    ########################### Get the offline training datasets ###############################
    # prepare the data loader for offline dataset
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"][0] for example in examples])
        
        rewards = torch.stack([example["rewards"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids, "rewards": rewards}

    actual_batch_size_per_device = config.train.batch_size * config.train.gradient_accumulation_steps

    # Prepare everything with our `accelerator`.
    unet, optimizer = accelerator.prepare(unet, optimizer)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=2)

    # Train!
    samples_per_epoch = config.sample.batch_size * accelerator.num_processes * config.sample.num_batches_per_epoch
    total_train_batch_size = (
        config.train.batch_size * accelerator.num_processes * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Total samples used for evaluation per epoch = {samples_per_epoch}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}")
    logger.info("")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
    logger.info(f"  Number of gradient updates per epoch = {config.train.num_steps_per_epoch}")

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0
    
    # start time for the main process
    if accelerator.is_local_main_process:
        start_time = time.time()

    temp_image_folder= "temp_image_folder"
    temp_reward_data_path = "temp_reward_data.json"

    for epoch in range(first_epoch, config.num_epochs):
        #################### SAMPLING (only for evaluate) ####################
        pipeline.unet.eval()
        samples = []
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            # generate prompts
            prompts, prompt_metadata = zip(
                *[prompt_fn(**config.prompt_fn_kwargs) for _ in range(config.sample.batch_size)]
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
            with autocast():
                images, _, _, _, mean_kl_div = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=config.sample.num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    eta=config.sample.eta,
                    output_type="pt",
                )

            # compute rewards asynchronously
            rewards = executor.submit(reward_fn, images, prompts, prompt_metadata)
            # yield to to make sure reward computation starts
            time.sleep(0)

            samples.append(
                {
                    "rewards": rewards,
                    "mean_kl_div": mean_kl_div,
                }
            )

        # wait for all rewards to be computed
        for sample in tqdm(
            samples,
            desc="Waiting for rewards",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            rewards, reward_metadata = sample["rewards"].result()
            # accelerator.print(reward_metadata)
            sample["rewards"] = torch.as_tensor(rewards, device=accelerator.device)

        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {k: torch.cat([s[k] for s in samples]) for k in samples[0].keys()}

        # this is a hack to force wandb to log the images as JPEGs instead of PNGs
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(images):
                pil = Image.fromarray((image.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
                pil = pil.resize((256, 256))
                pil.save(os.path.join(tmpdir, f"{i}.jpg"))
            accelerator.log(
                {
                    "images": [
                        wandb.Image(os.path.join(tmpdir, f"{i}.jpg"), caption=f"{prompt:.25} | {reward:.2f}")
                        for i, (prompt, reward) in enumerate(zip(prompts, rewards))  # only log rewards from process 0
                    ],
                },
                step=epoch,
            )

        # gather rewards across processes
        rewards = accelerator.gather(samples["rewards"]).cpu().numpy()

        # compute the mean_kl_div across processes via reduces
        reduced_mean_kl_div = accelerator.reduce(samples["mean_kl_div"].mean(), reduction="mean").item()

        # compute rewards quantiles
        use_quantiles = [0, 0.05, 0.1, 0.2, 0.5, 1.0]
        quantiles = np.quantile(rewards, use_quantiles)

        # log rewards and time at main process
        if accelerator.is_local_main_process:
            now_time = time.time() - start_time
            log_dict = {"time": now_time, "reward": rewards, "reward_mean": rewards.mean(), "reward_std": rewards.std(), "mean_kl_div": reduced_mean_kl_div}
            for q, v in zip(use_quantiles, quantiles):
                log_dict[f"reward_q{q}"] = v
            accelerator.log(log_dict,
                step=epoch,
            )


        #################### TRAINING ####################
        # function for computing weighted loss and do accumulated backward
        def per_sample_loss(batch):
            # Convert images to latent space
            batch_pixel_values = batch["pixel_values"].to(accelerator.device, dtype=inference_dtype)
            latents = pipeline.vae.encode(batch_pixel_values).latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
   
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            embeds = pipeline.text_encoder(batch["input_ids"].to(accelerator.device))[0]

            if config.train.cfg:
                # concat negative prompts to sample prompts to avoid two forward passes
                embeds = torch.cat([train_neg_prompt_embeds, embeds])


            # Predict the noise residual and compute loss
            with autocast():
                if config.train.cfg:
                    model_pred = unet(torch.cat([noisy_latents]*2), torch.cat([timesteps]*2), embeds).sample
                    model_pred_uncond, model_pred_text = model_pred.chunk(2)
                    model_pred = model_pred_uncond + config.sample.guidance_scale * (
                        model_pred_text - model_pred_uncond
                    )
                else:
                    model_pred = unet(noisy_latents, timesteps, embeds).sample

            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
            sample_loss = loss.mean(dim=list(range(1, len(loss.shape)))) 

            return sample_loss
        

        # collect the dataset used for training for every config.train.data_epoch 
        if epoch % config.train.data_epoch == 0:
            logger.info(f"start updating online dataset at epoch {epoch}")
            logger.info(f"Prompt function: {config.prompt_fn}")
            logger.info(f"Total number of samples: {config.train.data_size}")
            logger.info(f"Number of GPUs: {accelerator.num_processes}")

            online_data_generation(pipeline, prompt_fn, reward_fn, config, accelerator, temp_image_folder, temp_reward_data_path)

            logger.info(f"online dataset updated at epoch {epoch} finished")

            online_dataset = ImageRewardDataset(temp_image_folder, temp_reward_data_path, pipeline.tokenizer, threshold=config.train.filter_threshold)
            dataloader = torch.utils.data.DataLoader(online_dataset,
                                                                shuffle=True,
                                                                collate_fn=collate_fn,
                                                                batch_size=actual_batch_size_per_device,
                                                                num_workers=2,
                                                            )
            dataloader = accelerator.prepare(dataloader)
            

        # main training loop, execute num_steps_per_epoch * gradient_accumulation_steps times backpropogation
        data_iterable = iter(dataloader)
        pipeline.unet.train()


        for step in tqdm(range(config.train.num_steps_per_epoch),
            desc=f"Epoch {epoch}: training",
            position=0,
            disable=not accelerator.is_local_main_process,
        ):
            # load next batch
            try:
                batch = next(data_iterable)
            except:
                data_iterable = iter(offline_dataloader)
                batch = next(data_iterable)

            # batch weights
            batch_rewards = batch["rewards"].to(accelerator.device, dtype=torch.float32)

            # compute the weights
            reward_weights = reward2weight(batch_rewards, config)

            # backward pass for config.train.gradient_accumulation_steps times
            for now_acc_step in range(config.train.gradient_accumulation_steps):
                now_sub_batch = {k: v[(now_acc_step * config.train.batch_size) : ((now_acc_step + 1) * config.train.batch_size)] for k, v in batch.items()}
                now_sub_batch_weights = reward_weights[(now_acc_step * config.train.batch_size) : ((now_acc_step + 1) * config.train.batch_size)]
                with accelerator.accumulate(unet):
                    # computing loss and do accumulated backward
                    sample_loss = per_sample_loss(now_sub_batch)

                    #import ipdb; ipdb.set_trace()

                    loss = (sample_loss * now_sub_batch_weights).sum() / total_train_batch_size
                

                    # backward pass
                    accelerator.backward(loss)

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(unet.parameters(), config.train.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            # make sure we did an optimization step at the end of every batch
            assert accelerator.sync_gradients
   

        # ToDo: this save_state causes some bugs, don't know why
        #if epoch != 0 and epoch % config.save_freq == 0 and accelerator.is_main_process:
        #    accelerator.save_state()
        # maybe it is because save_state took too much time, so maybe we need to save the model manually


if __name__ == "__main__":
    app.run(main)
