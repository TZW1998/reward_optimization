import ml_collections
import imp
import os
from reward_opt.global_path import *


def base():
    config = ml_collections.ConfigDict()

    ###### General ######
    # run name for wandb logging and checkpoint saving -- if not provided, will be auto-generated based on the datetime.
    config.run_name = "online_wrl"
    # random seed for reproducibility.
    config.seed = 42
    # top-level logging directory for checkpoint saving.
    config.logdir = "logs"
    # number of epochs to train for. each epoch is one round of sampling from the model followed by training on those
    # samples.
    config.num_epochs = 30
    # number of epochs between saving model checkpoints.
    config.save_freq = 20
    # number of checkpoints to keep before overwriting old ones.
    config.num_checkpoint_limit = 5
    # mixed precision training. options are "fp16", "bf16", and "no". half-precision speeds up training significantly.
    config.mixed_precision = "fp16"
    # allow tf32 on Ampere GPUs, which can speed up training.
    config.allow_tf32 = True
    # resume training from a checkpoint. either an exact checkpoint directory (e.g. checkpoint_50), or a directory
    # containing checkpoints, in which case the latest one will be used. `config.use_lora` must be set to the same value
    # as the run that generated the saved checkpoint.
    config.resume_from = ""
    # whether or not to use LoRA. LoRA reduces memory usage significantly by injecting small weight matrices into the
    # attention layers of the UNet. with LoRA, fp16, and a batch size of 1, finetuning Stable Diffusion should take
    # about 10GB of GPU memory. beware that if LoRA is disabled, training will take a lot of memory and saved checkpoint
    # files will also be large.
    config.use_lora = True
    config.lora_rank = 1

    ###### Pretrained Model ######
    config.pretrained = pretrained = ml_collections.ConfigDict()
    # base model to load. either a path to a local directory, or a model name from the HuggingFace model hub.
    #pretrained.model = "/home/zhiwei/research_dev/diffusion_codebase/backbone_models/sdv1-5-full-diffuser"
    pretrained.model = SD_PRETRAINED_PATH
    # revision of the model to load.
    pretrained.revision = "main"

    ###### Sampling (Notice that this samples are only used for evaluation, unlike online algorithm) ######
    config.sample = sample = ml_collections.ConfigDict()
    # number of sampler inference steps.
    sample.num_steps = 50
    # eta parameter for the DDIM sampler. this controls the amount of noise injected into the sampling process, with 0.0
    # being fully deterministic and 1.0 being equivalent to the DDPM sampler.
    sample.eta = 1.0
    # classifier-free guidance weight. 1.0 is no guidance.
    sample.guidance_scale = 5.0
    # batch size (per GPU!) to use for sampling.
    sample.batch_size = 4
    # number of batches to sample per epoch. the total number of samples per epoch is `num_batches_per_epoch *
    # batch_size * num_gpus`.
    sample.num_batches_per_epoch = 1

    ###### Training ######
    config.train = train = ml_collections.ConfigDict()
    # should tune num_steps_per_epoch, batch_size, gradient_accumulation_steps so that each epoch consume roungly the same with ddpo.
    # number of gradient steps per epoch. This means that at each epoch, it will load num_steps * batch_size * num_gpu * gradient_accumulation_steps samples
    train.num_steps_per_epoch = 85 
    # batch size (per GPU!) to use for training.
    train.batch_size = 1
    # whether to use the 8bit Adam optimizer from bitsandbytes.
    train.use_8bit_adam = False
    # learning rate.
    train.learning_rate = 1e-4
    # Adam beta1.
    train.adam_beta1 = 0.9
    # Adam beta2.
    train.adam_beta2 = 0.999
    # Adam weight decay.
    train.adam_weight_decay = 1e-4
    # Adam epsilon.
    train.adam_epsilon = 1e-8
    # number of gradient accumulation steps. the effective batch size is `batch_size * num_gpus *
    # gradient_accumulation_steps`.
    train.gradient_accumulation_steps = 1
    # maximum gradient norm for gradient clipping.
    train.max_grad_norm = 1.0
    # number of inner epochs per outer epoch. each inner epoch is one iteration through the current subset of offline datasets
    train.num_inner_epochs = 1
    # whether or not to use classifier-free guidance during training. if enabled, the same guidance scale used during
    # sampling will be used during training.
    train.cfg = True
    train.temperatures = 0.2
    train.data_epoch = 10 # update the dataset every 10 epochs
    train.data_size = 32 # number of samples to use for each dataset
    train.filter_threshold = 0.0 # the threshold to filter the samples

    ###### Prompt Function (only for evaluate) ######
    # prompt function to use. see `prompts.py` for available prompt functions.
    config.prompt_fn = "simple_animals"
    # kwargs to pass to the prompt function.
    config.prompt_fn_kwargs = {}

    ###### Reward Function (only for evaluate) ######
    # reward function to use. see `rewards.py` for available reward functions.
    config.reward_fn = "jpeg_compressibility"

    return config



def aesthetic():
    config = base()
    config.reward_fn = "aesthetic_score"
    config.prompt_fn = "simple_animals"
 
    return config



def get_config(name):
    return globals()[name]()
