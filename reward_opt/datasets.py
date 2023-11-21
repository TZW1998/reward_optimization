import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from importlib import resources
from reward_opt.global_path import *
import json
import tqdm
from reward_opt.diffusers_patch.pipeline_with_logprob import pipeline_with_logprob

# A customize dataset class for loading offline training data from a given folder of dataset
class ImageRewardDataset(Dataset):
    """dataset for (image, prompt, reward) triplets."""

    def __init__(self, image_folder, reward_data_path, tokenizer, threshold=0.0):
        self.image_folder= image_folder
      
        with open(reward_data_path, "r") as f:
            self.reward_data = json.load(f)
        
        self.images_name_list = list(self.reward_data.keys())
        
        self.tokenizer = tokenizer
        self.image_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

        self.reward_list = list(self.reward_data.values()) # used for query reward statistics
        self.reward_max = max(self.reward_list)
        self.reward_min = min(self.reward_list)

        # only finetune on the top 10% of data with highest reward
        threshold = np.percentile(self.reward_list, threshold*100)
        self.filtered_images_name_list = [img for img in self.images_name_list if self.reward_data[img] >= threshold]


    def __len__(self):
        return len(self.filtered_images_name_list)

    def __getitem__(self, idx):
        now_image_name = self.filtered_images_name_list[idx]
        image = Image.open(os.path.join(self.image_folder, now_image_name))

        reward = self.reward_data[now_image_name]
        # normalize reward to [0,1]
        reward = (reward - self.reward_min) / (self.reward_max - self.reward_min)
        reward = torch.tensor(reward)

        prompt = now_image_name.split("_")[-1].strip(".png")

        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.image_transforms(image)
        input_id = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        sample = {"pixel_values": image, "input_ids": input_id, "rewards": reward}

        return sample


def online_data_generation(pipeline, prompt_fn, reward_fn, config, accelerator, temp_image_folder, temp_reward_data_path):
    if accelerator.is_local_main_process:
        # remove the temp folder and temp reward data json file if exists
        if os.path.exists(temp_image_folder):
            shutil.rmtree(temp_image_folder)
        if os.path.exists(temp_reward_data_path):
            os.remove(temp_reward_data_path)

        # re-create the new temp folder
        os.makedirs(temp_image_folder)

    accelerator.wait_for_everyone()

    total_rounds = config.train.data_size // (config.sample.batch_size * accelerator.num_processes)

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

    img_score = {}
    # start sampling
    for round in tqdm.trange(total_rounds, desc="generation round",
            disable=not accelerator.is_local_main_process):
        # generate prompts
        prompts, prompt_metadata = zip(
            *[prompt_fn() for _ in range(config.sample.batch_size)]
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
                compute_kl=False,
            )

        rewards = reward_fn(images, prompts, prompt_metadata)
        # save images with rewards and prompts in file name
        for i, (image, prompt) in enumerate(zip(images, prompts)):
            
            img = f"{round}-{accelerator.process_index}-{i}_{prompt}.png"
            img_path = os.path.join(
                    temp_image_folder,
                    img,
                )

            image = image.cpu().numpy().transpose(1, 2, 0)
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
            image.save(img_path)
            img_score[img] = rewards[i]

    with open(temp_reward_data_path, "w") as f:
        json.dump(img_score, f)
