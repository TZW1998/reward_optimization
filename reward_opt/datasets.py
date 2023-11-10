import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from importlib import resources
from reward_opt.global_path import *
import json

# A customize dataset class for loading offline training data from a given folder of dataset
class ImageRewardDataset(Dataset):
    """dataset for (image, prompt, reward) triplets."""

    def __init__(self, dataset_name, reward_data_name, tokenizer, threshold=0.0):
        self.image_folder= OFFLINE_IMAGE_PATH[dataset_name]
        reward_data_path = OFFLINE_REWARD_PATH[dataset_name][reward_data_name]
      
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