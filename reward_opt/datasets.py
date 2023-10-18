import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
from importlib import resources

ASSETS_PATH = resources.files("reward_opt.assets")

# A customize dataset class for loading offline training data from a given folder of dataset
class ImageRewardDataset(Dataset):
    """dataset for (image, prompt, reward) triplets."""

    def __init__(self, dataset_name, tokenizer):
        image_folder = ASSETS_PATH.joinpath(dataset_name + ".txt")
        images_name_list = os.listdir(image_folder)
        self.prompts_list = [name.strip(".png").split("_")[-1] for name in images_name_list]
        self.rewards_list = [float(name.strip(".png").split("_"))[2] for name in images_name_list]
        self.images_full_path_list = [os.path.join(image_folder, name) for name in images_name_list]
        self.tokenizer = tokenizer
        self.image_transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        )

    def __len__(self):
        return len(self.images_full_path_list)

    def __getitem__(self, idx):
        image = Image.open(self.images_full_path_list[idx])
        reward = torch.float32(self.rewards_list[0])

        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = self.image_transforms(image)
        input_id = self.tokenizer(
            self.prompts_list[idx],
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids

        sample = {"pixel_values": image, "input_ids": input_id, "rewards": reward}

        return sample