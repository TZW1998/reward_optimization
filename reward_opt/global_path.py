# Path to pretrained models
SD_PRETRAINED_PATH = 'reward_opt/pretrained_models/sd.pth'
CLIP_VIT_PRETRAINED_PATH = 'reward_opt/pretrained_models/clip_vit.pth'

# Path to offline image data
OFFLINE_IMAGE_PATH = {
    "simple_animals": "reward_opt/data/simple_animals",
}


# Path to offline reward data
OFFLINE_REWARD_PATH = {
    "simple_animals": {
        "aesthetic_score": "reward_opt/data/simple_animals/aesthetic.txt",
        "jpeg_compressibility" : "reward_opt/data/simple_animals/jpeg_compressibility.txt",
    }
}