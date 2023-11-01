# Path to pretrained models
SD_PRETRAINED_PATH = '/home/zhiweitang/sdv1-5-full-diffuser'
CLIP_VIT_PRETRAINED_PATH = '/home/zhiweitang/clip-vit-large-patch14'

# Path to offline image data
OFFLINE_IMAGE_PATH = {
    "simple_animals": "/home/zhiweitang/simple_animals_data",
}


# Path to offline reward data
OFFLINE_REWARD_PATH = {
    "simple_animals": {
        "aesthetic_score": "/home/zhiweitang/simple_animals_aesthetic_score.json",
        "jpeg_compressibility" : "/home/zhiweitang/simple_animals_jpeg_compressibility.json",
    }
}