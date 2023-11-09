from reward_opt.rewards import aesthetic_score, jpeg_compressibility
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
import json
from argparse import ArgumentParser
import reward_opt.rewards


def main(args):
    
    reward_fn = getattr(reward_opt.rewards, args.reward_fn)()

    full_image_list = os.listdir(args.data_dir)

    img_score = {}

    processor = transforms.ToTensor()
    for img in tqdm(full_image_list):
        img_path = os.path.join(args.data_dir, img)
        
        image = processor(Image.open(img_path))
        
        if args.reward_fn == "jpeg_compressibility":
            score = reward_fn(image.unsqueeze(0),None,None)
        else:
            score = reward_fn(image,None,None)

        img_score[img] = score[0].item()

    root_dir = os.path.dirname(args.data_dir)
    data_name = os.path.basename(args.data_dir)
    with open(os.path.join(root_dir, f"{data_name}_{args.reward_fn}.json"), "w") as f:
        json.dump(img_score, f)

    print("saved")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--data-dir", )
    args.add_argument("--reward-fn", type=str, default="aesthetic_score")
    main(args.parse_args())