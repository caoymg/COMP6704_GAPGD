import torch
import torchvision
import pandas as pd
from PIL import Image

from utils.misc import str2bool, dict2namespace
from clip_utils.clip_models import load_surrogate_clips
from clip_utils.transforms import transform_clean, clip_preprocess, source_crop, target_crop
from attack.gapgd_attack import gapgd_generate
from argparse import ArgumentParser
import yaml, os

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='imagenet.yml')
    parser.add_argument('--eval_nums', type=int, default=23)
    parser.add_argument('--attack_iter', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=1/255)
    parser.add_argument('--adv_eps', type=float, default=16/255)
    parser.add_argument('--attack_version', type=str, default='gaussian_gapgd')
    return parser.parse_args()

def main():
    args = parse_args()

    # config
    with open(os.path.join('configs', args.config), 'r') as f:
        config = dict2namespace(yaml.safe_load(f))
    config.device = "cuda"

    # surrogate CLIP models
    surrogate_names = [
        "laion/CLIP-ViT-G-14-laion2B-s12B-b42K",
        "openai/clip-vit-base-patch32",
        "openai/clip-vit-base-patch16"
    ]
    surrogate_devices = ["cuda"] * 3
    surrogate_dict = load_surrogate_clips(surrogate_names, surrogate_devices)

    df = pd.read_csv("data/imagenet1000.csv")

    for idx, row in df.iterrows():
        if idx >= args.eval_nums:
            break
        print(f"\nProcessing image {idx+1}/{args.eval_nums}")

        clean_img = Image.open(f"/path/to/imagenet/{row['Image Names']}.JPEG").convert('RGB')
        target_img = Image.open(f"/path/to/sd_out/{idx:05d}.png").convert('RGB')

        clean = transform_clean(clean_img).unsqueeze(0).to(config.device)
        target = transform_clean(target_img).unsqueeze(0).to(config.device)

        adv, delta, avg_trH = gapgd_generate(
            args, config, clean, target,
            surrogate_dict, clip_preprocess,
            source_crop, target_crop, transform_clean
        )

        save_dir = f"results/{args.attack_version}"
        os.makedirs(save_dir, exist_ok=True)
        torchvision.utils.save_image(adv, f"{save_dir}/{row['Image Names']}.png")

if __name__ == "__main__":
    main()
