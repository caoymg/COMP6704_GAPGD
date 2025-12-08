import argparse
import logging
import yaml
import os
import time

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import sys
import torchvision

import clip
from transformers import CLIPVisionModel, CLIPProcessor, CLIPModel, CLIPTokenizer
from attack.gapgd_attack import gapgd_generate

# Transform PIL.Image to PyTorch Tensor
def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(
        np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
    )
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

def get_image_embedding(image, surrogate_clip_model):

    image_features = surrogate_clip_model.vision_model(image).last_hidden_state[:, 0, :]
    image_features = surrogate_clip_model.visual_projection(image_features)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    return image_features


def robustness_eval(args, device):


    surrogate_name_lst = ["laion/CLIP-ViT-G-14-laion2B-s12B-b42K", "openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16"]
    surrogate_dict = {}
    for i, surrogate_name in enumerate(surrogate_name_lst):
        temp_dict = {}
        surrogate_clip_model = CLIPModel.from_pretrained(surrogate_name).to(device)  
        surrogate_tokenizer = CLIPTokenizer.from_pretrained(surrogate_name)
        temp_dict["surrogate_clip_model"] = surrogate_clip_model
        temp_dict["surrogate_tokenizer"] = surrogate_tokenizer
        temp_dict["device"] = device
        surrogate_dict[surrogate_name] = temp_dict


    # load data
    # ym: data to image and target caption
    index_df = pd.read_csv("/media/ssd1/cym/imagenet1000.csv")
    all_avg_trH = []
    for i, row in index_df.iterrows():

        if i+1 > args.eval_nums:
            break

        print(f"\nProcessing image {i+1}/{args.eval_nums}")
        
        print("index: ", i)
        clean_image_name = f"ILSVRC2012_val_{row['Image Names']}.JPEG"
        target_image_name = f"{i:05d}.png"
        clean_image = Image.open(f"/path/imageNet-1K_validation/{clean_image_name}").convert('RGB')
        target_image = Image.open(f"/path/{target_image_name}").convert('RGB')
        clean_image = transform_fn(clean_image).unsqueeze(0).to(device)
        target_image = transform_fn(target_image).unsqueeze(0).to(device)
        clean_text = row['Clean Text']
        target_text = row['Target Text']

        adv_image_adp, clean_image, delta, avg_tr_H = gapgd_generate(args, device, clean_image, target_image, clean_text, target_text,
                    surrogate_dict)
        
        folder_to_save = f"./temp/temp_comp6704/{args.attack_version}"
        os.makedirs(folder_to_save, exist_ok=True)

        torchvision.utils.save_image(
                adv_image_adp, os.path.join(folder_to_save, f"{args.attack_version}_"+row['Image Names']) + ".png"
            )
        
        folder_to_save_clean_delta = f"./temp/temp_comp6704/{args.attack_version}/clean_delta"
        os.makedirs(folder_to_save_clean_delta, exist_ok=True)
        torchvision.utils.save_image(
                clean_image, os.path.join(folder_to_save_clean_delta, f"{args.attack_version}_"+row['Image Names']+"_clean") + ".png"
            )
        delta_vis = (delta - delta.min()) / (delta.max() - delta.min())
        torchvision.utils.save_image(
                delta_vis, os.path.join(folder_to_save_clean_delta, f"{args.attack_version}_"+row['Image Names']+"_delta") + ".png"
            )


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    parser.add_argument('--data_seed', type=int, default=2, help='Random seed')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    # adv
    parser.add_argument('--domain', type=str, default='imagenet', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='cifar10-resnet-50', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=64)
    parser.add_argument('--attack_type', type=str, default='apgd-ce')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    # parser.add_argument('--attack_version', type=str, default='custom')

    parser.add_argument('--num_sub', type=int, default=23, help='imagenet subset')
    # parser.add_argument('--adv_eps', type=float, default=0.031373)

    # ym
    parser.add_argument('--eval_nums', type=float, default=100)
    parser.add_argument('--attack_iter', type=float, default=50)
    parser.add_argument('--alpha', type=float, default=1/255)
    parser.add_argument('--adv_eps', type=float, default=16/255)
    parser.add_argument('--attack_version', type=str, default='comp6704_gaussian_alpha09', choices=['noised3_x0_ii_alpha03_10t_1dt_con', "noised3_x0_ii_alpha03_10t_1dt_con_K5"])

    args = parser.parse_args()

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args


if __name__ == '__main__':
    args = parse_args_and_config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids


    transform_fn = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
        ]
    )

    clip_preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224, interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
            torchvision.transforms.Lambda(lambda img: torch.clamp(img, 0.0, 1.0)),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)), # CLIP imgs mean and std.
        ]
    )

    source_crop = (
        transforms.RandomResizedCrop(224, scale=[0.5, 0.9])
    )
    target_crop = (
        transforms.RandomResizedCrop(224, scale=[0.5, 0.9])
    )

    device = "cuda:5"

    robustness_eval(args, device)


