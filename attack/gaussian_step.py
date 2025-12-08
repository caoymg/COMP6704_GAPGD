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

def get_image_embedding(image, surrogate_clip_model):

    image_features = surrogate_clip_model.vision_model(image).last_hidden_state[:, 0, :]
    image_features = surrogate_clip_model.visual_projection(image_features)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)

    return image_features

def gaussian_one_step(device, adv_image, target_image, surrogate_dict):

    total_loss = []
    for surrogate_name in surrogate_dict.keys():
        device = surrogate_dict[surrogate_name]["device"]
        surrogate_clip_model = surrogate_dict[surrogate_name]["surrogate_clip_model"]
        tgt_image_features = surrogate_dict[surrogate_name]["tgt_image_features"]
        adv_image_features_x0 = get_image_embedding(source_crop(clip_preprocess(adv_image)).to(device), surrogate_clip_model)
        # adv_image_features = get_image_embedding(source_crop(clip_preprocess(adv_image)).to(device), surrogate_clip_model)
        tar_sim_x0 = torch.mean(torch.sum(adv_image_features_x0 * tgt_image_features, dim=1))  # cos. sim
        # tar_sim = torch.mean(torch.sum(adv_image_features * tgt_image_features, dim=1))  # cos. sim
        sigma_noise = 16/255
        noise = torch.randn_like(adv_image) * sigma_noise
        noised_adv_image_features_1 = get_image_embedding(source_crop(clip_preprocess(adv_image)+noise).to(device), surrogate_clip_model)
        noise = torch.randn_like(adv_image) * sigma_noise
        noised_adv_image_features_2 = get_image_embedding(source_crop(clip_preprocess(adv_image)+noise).to(device), surrogate_clip_model)
        noise = torch.randn_like(adv_image) * sigma_noise
        noised_adv_image_features_3 = get_image_embedding(source_crop(clip_preprocess(adv_image)+noise).to(device), surrogate_clip_model)

        noised_tar_sim_1 = torch.mean(torch.sum(noised_adv_image_features_1 * tgt_image_features, dim=1))  # cos. sim
        noised_tar_sim_2 = torch.mean(torch.sum(noised_adv_image_features_2 * tgt_image_features, dim=1))  # cos. sim
        noised_tar_sim_3 = torch.mean(torch.sum(noised_adv_image_features_3 * tgt_image_features, dim=1))  # cos. sim
        
        noised_avg_tar_sim = (noised_tar_sim_1 + noised_tar_sim_2 + noised_tar_sim_3)/3   # K3

        alpha = 0.3 # original

        loss = alpha*tar_sim_x0 + (1-alpha)*noised_avg_tar_sim

        total_loss.append((loss).to(device))
    
    loss = sum(total_loss)/len(total_loss)
    
    return loss