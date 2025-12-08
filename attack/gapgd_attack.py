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
from attack.gaussian_step import gaussian_one_step





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


def gapgd_generate(args, device, clean_image, target_image, clean_text, target_text,
                    surrogate_dict):

    delta = torch.zeros_like(clean_image, requires_grad=True).to(device)

    for surrogate_name in surrogate_dict.keys():
        device = surrogate_dict[surrogate_name]["device"]
        surrogate_clip_model = surrogate_dict[surrogate_name]["surrogate_clip_model"]
        with torch.no_grad():
            tgt_image_features = get_image_embedding(target_crop(clip_preprocess(target_image)).to(device), surrogate_clip_model)
            surrogate_dict[surrogate_name]["tgt_image_features"] = tgt_image_features

    trace_record = []

    for epoch in range(args.attack_iter):

        delta = delta.detach()
        delta.requires_grad_(True)
        adv_image = clean_image + delta
        grad_all = torch.zeros_like(delta).to(device)

        loss = - gaussian_one_step(device, adv_image, target_image, surrogate_dict)

        # First derivative (no backward)
        grad = torch.autograd.grad(loss, delta, create_graph=True)[0]

        # Hutchinson trace estimation
        def hutchinson_trace(delta, grad, num_samples=1):
            trace_estimates = []
            for _ in range(num_samples):
                v = torch.randint_like(delta, 0, 2).float()
                v = 2 * v - 1
                Hv = torch.autograd.grad(
                    outputs=(grad * v).sum(),
                    inputs=delta,
                    retain_graph=True
                )[0]
                trace_estimates.append((Hv * v).sum().item())
            return sum(trace_estimates) / len(trace_estimates)
        
        tr_H = hutchinson_trace(delta, grad)
        trace_record.append(tr_H)

        # Update PGD step
        grad = grad.detach()
        grad_all += grad

        grad_norm = grad_all / (grad_all.abs().mean(dim=(1,2,3), keepdim=True))
        delta = torch.clamp(delta - args.alpha * grad_norm, -args.adv_eps, args.adv_eps).detach()

        if epoch % 10 == 0:
            print(f"epoch {epoch}, loss {loss.item()}")
    # Create final adversarial image
    adv_image = clean_image + delta
    adv_image = torch.clamp(adv_image, 0.0, 1.0)
    avg_tr_H = sum(trace_record) / len(trace_record)
    # print("Average trace(H):", avg_tr_H)
    return adv_image, clean_image, delta, avg_tr_H

