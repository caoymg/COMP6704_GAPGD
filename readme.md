# Project: Gaussian-Augmented PGD with Hutchinson Curvature Tracking

## Purpose

This project implements a Gaussian-augmented PGD adversarial attack using multi-surrogate CLIP models. During optimization, the algorithm estimates the trace of the Hessian (tr(H)) using Hutchinson’s method to analyze the curvature of the loss landscape. The code is modularized into separate files for clarity.

## Directory Structure

```
project_root/
├── attack_generate.py
├── noisy_image_generate.py
└── attacks/
    ├── gaussian_step.py
    └── gapgd_attack.py

```


## File Responsibilities

### `attack_generate.py`
- Entry point of the attack pipeline.

### `gapgd_attack.py`
- Executes the evaluation loop.
- Saves adversarial results and curvature logs.

### `attacks/gaussian_step.py`
- Implements the Gaussian-augmented loss.
- Computes original similarity and K=3 noisy similarities.


### `surrogate/surrogate_manager.py`
- Loads multiple CLIP models.
- Stores tokenizer, device, target embeddings.
- Provides interface to compute image embeddings.


## How the Attack Works

1. For each image, initialize delta = 0.
2. For each epoch:
   a. Compute `loss = alpha * clean_similarity + (1 - alpha) * noisy_similarity`.
   b. Compute first-order gradient of loss w.r.t delta.
   c. Estimate `trace(H)` using Hutchinson’s estimator.
   d. Normalize the gradient.
   e. Update delta using PGD step with clipping.
3. At the end, save:
   - Adversarial image.
   - Clean image.
   - Normalized delta visualization.
   - Average `tr(H)` across epochs.

## Gaussian-Augmented Loss
loss = alpha * sim_x0 + (1 - alpha) * average(sim_noisy)


Where `sim_x0` is the cosine similarity between adversarial features and target features, and `sim_noisy` is averaged across 3 noisy samples.

## Dependencies

- Python 3.9+
- PyTorch
- Transformers (for CLIP)
- Torchvision
- Pandas
- PIL

## Running the Code

```bash
python3 attack_generate.py \
  --attack_iter 100 \
  --alpha 0.0039 \
  --adv_eps 16/255 \
  --eval_nums 100 \
  --attack_version gaussian_alpha03
