Project: Gaussian-Augmented PGD with Hutchinson Curvature Tracking
Purpose:
This project implements a Gaussian-augmented PGD adversarial attack using multi-surrogate CLIP models. During optimization, the algorithm estimates the trace of the Hessian (tr(H)) using Hutchinson’s method to analyze the curvature of the loss landscape. The code is modularized into separate files for clarity.

Directory Structure

project_root/
main.py
config/
imagenet.yml
attacks/
gaussian_step.py
curvature.py
surrogate/
surrogate_manager.py
clip_utils.py
utils/
common.py
image_ops.py
readme.txt

File Responsibilities

main.py

Entry point of the attack pipeline

Parses arguments and loads config

Loads CLIP surrogate models

Executes the evaluation loop

Saves adversarial results and curvature logs

attacks/gaussian_step.py

Implements the Gaussian-augmented loss

Computes original similarity and K=3 noisy similarities

Supports adjustable alpha

Returned value is the final loss for backward

attacks/curvature.py

Computes gradient of the loss

Implements Hutchinson trace estimator for tr(H)

Records trace values each epoch

Returned value is tr(H)

surrogate/surrogate_manager.py

Loads multiple CLIP models

Stores tokenizer, device, target embeddings

Provides interface to compute image embeddings

surrogate/clip_utils.py

Helper functions for CLIP preprocessing

get_image_embedding() implementation

Preprocessing pipelines for CLIP input

RandomResizedCrop functions

utils/common.py

Configuration parsing

Logging utilities

Random seed initialization

Argument parsing helpers

utils/image_ops.py

Image transforms

Tensor conversion

Saving clean, delta, and adversarial images

How the Attack Works

For each image, initialize delta = 0

For each epoch:
a. Compute loss = alpha * clean_similarity + (1 - alpha) * noisy_similarity
b. Compute first-order gradient of loss w.r.t delta
c. Estimate trace(H) using Hutchinson’s estimator
d. Normalize the gradient
e. Update delta using PGD step with clipping

At the end, save:

adversarial image

clean image

normalized delta visualization

average tr(H) across epochs

Gaussian-Augmented Loss

loss = alpha * sim_x0 + (1 - alpha) * average(sim_noisy)

where sim_x0 is the cosine similarity between adversarial features and target features, and sim_noisy is averaged across 3 noisy samples.

Curvature Estimation

tr(H) is estimated using Hutchinson’s method:

Sample random vector v from {−1, +1}

Compute Hv using second backward

Approximate tr(H) = v^T Hv

Store this value for each epoch and compute average over the attack

This value indicates curvature; a positive trace shows local convexity around the optimization trajectory.

Dependencies

Python 3.9+
PyTorch
Transformers (for CLIP)
Torchvision
Pandas
PIL

Running the Code

python3 main.py
--attack_iter 100
--alpha 0.0039
--adv_eps 0.0627
--eval_nums 23
--attack_version gaussian_alpha09

Output images and logs will be saved under the specified folder in robustness_eval().

Output Files

For each image, the program saves:

adversarial_image.png

clean_image.png

delta_visualization.png

printed average trace(H)

Notes

Multiple CLIP surrogates improve transferability

Gaussian augmentation provides stability and smooths the loss landscape

The recorded tr(H) values can be analyzed to study curvature regularization effects

Increasing the number of Hutchinson samples improves accuracy but slows computation