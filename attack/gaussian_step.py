import torch
from clip_utils.image_embedding import get_image_embedding

def gaussian_one_step(adv_image, surrogate_dict, clip_preprocess, source_crop, sigma=16/255, alpha=0.3, loss_device="cuda"):
    losses = []

    for name, sd in surrogate_dict.items():
        device = sd["device"]
        model = sd["surrogate_clip_model"]
        tgt = sd["tgt_image_features"]

        # x0 similarity
        x0_feat = get_image_embedding(source_crop(clip_preprocess(adv_image)).to(device), model)
        sim_x0 = torch.mean(torch.sum(x0_feat * tgt, dim=1))

        # Gaussian augmentations (K=3)
        sims = []
        for _ in range(3):
            noise = torch.randn_like(adv_image) * sigma
            feat = get_image_embedding(source_crop(clip_preprocess(adv_image + noise)).to(device), model)
            sims.append(torch.mean(torch.sum(feat * tgt, dim=1)))

        sim_noised = sum(sims) / 3

        loss = alpha * sim_x0 + (1 - alpha) * sim_noised
        losses.append(loss.to(loss_device))

    return sum(losses) / len(losses)
