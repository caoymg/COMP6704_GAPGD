import torch
import torchvision
from attack.gaussian_step import gaussian_one_step
from attack.trace import hutchinson_trace

def gapgd_generate(args, config, clean_image, target_image, surrogate_dict,
                       clip_preprocess, source_crop, target_crop, transform_clean):

    # initialize perturbation
    delta = torch.zeros_like(clean_image, requires_grad=True).to(config.device)

    # precompute target image embeddings
    for name, sd in surrogate_dict.items():
        device = sd["device"]
        model = sd["surrogate_clip_model"]
        with torch.no_grad():
            tgt_feat = gaussian_one_step.get_image_embedding(target_crop(clip_preprocess(target_image)).to(device), model)
            sd["tgt_image_features"] = tgt_feat

    trace_record = []

    for epoch in range(args.attack_iter):
        delta = delta.detach().requires_grad_(True)
        adv_image = clean_image + delta

        loss = gaussian_one_step(adv_image, surrogate_dict,
                                 clip_preprocess, source_crop,
                                 sigma=16/255, alpha=0.3,
                                 loss_device=config.device)

        grad = torch.autograd.grad(loss, delta, create_graph=True)[0]
        trH = hutchinson_trace(delta, grad)
        trace_record.append(trH)

        # PGD update (scaled gradient)
        grad_norm = grad.detach() / grad.detach().abs().mean(dim=(1,2,3), keepdim=True)
        delta = (delta + args.alpha * grad_norm).clamp(-args.adv_eps, args.adv_eps)

        if epoch % 30 == 0:
            print(f"[{epoch}] loss={loss.item():.4f}, tr(H)={trH:.4f}")

    adv_image = (clean_image + delta).clamp(0, 1)
    return adv_image, delta, sum(trace_record)/len(trace_record)
