import torch
import numpy as np
from PIL import Image

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())


def get_image_embedding(image, surrogate_clip_model):
    image_features = surrogate_clip_model.vision_model(image).last_hidden_state[:, 0, :]
    image_features = surrogate_clip_model.visual_projection(image_features)
    return image_features / image_features.norm(dim=1, keepdim=True)
