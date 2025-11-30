import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

transform_clean = T.Compose([
    T.Resize(224, interpolation=InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
])

clip_preprocess = T.Compose([
    T.Resize(224, interpolation=InterpolationMode.BICUBIC, antialias=True),
    T.Lambda(lambda img: img.clamp(0.0, 1.0)),
    T.CenterCrop(224),
    T.Normalize((0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711)),
])

source_crop = T.RandomResizedCrop(224, scale=[0.5, 0.9])
target_crop = T.RandomResizedCrop(224, scale=[0.5, 0.9])
