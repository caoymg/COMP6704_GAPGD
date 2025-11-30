from transformers import CLIPModel, CLIPTokenizer

def load_surrogate_clips(model_names, devices):
    surrogate_dict = {}
    for i, name in enumerate(model_names):
        model = CLIPModel.from_pretrained(name).to(devices[i % len(devices)])
        tokenizer = CLIPTokenizer.from_pretrained(name)

        surrogate_dict[name] = {
            "surrogate_clip_model": model,
            "surrogate_tokenizer": tokenizer,
            "device": devices[i % len(devices)],
        }
    return surrogate_dict
