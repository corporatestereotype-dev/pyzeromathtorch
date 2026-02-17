import torch

def resume_from_checkpoint(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
    return model
