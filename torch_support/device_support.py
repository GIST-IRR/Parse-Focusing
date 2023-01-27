import torch

def set_device(device, verbose=True):
    if verbose:
        print(f"Set the device with ID {device} visible")
    device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    return device