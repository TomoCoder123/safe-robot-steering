import torch

def print_gpu():
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved  = torch.cuda.memory_reserved()  / 1024**3
    print(f"GPU | allocated: {allocated:.2f} GB | reserved: {reserved:.2f} MB")