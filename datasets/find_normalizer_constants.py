from safetensors.torch import load_file
import torch

def obtain_dataset_unnormalizer_stats():
    file_path = "./policy_preprocessor_step_5_normalizer_processor.safetensors"

    tensors = load_file(file_path, device="cpu")
    return tensors