from safetensors.torch import load_file
import torch

def obtain_dataset__normalizer_stats():

    file_path = "./policy_postprocessor_step_1_unnormalizer_processor.safetensors"

    tensors = load_file(file_path, device="cpu")

    return tensors