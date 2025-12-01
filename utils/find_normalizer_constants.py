from safetensors.torch import load_file

def obtain_dataset_normalizer_stats():
    file_path = "./policy_preprocessor_step_5_normalizer_processor.safetensors"

    tensors = load_file(file_path, device="cpu")
    # print(f"normalizer constants: {tensors}")
    return tensors