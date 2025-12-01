from safetensors.torch import load_file

def obtain_dataset_unnormalizer_stats():

    file_path = "./policy_postprocessor_step_1_unnormalizer_processor.safetensors"

    tensors = load_file(file_path, device="cpu")
    # print(f"unnormalizer constants: {tensors}")
    return tensors