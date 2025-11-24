
import torch
import torch.nn as nn
import json
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

def main():
    policy = SmolVLAPolicy.from_pretrained(
        "HuggingFaceVLA/smolvla_libero")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95)
    )
    
    
    env = make_libero_env("libero_spatial")


if __name__ == "__main__":
    main()