import torch
from torch import nn
import numpy as np
# torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
# from gymnasium.vector import AsyncVectorEnv
# from env import libero_factory
from env.env import make_libero_env

# TODO: if want to vectorize envs, finish that by figuring out how to map LIBERO observations to SmolVLA compatible ones in env.py
# TODO: but probably for now, get things working with one environment first. Need to figure out SmolVLA input/output format exactly
# (may expect multiple camera views, different amount of joints, etc.)

TASK_SUITE_NAME = "libero_10" # long-range tasks
NUM_ENVS = 8

def main():
    # DEBUG
    # policy = nn.Linear(768, 1)
    policy = SmolVLAPolicy.from_pretrained(
       "HuggingFaceVLA/smolvla_libero")
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy.to(device)

    # policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95)
    )
    
    # switch task_id to random generation during training
    env = make_libero_env(TASK_SUITE_NAME)
    # factories = [libero_factory(TASK_SUITE_NAME) for _ in range(NUM_ENVS)]
    # vec_env = AsyncVectorEnv(factories)
    # print(vec_env)
    
    policy.eval()

    obs = env.reset()
    
    for t in range(50):
        pixel_values = obs.pixel_values.to(device)
        prompts = obs.prompts  # list of strings
        
        with torch.no_grad():
            action_output = policy(pixel_values=pixel_values, prompts=prompts)
        
        actions = action_output["actions"].cpu().numpy()
        
        obs, rewards, dones, truncated, info = env.step(actions)
        
        if dones.any():
            print("Episode ended:", dones)
            break

if __name__ == "__main__":
    main()
