# run_experiment.py

import numpy as np
import torch
from model.smolvla_policy import SmolVLALiberoPolicy
from env.env import make_libero_env

ENV_NUM = 1
ACTION_DIM = 7
def main():
    # Load SmolVLA policy
    policy = SmolVLALiberoPolicy(
        "HuggingFaceVLA/smolvla_libero", device="cuda"
    )

    # If you want to train it (e.g., GRPO)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95)
    )

    # Create LIBERO environment
    env, language = make_libero_env()
    

    # Reset


    env.reset()
    

    dummy_actions = [0.] * 7
    for _ in range(5):
        obs, _, _, _ = env.step(dummy_actions)

    # Simple rollout
    for step in range(10):
        action = policy.get_action(obs, language)
        obs, reward, done, info = env.step(action)
        
        print(f"Step {step} | Reward: {reward:.3f}")
        if done:
            break

    env.close()


if __name__ == "__main__":
    main()
