# run_experiment.py

import torch
from model.smolvla_policy import SmolVLALiberoPolicy
from env.env import make_libero_env

TASK_SUITE_NAME = "libero_10"

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

    env, language = make_libero_env(TASK_SUITE_NAME)
    print(f"Task description: {language}")
    obs = env.reset()
    
    # simple rollout
    for step in range(10):
        action = policy.get_action(obs, language)
        action = action.clone().cpu()[0]
        obs, reward, done, info = env.step(action)
        
        print(f"Step {step} | Reward: {reward:.3f}")
        if done:
            break

    env.close()

if __name__ == "__main__":
    main()
