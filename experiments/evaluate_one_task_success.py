# run_experiment.py

import torch
from model.smolvla_policy import SmolVLALiberoPolicy
from env.env import make_libero_env, snapshot_obs

TASK_SUITE_NAME = "libero_10"
STEPS = 280 #lerobot libero default
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
    snapshot_obs(obs, "before.png")
    
    # simple rollout. For GRPO we can rollout with no_grad but will need grads when recomputing new model log densities for chosen actions
    for step in range(STEPS):
        with torch.no_grad():
            action = policy.get_action(obs, language)
            action = action.cpu().clone().detach().tolist()[0]
        obs, reward, done, info = env.step(action)
        
        print(f"Step {step} | Reward: {reward:.3f}")
        if done:
            break

    snapshot_obs(obs, "after.png")
    env.close()

if __name__ == "__main__":
    main()
