# run_experiment.py

import torch
from model.smolvla_policy import SmolVLALiberoPolicy
from env.env import make_libero_env, snapshot_obs, get_agentview_frame, get_wrist_frame
import imageio
import os
FPS = 60

TASK_SUITE_NAME = "libero_spatial" 
STEPS = 280 #lerobot libero default for object, 520 is default for long
def main():
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())

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
    
    agentview_path = "agentview.mp4"
    wristcam_path = "wristcam.mp4"

    agent_writer = imageio.get_writer(agentview_path, fps=FPS)
    wrist_writer = imageio.get_writer(wristcam_path, fps=FPS)

    print(f"Recording agentview -> {agentview_path}")
    print(f"Recording wristcam -> {wristcam_path}")
    
    # Write first frames
    agent_writer.append_data(get_agentview_frame(obs))
    wrist_writer.append_data(get_wrist_frame(obs))
    
    # simple rollout. For GRPO we can rollout with no_grad but will need grads when recomputing new model log densities for chosen actions
    for step in range(STEPS):
        with torch.no_grad():
            action = policy.get_action(obs, language)
            action = action.cpu().clone().detach().tolist()[0]
        obs, reward, done, info = env.step(action)
        
        print(f"Step {step} | Reward: {reward:.3f}")
            
        agent_writer.append_data(get_agentview_frame(obs))
        wrist_writer.append_data(get_wrist_frame(obs))
        if done:
            break

    snapshot_obs(obs, "after.png")
    env.close()

if __name__ == "__main__":
    main()
