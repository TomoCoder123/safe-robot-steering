# run_experiment.py

import torch
from model.smolvla_policy import SmolVLALiberoPolicy
from env.env import make_libero_env, snapshot_obs, get_agentview_frame, get_wrist_frame
import imageio
import os
FPS = 60

TASK_SUITE_NAME = "libero_spatial" 
STEPS = 150 #lerobot libero default for object, 520 is default for long
def main():
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count():", torch.cuda.device_count())

    # Load SmolVLA policy
    policy = SmolVLALiberoPolicy(
        "HuggingFaceVLA/smolvla_libero", device="cuda"
    )

    env, language = make_libero_env(TASK_SUITE_NAME, task_id=1)
    print(f"Task description: {language}")
    obs = env.reset()

    _, log_prob, _ = policy.sample_action(obs, language)
    print(f"Log probability gradient function: {log_prob.grad_fn}")
    assert log_prob.grad_fn is not None, "The log probability should definitely have a gradient function"
    # policy.set_log_std(-10)
    distr = policy.get_action_distr(obs, language)
    print(f"Sample distribution mean: {distr.loc}, std: {distr.scale}")

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
    
    # because we're treating pretrained action outputs as means, treating the means from the GRPO pathway as actions
    # without use in distributions should lead to the same behavior as just choosing actions 
    for step in range(STEPS):
        with torch.no_grad():
            # action, _ = policy.get_action_distr_params(obs, language)
            action, _, _ = policy.sample_action(obs, language)
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
