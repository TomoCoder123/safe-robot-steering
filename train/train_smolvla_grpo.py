import torch
from torch import nn
import numpy as np
import torch.serialization
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from LIBERO.libero.libero import benchmark
import os
from LIBERO.libero.libero.envs import OffScreenRenderEnv

# from libero README getting started
def make_libero_env():
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()

    

    # retrieve a specific task
    task_id = 0
    task = task_suite.get_task(task_id)
    task_description = task.language
    print(benchmark.get_libero_path("bddl_files"))
    task_bddl_file = os.path.join(benchmark.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 128,
        "camera_widths": 128
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    init_state_id = 0
    env.set_init_state(init_states[init_state_id])
    return env

#    dummy_action = [0.] * 7
#    for step in range(10):
#        obs, reward, done, info = env.step(dummy_action)
#        q = env.sim.data.qvel.copy()
#        print(q)
#    env.close()

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
    
    env = make_libero_env()
    
    policy.eval()

    env = make_libero_env()
    obs, info = env.reset()
    
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