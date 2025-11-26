import torch
from torch import nn
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from LIBERO.libero.libero import benchmark
import os
from LIBERO.libero.libero.envs import OffScreenRenderEnv
import numpy as np

TASK_SUITE_NAME = "libero_10" # long-range tasks

# take before image, move robot around, take after image
def test_libero_env(env, num_steps=50):
    from PIL import Image
    import numpy as np
    obs = env.reset()
    # flip because OpenGL and PIL coordinate systems are upside down of each other
    Image.fromarray(np.flipud(obs["agentview_image"])).save("before.png")
    print(f"Saved before image")
    dummy_action = [1.] * 7
    final_obs = None
    for step in range(num_steps):
        print(f"Step {step}")
        obs, reward, done, info = env.step(dummy_action)
        final_obs = obs
    Image.fromarray(np.flipud(final_obs["agentview_image"])).save("after.png")
    print(f"Saved after image")

"""Implementation based off of what's in libero's README getting started section. This function sets up
the environment for one task within the specified task suite with a random initialization

task_suite_name is one of libero_10, libero_spatial, etc. 
camera_heights and camera_widths determine image resolution of agentview observations per time step"""
def make_libero_env(task_suite_name, task_id, camera_heights=256, camera_widths=256):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    # retrieve a specific task
    task = task_suite.get_task(task_id)
    task_description = task.language
    task_bddl_file = os.path.join(benchmark.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
        f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": camera_heights,
        "camera_widths": camera_widths,
    }
    env = OffScreenRenderEnv(**env_args)
    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    if init_states is not None and len(init_states) > 0:
        random_id = np.random.randint(len(init_states))
        env.set_init_state(init_states[random_id])    
        print(f"Setting rand init state: {init_states[random_id]}")
    env.reset()

    return env

def main():
    # DEBUG
    policy = nn.Linear(768, 1)
#    policy = SmolVLAPolicy.from_pretrained(
#        "HuggingFaceVLA/smolvla_libero")
#    
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    policy.to(device)
#
#    policy.train()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=1e-5,
        betas=(0.9, 0.95)
    )
    
    # during training generate random task ids
    env = make_libero_env(TASK_SUITE_NAME, 0)
    test_libero_env(env)
    env.close()

if __name__ == "__main__":
    main()