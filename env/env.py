from LIBERO.libero.libero import benchmark
from LIBERO.libero.libero.envs import OffScreenRenderEnv
import numpy as np
import multiprocessing as mp
import os
import gymnasium

mp.set_start_method("spawn", force=True) # avoid deadlocks form fork
# these are just defined by the task formulation
ACTION_SPACE = gymnasium.spaces.Box(
    low=-1.0, high=1.0, shape=(7,), dtype=np.float32
)
OBSERVATION_SPACE = gymnasium.spaces.Dict({
    "images": gymnasium.spaces.Box(
        low=0, high=255, shape=(224, 224, 3), dtype=np.uint8
    ),
    "prompts": gymnasium.spaces.Text(max_length=1000),
    "proprioception": gymnasium.spaces.Box(
        low=-np.inf, high=np.inf, shape=(39, ), dtype=np.float64
    )
})

# wrapper to store sub environment info (task language/id), set some instance variables that 
# gymnasium's vectorization requires, convert LIBERO sub environment outputs into what SmolVLA needs
class VectorizedOffScreenEnv(OffScreenRenderEnv):
    def __init__(self, task, task_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_language = task.language
        self.task_id = task_id
        self.render_mode = None
        self.metadata = {} # expected by gynamsium vectorization
        self.action_space = ACTION_SPACE
        self.observation_space = OBSERVATION_SPACE

# sub environment creation callable for gymnasium vectorization 
def libero_factory(task_suite_name, task_id=None, camera_heights=256, camera_widths=256):
    def _thunk():
        env = make_libero_env(task_suite_name, task_id, camera_heights, camera_widths, vectorized=True)
        return env
    return _thunk

# take before image of environment, move robot around, take after image
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

task_id is an optional int. If left as None, a random task_id will be chosen
task_suite_name is one of libero_10, libero_spatial, etc. 
camera_heights and camera_widths determine image resolution of agentview observations per time step"""
def make_libero_env(task_suite_name, task_id=None, camera_heights=256, camera_widths=256, vectorized=False, seed=None):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    # retrieve a specific task
    if task_id is None:
        task_id = np.random.randint(task_suite.n_tasks)
        print(f"Setting random task")
    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(benchmark.get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

    print(f"[INFO] Loading LIBERO Task {task_id} ({task_suite_name})")
    print(f"[INFO] Instruction: {task.language}")
    print(f"[INFO] BDDL File: {task_bddl_file}")

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": camera_heights,
        "camera_widths": camera_widths,
    }
    if vectorized:
        env = VectorizedOffScreenEnv(task, task_id, **env_args)
    else:
        env = OffScreenRenderEnv(**env_args)
    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
    if init_states is not None and len(init_states) > 0:
        random_id = np.random.randint(len(init_states))
        env.set_init_state(init_states[random_id])    
        print(f"Setting rand init state: {init_states[random_id]}")

    if seed:
        env.seed(seed)
    env.reset()

    return env, task.language
