# libero_env_runner.py

import os
import torch
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


def make_libero_env(task_suite_name="libero_10", task_id=0, seed=0):
    """
    Creates a LIBERO OffScreenRenderEnv using the benchmark
    dictionary (your preferred method).
    """

    # Load benchmark suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    # Retrieve the task spec
    task = task_suite.get_task(task_id)
    language = task.language


    # BDDL file for this task
    bddl_file = os.path.join(
        benchmark.get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file
    )

    print(f"[INFO] Loading LIBERO Task {task_id} ({task_suite_name})")
    print(f"[INFO] Instruction: {language}")
    print(f"[INFO] BDDL File: {bddl_file}")

    # Create the environment
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_file,
        camera_widths=1284,
        camera_heights=128,
    )

    env.seed(seed)

    # Load deterministic initial states (important for benchmarking)
    init_states = task_suite.get_task_init_states(task_id)
    init_state_id = 0
    env.set_init_state(init_states[init_state_id])
    

    return env, language
