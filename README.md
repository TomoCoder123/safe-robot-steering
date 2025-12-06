# Steering Robots Safely
<img width="1234" height="772" alt="ArchitectureDiagram" src="https://github.com/user-attachments/assets/0890dcef-9681-41d4-9a15-9365b5814249" />

# Project Setup and Usage

This README describes how to set up the environment, install dependencies, configure paths, and run TensorBoard locally and on Google Cloud.

---

## Installation

```bash
poetry install
git submodule update --init --recursive
```

---

## Environment Variables

```bash
export MUJOCO_GL=osmesa
export LD_PRELOAD=/lib/x86_64-linux-gnu/libstdc++.so.6

export LIBERO_CONFIG_PATH=/home/navlab-aa290/tdeng/safe-robot-steering/libero/libero
export CUDA_VISIBLE_DEVICES=1

source ~/.bashrc
```

---

## Add Local `lerobot` Package

From the project root:

```bash
poetry add lerobot-local/
```

---

## Set Project Root for Python

```bash
export PYTHONPATH="$HOME/safe-robot-steering:$PYTHONPATH"
```

This allows Python to find the `safe-robot-steering` modules.

---

## TensorBoard (Local)

To run TensorBoard locally:

```bash
tensorboard --logdir=runs/grpo_smolvla_tb --bind_all --port 6006
```

Then open in your browser:

```text
http://localhost:6006
```

---

## TensorBoard on Cloud (GCP)

### 1. Start TensorBoard on the Cloud VM (inside `tmux`)

```bash
tensorboard --logdir runs --bind_all --port 6006
```

### 2. SSH Tunnel From Local Machine

```bash
gcloud compute ssh --zone "<zone>" "<instancename>" --project ">project_name>" --ssh-flag "-L 6006:localhost:6006"
```

### 3. Access TensorBoard in Browser

```text
http://localhost:6006
```

TensorBoard will be forwarded from the VM to your local machine.
