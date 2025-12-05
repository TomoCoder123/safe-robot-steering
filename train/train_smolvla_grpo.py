import torch
import numpy as np
import argparse
import copy
from torch.utils.tensorboard import SummaryWriter
from utils.print_gpu import print_gpu

from model.smolvla_policy import SmolVLALiberoPolicy
from env.env import make_libero_env
import logging

MAX_STEPS = 520
GROUP_SIZE = 4
UPDATE_EPOCHS = 2
UPDATE_CHUNK_SIZE = 20
EULER_STEP_NOISE_STD = 0.25
INIT_LOG_STD = -2
GRPO_EPSILON = 0.2

# TODO: change to logging module for speed/qol



logger = logging.getLogger("rollout")
logger.setLevel(logging.INFO) 
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(name)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def rollout_one_trajectory(env, policy_old, language,  group_num, rollout_idx=None):
    logger.info("-" * 60)
    if rollout_idx is not None:
        logger.info(f"[ROLLOUT {rollout_idx}] Starting rollout for language:")
    else:
        logger.info(f"[ROLLOUT] Starting rollout for language:")
    logger.info(f"  -> {language}")

    obs = env.reset()
    traj_obs, traj_actions, traj_logprobs = [], [], []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                action, log_prob, unsquished_action = policy_old.sample_action(obs, language)

        env_action = action.cpu().numpy()[0]

        logger.info(f"[ROLLOUT {rollout_idx} step {step:03d} GROUP {group_num}] reward={float(total_reward):+.3f} "
              f"action_norm={float(torch.norm(unsquished_action)):.4f} "
              f"logp={float(log_prob):+.4f}")

        traj_obs.append(obs)
        traj_actions.append(unsquished_action.detach())
        traj_logprobs.append(log_prob.detach())

        obs, reward, done, info = env.step(env_action)
        total_reward += float(reward)

        if done:
            logger.info(f"[ROLLOUT] Episode finished early at step {step}")
            break

    logger.info(f"[ROLLOUT DONE] total_reward={total_reward:.3f}")
    logger.info("-" * 60)

    return {
        "obs": traj_obs,
        "actions": traj_actions,
        "logprobs": traj_logprobs,
        "reward": total_reward,
        "language": language,
    }


def sample_group_trajectories(env, policy_old, language, G, group_num):
    logger.info("\n" + "-" * 60)
    logger.info(f"[GROUP] Collecting {G} trajectories for prompt:\n  \"{language}\"")
    logger.info("-" * 60 + "\n")

    rollouts = []
    for i in range(G):
        rollouts.append(rollout_one_trajectory(env, policy_old, language,  group_num, rollout_idx=i + 1))

    rewards = [r["reward"] for r in rollouts]
    logger.info(f"[GROUP SUMMARY] rewards={rewards}")
    logger.info(f"[GROUP SUMMARY] mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
    logger.info()

    return rollouts



def compute_group_advantages(trajs):
    rewards = torch.tensor([t["reward"] for t in trajs], dtype=torch.float32)
    
    mean_r = rewards.mean()
    std_r = rewards.std()
    mean_r = torch.nan_to_num(mean_r, nan=0.0) # is nan when there is 0 reward
    std_r = torch.nan_to_num(std_r, nan=0.0)

    advantages = (rewards - mean_r) / (std_r + 1e-8)

    logger.info("[ADVANTAGES] Rewards:", rewards.tolist())
    logger.info("[ADVANTAGES] Advantages:", advantages.tolist())
 

    return advantages


def compute_grpo_loss(policy_theta, trajs, advantages, epsilon, timestep_chunk_size=10):
    """
    Compute GRPO loss with gradient accumulation over timestep chunks to save memory.
    
    Args:
        policy_theta: training policy
        trajs: list of trajectories
        advantages: tensor of advantages for each trajectory
        epsilon: clipping parameter
        timestep_chunk_size: number of timesteps to process per trajectory before calling backward
    """
    all_step_losses = []
    num_trajs = len(trajs)

    for i, traj in enumerate(trajs):
        A_i = advantages[i]
        prompt = traj["language"]
        
        obs_list = traj["obs"]
        unsq_list = traj["actions"]
        old_lp_list = traj["logprobs"]
        total_timesteps = len(obs_list)
        
        # Process timesteps in chunks
        for chunk_start in range(0, total_timesteps, timestep_chunk_size):
            chunk_end = min(chunk_start + timestep_chunk_size, total_timesteps)
            chunk_obs = []
            unsquished_actions = []
            old_lp_chunk = []
            for t in range(chunk_start, chunk_end):
                chunk_obs.append(obs_list[t]) 
                unsquished_actions.append(unsq_list[t])
                old_lp_chunk.append(old_lp_list[t])

            unsquished_actions = torch.stack(unsquished_actions)
            old_log_probs = torch.stack(old_lp_chunk)
            new_log_probs = policy_theta.get_action_probs(chunk_obs, prompt, unsquished_actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            unclipped = ratio * A_i
            clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * A_i
            step_losses = -torch.min(unclipped, clipped)
            
            # Compute mean loss for this timestep chunk and backpropagate. Divide by (total_timesteps *)
            # Divide by (total_timesteps * num_trajs) so gradients are scaled to the overall average (since in GRPO
            # we average over rollout length and group size)
            chunk_loss = step_losses.mean() / (total_timesteps * num_trajs)
            chunk_loss.backward()
            # test grads being accumd when force advantages to non zero
            
            # Collect losses for final averaging (detach to avoid keeping graph)
            all_step_losses.append(step_losses.detach())
            
            # Clear unused memory
            del chunk_loss, step_losses

    # Average over all timesteps across all trajectories
    all_losses = torch.cat(all_step_losses)
    avg_loss = all_losses.mean().item()
    
    return avg_loss

def collect_batch(env_factory, batch_size, policy_old):
    batch = []
    for group_num in range(batch_size):
        print(f"creating batch group {group_num+1}")
        env, lang = env_factory()  # new task each time
        rollout_group = sample_group_trajectories(env, policy_old, lang, GROUP_SIZE, group_num + 1)
        advantages = compute_group_advantages(rollout_group)
        batch.append((rollout_group, advantages))
        env.close()
    return batch

# freeze everything but lm_expert and the added log_std parameter
def set_up_policy_grads(policy):
    for param in policy.policy.parameters():
        param.requires_grad = False
    for param in policy.policy.model.vlm_with_expert.lm_expert.parameters():
        param.requires_grad = True
    policy.policy.model.log_std.requires_grad = True

def train_grpo(args):
    writer = SummaryWriter(log_dir="runs/grpo_smolvla")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = SmolVLALiberoPolicy("HuggingFaceVLA/smolvla_libero", device=device)
    set_up_policy_grads(policy)
    policy.set_log_std(INIT_LOG_STD)
    policy.set_euler_step_noise_std(EULER_STEP_NOISE_STD)
    logger.info("Set policy gradients and log std.")
    trainable_params = [p for p in policy.policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95)
    )

    def env_factory():
        return make_libero_env(args.task_suite)

    policy_old = copy.deepcopy(policy)
    policy_old.eval()

    for update in range(args.num_updates):
        batch = collect_batch(env_factory, args.batch_size, policy_old)
        policy.train()

        epoch_losses = []
        for epoch in range(UPDATE_EPOCHS):
            total_loss = 0.0
            for rollout_group, advantages in batch:
                optimizer.zero_grad()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = compute_grpo_loss(policy, rollout_group, advantages, GRPO_EPSILON, timestep_chunk_size=UPDATE_CHUNK_SIZE)
                # backward() is now called inside compute_grpo_loss to accumulate gradients
                torch.nn.utils.clip_grad_norm_(policy.policy.parameters(), 1.0)
                optimizer.step()
                total_loss += loss

            epoch_losses.append(total_loss / len(batch))

        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean([np.mean([t["reward"] for t in g]) for g, _ in batch])

        logger.info(f"[Update {update}] loss={avg_loss:.4f}, avg_reward={avg_reward:.3f}")

        writer.add_scalar("Loss/update", avg_loss, update)
        writer.add_scalar("Reward/update", avg_reward, update)

        policy_old_state_dict = {k: v.clone() for k, v in policy.policy.state_dict().items()}
        policy_old.policy.load_state_dict(policy_old_state_dict)
        policy_old.eval()
        if (update + 1) % args.save_every == 0:
            save_path = f"{args.save_dir}/policy_update_{update+1}.pt"
            torch.save({
                "policy_state_dict": policy.policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update,
                "avg_loss": avg_loss,
                "avg_reward": avg_reward,
                "args": vars(args),
            }, save_path)
            logger.info(f"[CHECKPOINT] Saved model to {save_path}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-suite", type=str, default="libero_10")
    parser.add_argument("--batch-size", type=int, default=1)     # number of tasks per update
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-updates", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default = "checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()
    logger.info("Starting train")
    train_grpo(args)


if __name__ == "__main__":
    main()