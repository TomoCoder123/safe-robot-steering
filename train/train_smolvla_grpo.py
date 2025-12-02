import torch
import numpy as np
import argparse
import copy
from torch.utils.tensorboard import SummaryWriter


from model.smolvla_policy import SmolVLALiberoPolicy
from env.env import make_libero_env

MAX_STEPS = 360
GROUP_SIZE = 4
UPDATE_EPOCHS = 2
GRPO_EPSILON = 0.2

# TODO: logwriting (probably with tensorboard)
# TODO: added autocast so when running things just make sure that isn't causing any errors
# TODO: maybe add debug logging using logging module just for quality of life

def rollout_one_trajectory(env, policy_old, language,  group_num, rollout_idx=None):
    print("-" * 60)
    if rollout_idx is not None:
        print(f"[ROLLOUT {rollout_idx}] Starting rollout for language:")
    else:
        print(f"[ROLLOUT] Starting rollout for language:")
    print(f"  -> {language}")

    obs = env.reset()
    traj_obs, traj_actions, traj_logprobs = [], [], []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                action, log_prob, unsquished_action = policy_old.sample_action(obs, language)

        env_action = action.cpu().numpy()[0]

        print(f"[ROLLOUT {rollout_idx} step {step:03d} GROUP {group_num}] reward={float(total_reward):+.3f} "
              f"action_norm={float(torch.norm(unsquished_action)):.4f} "
              f"logp={float(log_prob):+.4f}")

        traj_obs.append(obs)
        traj_actions.append(unsquished_action.detach())
        traj_logprobs.append(log_prob.detach())

        obs, reward, done, info = env.step(env_action)
        total_reward += float(reward)

        if done:
            print(f"[ROLLOUT] Episode finished early at step {step}")
            break

    print(f"[ROLLOUT DONE] total_reward={total_reward:.3f}")
    print("-" * 60)

    return {
        "obs": traj_obs,
        "actions": traj_actions,
        "logprobs": traj_logprobs,
        "reward": total_reward,
        "language": language,
    }


def sample_group_trajectories(env, policy_old, language, G, group_num):
    print("\n" + "-" * 60)
    print(f"[GROUP] Collecting {G} trajectories for prompt:\n  \"{language}\"")
    print("-" * 60 + "\n")

    rollouts = []
    for i in range(G):
        rollouts.append(rollout_one_trajectory(env, policy_old, language,  group_num, rollout_idx=i + 1))

    rewards = [r["reward"] for r in rollouts]
    print(f"[GROUP SUMMARY] rewards={rewards}")
    print(f"[GROUP SUMMARY] mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}")
    print()

    return rollouts



def compute_group_advantages(trajs):
    rewards = torch.tensor([t["reward"] for t in trajs], dtype=torch.float32)
    
    mean_r = rewards.mean()
    std_r = rewards.std()
    mean_r = torch.nan_to_num(mean_r, nan=0.0) # is nan when there is 0 reward
    std_r = torch.nan_to_num(std_r, nan=0.0)

    advantages = (rewards - mean_r) / (std_r + 1e-8)

    print("[ADVANTAGES] Rewards:", rewards.tolist())
    print("[ADVANTAGES] Advantages:", advantages.tolist())
 

    return advantages


def compute_grpo_loss(policy_theta, policy_old, trajs, advantages, epsilon):
    group_losses = []

    for i, traj in enumerate(trajs):
        A_i = advantages[i]
        step_losses = []
        lang = traj["language"]

        for obs_t, unsquished_action_t, old_lp_t in zip(
            traj["obs"], traj["actions"], traj["logprobs"]
        ):
            new_lp = policy_theta.get_action_prob(obs_t, lang, unsquished_action_t.to(policy_theta.device))
            ratio = torch.exp(new_lp - old_lp_t.to(policy_theta.device))

            unclipped = ratio * A_i
            clipped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * A_i
            step_losses.append(-torch.min(unclipped, clipped))

        traj_loss = torch.stack(step_losses).mean()
        group_losses.append(traj_loss)

    return torch.stack(group_losses).mean()


def collect_batch(env_factory, languages, batch_size, policy_old):
    batch = []
    for group_num in range(batch_size):
        print(f"creating batch group {group_num+1}")
        env, lang = env_factory()  # new task each time
        rollout_group = sample_group_trajectories(env, policy_old, lang, GROUP_SIZE, group_num + 1)
        advantages = compute_group_advantages(rollout_group)
        batch.append((rollout_group, advantages, lang))
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
    policy.set_log_std(-3.0)
    print("Set policy gradients and log std.")
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
        batch = collect_batch(env_factory, None, args.batch_size, policy_old)
        policy.train()

        epoch_losses = []
        for epoch in range(UPDATE_EPOCHS):
            total_loss = 0.0
            for rollout_group, advantages, lang in batch:
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    loss = compute_grpo_loss(policy, policy_old, rollout_group, advantages, GRPO_EPSILON)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.policy.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            epoch_losses.append(total_loss / len(batch))

        avg_loss = np.mean(epoch_losses)
        avg_reward = np.mean([np.mean([t["reward"] for t in g]) for g, _, _ in batch])

        print(f"[Update {update}] loss={avg_loss:.4f}, avg_reward={avg_reward:.3f}")

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
            print(f"[CHECKPOINT] Saved model to {save_path}")



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
    print("Starting train")
    train_grpo(args)


if __name__ == "__main__":
    main()
