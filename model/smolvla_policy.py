# smolvla_policy.py

import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
import transforms3d


class SmolVLALiberoPolicy:
    """
    Adapter that converts LIBERO's obs format to the LeRobot SmolVLA format.

    LIBERO obs keys:
        agentview_rgb      -> main camera
        eye_in_hand_rgb    -> wrist camera
        joint_states       -> 7D
        gripper_states     -> 2D (take 1D)
    """

    def __init__(self, model_name="HuggingFaceVLA/smolvla_libero", device="cuda"):
        print(f"[SmolVLA] Loading pretrained model: {model_name}")

        self.device = device
        self.policy = SmolVLAPolicy.from_pretrained(model_name)
        self.policy.to(device)
        self.policy.eval()
        self.parameters = self.policy.parameters 

    def _extract_images(self, obs):
        img1 = obs["agentview_image"]       # (H,W,3)
        img2 = obs["robot0_eye_in_hand_image"]     # (H,W,3)

        # Convert to torch CHW
        img1 = torch.from_numpy(img1).permute(2, 0, 1)   # (3,H,W)
        img2 = torch.from_numpy(img2).permute(2, 0, 1)
        
        img1 = img1.float() / 255.0
        img2 = img2.float() / 255.0

        return img1, img2


    def _extract_state(self, obs):
        pos = obs["robot0_eef_pos"]   # (3,)

        quat = obs["robot0_eef_quat"]  # (4,)
        roll, pitch, yaw = transforms3d.euler.quat2euler(quat)  # matches LIBERO conventions

        # 3. Gripper (2)
        grip = obs["robot0_gripper_qpos"]  # (2,)

        # 4. Final 8-dim state
        state = np.array([
            pos[0], pos[1], pos[2],
            roll, pitch, yaw,
            grip[0], grip[1]
        ], dtype=np.float32)

        return torch.from_numpy(state).float()

    def _build_batch(self, obs, language):
        # images
        img1, img2 = self._extract_images(obs)
        img1 = img1.unsqueeze(0).to(self.device)
        img2 = img2.unsqueeze(0).to(self.device)

        # state
        state = self._extract_state(obs)
        state = state.unsqueeze(0).to(self.device)

        batch = {
            # FLATTENED KEYS HERE
            "observation.images.image": img1,
            "observation.images.image2": img2,
            "observation.state": state,
        }
        

        return batch


    @torch.no_grad()
    def get_action(self, obs, language):
        batch = self._build_batch(obs, language)
        out = self.policy(batch)

        # SmolVLA returns either "action" or "actions"
        if "action" in out:
            action = out["action"][0].cpu().numpy()
        else:
            action = out["actions"][0].cpu().numpy()

        # LIBERO requires action in [-1, 1]
        action = np.clip(action, -1.0, 1.0)

        return action
    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()
