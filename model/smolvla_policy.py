# smolvla_policy.py

import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from torchvision.transforms import v2
from transformers import AutoTokenizer
import transforms3d

# NOTE: there are these already-configured pre/post processing pipelines in lerobot but they're very convoluted and buggy.
# I just referenced how they do things and made it so we do each step within our own mapping code from LIBERO observation -> model input
# from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

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
        self.img_transform = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        # tokenizer as determined by the tokenization step of the preprocessing pipeline defined in make_smolvla_pre_post_processors
        vla_config = SmolVLAConfig()
        self.max_token_length = vla_config.tokenizer_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
        )

    def _extract_images(self, obs):
        agentview_img = obs["agentview_image"]       # (H,W,3)
        eye_img = obs["robot0_eye_in_hand_image"]     # (H,W,3)

        agentview_img = self.img_transform(agentview_img)
        eye_img = self.img_transform(eye_img)

        return agentview_img, eye_img

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

    def _tokenize_prompt(self, task_description):
        if not task_description.endswith("\n"): # model expects \n at end of each prompt
            task_description = f"{task_description}\n"

        tokenized = self.tokenizer(
            [task_description],  
            max_length=self.max_token_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        tokens = tokenized["input_ids"]  # (1, max_token_length)
        attn_mask = tokenized["attention_mask"].bool() # (1, max_token_length)
        return tokens, attn_mask

    def _build_batch(self, obs, task_description):
        # images
        agentview_img, eye_img = self._extract_images(obs)
        agentview_img = agentview_img.unsqueeze(0).to(self.device)
        eye_img = eye_img.unsqueeze(0).to(self.device)

        # state
        state = self._extract_state(obs)
        state = state.unsqueeze(0).to(self.device)

        # task language prompt
        tokens, attn_mask = self._tokenize_prompt(task_description)
        tokens = tokens.to(self.device)
        attn_mask = attn_mask.to(self.device)
       
        # the names for the keys must match what's specified in the SmolVLAConfig dataclass
        return {
            "observation.images.image1": agentview_img,
            "observation.images.image2": eye_img,
            "observation.state": state,
            "observation.language.tokens": tokens,
            "observation.language.attention_mask": attn_mask,
        }

    def get_action(self, obs, language):
        batch = self._build_batch(obs, language)
        
        # Use select_action() for inference (not forward() which is for training)
        # select_action() returns a single action: (batch_size, action_dim)
        action = self.policy.select_action(batch)
        
        # LIBERO requires action in [-1, 1]. GRPO update requires tensor with grad history so return that.
        # Can convert to other formats if needed (eg for passing into env.step)
        action = torch.clamp(action, -1.0, 1.0)
        # TODO: this should be fixed by removing the torch.no_grad annotation on select_action, but it appears the libero version being used
        # is the public package and local changes to the submodule aren't being used
        assert action.grad is not None

        return action

    @torch.no_grad()
    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()
