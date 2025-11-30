# smolvla_policy.py

import torch
import numpy as np
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from torchvision.transforms import v2
from transformers import AutoTokenizer
from utils.find_normalizer_constants import obtain_dataset_normalizer_stats
from utils.find_unnormalizer_constants import obtain_dataset_unnormalizer_stats
from robosuite.utils.transform_utils import quat2axisangle

# NOTE: there are these already-configured pre/post processing pipelines in lerobot but they're very convoluted and buggy.
# I just referenced how they do things and made it so we do each step within our own mapping code from LIBERO observation -> model input
# from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

# NOTE: for some reason accelerate may try to distribute the model during runtime, leading to device mismatches. If this happens,
# set CUDA_VISIBLE_DEVICES to one device

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
            "HuggingFaceTB/SmolVLM2-500M-Instruct" # is it that or SmolVLM2-500M-Instruct
        )
        self.state_mean = obtain_dataset_normalizer_stats()["observation.state.mean"]
        self.state_std =  obtain_dataset_normalizer_stats()["observation.state.std"]

        self.action_mean = obtain_dataset_unnormalizer_stats()["action.mean"]
        self.action_std =  obtain_dataset_unnormalizer_stats()["action.std"]
        self.eps = 1e-8

    def _extract_images(self, obs):
        agentview_img = obs["agentview_image"]       # (H,W,3)

        eye_img = obs["robot0_eye_in_hand_image"]     # (H,W,3)

        agentview_img = np.fliplr(np.flipud(obs["agentview_image"]).copy()).copy() # could be inefficient not sure
        eye_img = np.fliplr(np.flipud(obs["robot0_eye_in_hand_image"]).copy()).copy()


        agentview_img = self.img_transform(agentview_img)
        eye_img = self.img_transform(eye_img)

        return agentview_img, eye_img

    def _extract_state(self, obs):
        pos = obs["robot0_eef_pos"]                  # (3,) #from lerobot make_env
        quat = obs["robot0_eef_quat"]                # (4,)
        axis_angle = quat2axisangle(quat)            # (3,)
        g0, g1 = obs["robot0_gripper_qpos"]          # (2,)

        state = np.concatenate(
            (pos, axis_angle, np.array([g0, g1], dtype=np.float32)),
            dtype=np.float32,
        )

        normalized_state = self._normalize_state(state)
        return normalized_state
    def _normalize_state(self, state_raw):
        """
        Normalize the 8D state vector using the dataset statistics stored
        in the normalizer processor safetensors file.
        """

        if not isinstance(state_raw, torch.Tensor):
            state_raw = torch.tensor(state_raw, dtype=torch.float32)


        state_norm = (state_raw - self.state_mean) / (self.state_std + self.eps)

        

        return state_norm
    
    def _unnormalize_action(self, action_norm):
        """
        Convert normalized action from the model back to real action.
        SmolVLA outputs normalized actions → we unnormalize using:
            real = norm * std + mean
        """
        action_norm = action_norm.to('cpu') # hacky fix but in a rush
        self.action_std = self.action_std.to('cpu')
        self.action_mean = self.action_mean.to('cpu')

        return action_norm * self.action_std + self.action_mean
    
    def _tokenize_prompt(self, task_description):
        if not task_description.endswith("\n"): # model expects \n at end of each prompt
            task_description = f"{task_description}\n"

        tokenized = self.tokenizer(
            [task_description],  
            max_length=self.max_token_length,
            truncation=True,
            padding='do_not_pad',
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
            "observation.images.image": agentview_img,
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
        action_real = self._unnormalize_action(action)
        
        # LIBERO requires action in [-1, 1]. GRPO update requires tensor with grad history so return that.
        # Can convert to other formats if needed (eg for passing into env.step)
        action = torch.clamp(action_real, -1.0, 1.0)

        return action

    @torch.no_grad()
    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()
