# smolvla_policy.py

import torch
from torch import nn
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
        self.policy = SmolVLAPolicy.from_pretrained(model_name, strict=False)
        self.policy.to(device)
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
        self.action_std = self.action_std.to(self.device)
        self.action_mean = self.action_mean.to(self.device)
        self.eps = 1e-8

    # by default set log_std to -3. Want very small stds to avoid useless jitter. GRPO will just focus on changing distribution means
    def set_log_std(self, log_std):
        new_log_std = torch.full(
            self.policy.model.log_std.shape,
            fill_value=log_std,
            device=self.device,
            dtype=self.policy.model.log_std.dtype,
        )
        self.policy.model.log_std = nn.Parameter(new_log_std)

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def _extract_images(self, obs):
        agentview_img = obs["agentview_image"]        # (H,W,3)
        eye_img = obs["robot0_eye_in_hand_image"]     # (H,W,3)

        agentview_img = np.fliplr(np.flipud(agentview_img).copy()).copy() # could be inefficient not sure
        eye_img = np.fliplr(np.flipud(eye_img).copy()).copy()

        agentview_img = self.img_transform(agentview_img)
        eye_img = self.img_transform(eye_img)

        return agentview_img, eye_img

    def _extract_state(self, obs):
        pos = obs["robot0_eef_pos"]                  # (3,) from lerobot make_env
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

    # GRPO functions
    def get_action_distr_params(self, obs, language):
        batch = self._build_batch(obs, language)
        mean, log_std = self.policy.select_action_distr_params(batch)
        # we want the pretrained action outputs to be our means. Currently, mean would not be equal to the action output of
        # the pretrained policy because we haven't normalized, so normalize it
        mean = self._unnormalize_action(mean)
        
        return mean, log_std

    def get_action_distr(self, obs, language):
        mean, log_std = self.get_action_distr_params(obs, language)

        std = torch.exp(log_std)
        distr = torch.distributions.Normal(mean, std)

        return distr

    def calculate_log_prob(self, distr, unsquished_action, squished_action):
        # tanh squish correction
        log_prob_unsquished = distr.log_prob(unsquished_action).sum(dim=-1)
        correction = torch.sum(torch.log(1 - squished_action.pow(2) + self.eps), dim=-1)
        log_prob = log_prob_unsquished - correction
        return log_prob

    # for when we need probability ratios. This uses self's outputted distribution given obs and language to calculate
    # the probability 
    def get_action_prob(self, obs, language, unsquished_action):
        distr = self.get_action_distr(obs, language)
        return self.calculate_log_prob(distr, unsquished_action, torch.tanh(unsquished_action))
    
    def sample_action(self, obs, language):
        distr = self.get_action_distr(obs, language)

        # tanh squish action to -1, 1. Better than clamping because it just squishes the Gaussian, keeping it entirely differentiable
        unsquished_action = distr.rsample()
        action = torch.tanh(unsquished_action)

        log_prob = self.calculate_log_prob(distr, unsquished_action, action)

        # return unsquished_action to use it in calculating probability ratio because can't always invert squished to get unsquished term needed in correction
        return action, log_prob, unsquished_action

    @torch.no_grad()
    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()
