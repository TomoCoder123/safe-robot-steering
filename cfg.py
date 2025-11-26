from hydra import compose, initialize

from LIBERO.libero.libero import benchmark, get_libero_path
import hydra
import pprint
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYOPENGL_PLATFORM'] = 'egl'
from omegaconf import OmegaConf
import yaml
from easydict import EasyDict

hydra.core.global_hydra.GlobalHydra.instance().clear()

### load the default hydra config
initialize(config_path="./LIBERO/libero/configs")
hydra_cfg = compose(config_name="config")
yaml_config = OmegaConf.to_yaml(hydra_cfg)
cfg = EasyDict(yaml.safe_load(yaml_config))

pp = pprint.PrettyPrinter(indent=2)
# pp.pprint(cfg.policy)
pp.pprint(cfg)
