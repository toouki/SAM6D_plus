import os
import logging
import os, sys
import os.path as osp
from utils.inout import get_root_project

# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import hydra
from omegaconf import DictConfig, OmegaConf

# Ultralytics SAM模型字典
model_dict = {
        "sam_b.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/sam_b.pt",  # SAM Base
        "sam_l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/sam_l.pt",  # SAM Large
        "sam_s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/sam_s.pt",  # SAM Small
    }

def download_model(url, output_path):
    import os

    command = f"wget -O {output_path}/{url.split('/')[-1]} {url} --no-check-certificate"
    os.system(command)


@hydra.main(
    version_base=None,
    config_path="./configs",
    config_name="download",
)
def download(cfg: DictConfig) -> None:
    model_name = "sam_b.pt" # default Ultralytics SAM Base model
    save_dir = osp.join(get_root_project(), "checkpoints/SAM")
    os.makedirs(save_dir, exist_ok=True)
    download_model(model_dict[model_name], save_dir)
    
if __name__ == "__main__":
    download()
    
    
