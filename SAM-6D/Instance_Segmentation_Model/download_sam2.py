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

# Ultralytics SAM2模型字典
model_dict = {
        "sam2_b.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/sam2_b.pt",  # SAM2 Base
        "sam2_l.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/sam2_l.pt",  # SAM2 Large
        "sam2_s.pt": "https://github.com/ultralytics/assets/releases/download/v0.0.0/sam2_s.pt",  # SAM2 Small
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
    model_name = "sam2_b.pt" # default Ultralytics SAM2 Base model
    save_dir = osp.join(get_root_project(), "checkpoints/SAM2")
    os.makedirs(save_dir, exist_ok=True)
    download_model(model_dict[model_name], save_dir)
    
if __name__ == "__main__":
    download()