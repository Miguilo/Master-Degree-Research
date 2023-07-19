import sys
sys.path.append('../src/')

import hydra
from omegaconf import DictConfig
from utils.data import get_absolute_path, get_data


@hydra.main(config_path="../config", config_name="main.yaml")
def main(cfg: DictConfig):
    path = cfg.data.polar.final.path
    print(path)
    return


if __name__ == "__main__":
    main()
