"""
test hydra
"""
import warnings

import hydra
from omegaconf import DictConfig

warnings.filterwarnings("ignore")


@hydra.main(config_path="../config", config_name="main")
def process_data(config: DictConfig):
    """Function to process the data"""

    raw_path = config.raw.path
    print(f"Process data using {raw_path}")
    print(f"Columns used: {config.process.use_columns}")


if __name__ == "__main__":
    process_data()
