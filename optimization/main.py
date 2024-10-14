import logging

import hydra
from omegaconf import DictConfig

from optimization.experiments.feature_generation import (  # noqa: F401 # pylint: disable=unused-import
    run_feature_generation_experiment,
    run_feature_generation_experiment_genetic,
)


@hydra.main(
    version_base="1.2", config_path="/home/iiov/llm/LAAFE/cfg", config_name="cfg"
)
def main(cfg: DictConfig) -> None:
    run_feature_generation_experiment(cfg, log_level=logging.DEBUG)
    # run_feature_generation_experiment_genetic(cfg, log_level=logging.DEBUG)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
