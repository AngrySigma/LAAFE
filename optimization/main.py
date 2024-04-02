import logging

import hydra
from omegaconf import DictConfig

from optimization.experiments.feature_generation import (  # pylint: disable=unused-import # type: ignore
    run_feature_generation_experiment,
    run_feature_generation_experiment_genetic,
)
from optimization.experiments.initial_assumption_generation import (  # type: ignore # pylint: disable=unused-import
    run_initial_assumption_generation_experiment,
)


@hydra.main(version_base="1.2", config_path="D:/PhD/LAAFE/cfg", config_name="cfg")
def main(cfg: DictConfig) -> None:
    run_feature_generation_experiment(cfg, log_level=logging.DEBUG)
    # run_feature_generation_experiment_genetic(cfg, log_level=logging.DEBUG)
    # run_initial_assumption_generation_experiment(cfg, log_level=logging.INFO)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
