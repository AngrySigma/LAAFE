import logging
import os
from time import sleep

import hydra
from omegaconf import DictConfig

from src.optimization.optimizers.optimizer import Optimizer


def run_feature_generation_experiment(cfg: DictConfig, log_level: int) -> None:
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    optimizer = Optimizer(cfg)

    for num, dataset in enumerate(optimizer.dataset_loader):
        # create all dirs to save dataset results
        dataset_dir = optimizer.results_dir / dataset.name
        os.makedirs(dataset_dir)
        logger.info(
            r"Processing dataset %s/%s: %s",
            num + 1,
            len(optimizer.dataset_loader),
            dataset.name,
        )
        optimizer.write_dataset_name(dataset.name)
        metrics, operations_pipeline = optimizer.train_initial_model(
            dataset=dataset, dataset_dir=dataset_dir
        )
        logger.info("Initial 0: %s", metrics["accuracy"])
        optimizer.llm_template.generate_initial_llm_message(
            dataset, metrics, operations_pipeline
        )

        for iteration in range(cfg.experiment.num_iterations):
            completion = optimizer.get_completion()
            try:
                pipeline_str, metrics = optimizer.fit_model(
                    dataset,
                    completion=completion,
                    plot_path=dataset_dir / f"pipeline_{iteration + 1}.png",
                )
                optimizer.llm_template.update_evaluations(
                    f"Iteration {iteration + 1}", metrics, pipeline_str
                )
                # if operations_pipeline.errors:
                #     error_msg = "\n".join(operations_pipeline.errors)
                #     optimizer.llm_template.messages[-2] += (f"\nErrors: "
                #                                             f"{error_msg}\n")
                logger.info(
                    r"Iteration %s/%s metrics: %s",
                    iteration + 1,
                    cfg.experiment.num_iterations,
                    metrics["accuracy"],
                )
            except (KeyError, ValueError) as e:
                logger.error(r"KeyError: %s", e)
                with open(
                    dataset_dir / "prompt_result.txt", "w", encoding="utf-8"
                ) as f:
                    f.write("\n".join(optimizer.llm_template.messages))
                    f.write("\nCompletion:" + completion + "\n")
                    f.write(f"\n{type(e).__name__}, {str(e)}")
            else:
                with open(
                    dataset_dir / "prompt_result.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(str(optimizer.llm_template))
            if not cfg.experiment.test_mode:
                sleep(15)  # to avoid openai api limit


@hydra.main(version_base=None, config_path="D:/PhD/LAAFE/cfg", config_name="cfg")
def main(cfg: DictConfig):
    run_feature_generation_experiment(cfg, log_level=logging.INFO)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
