import logging
import os
from time import sleep

import hydra
from omegaconf import DictConfig

from optimization.optimizers.feature_optimizer import (
    FeatureOptimizer,
    FeatureOptimizerGenetic,
)
from optimization.utils.visualisation import plot_metrics_history


def run_feature_generation_experiment(cfg: DictConfig, log_level: int) -> None:
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    optimizer = FeatureOptimizer(cfg)

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
            message = optimizer.get_message()
            completion = optimizer.get_completion(message)
            try:
                pipeline_str, metrics = optimizer.fit_model(
                    dataset,
                    completion=completion,
                    plot_path=dataset_dir,
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
                logger.error(r"KeyError: %s. Completion: %s", e, completion)
                # with open(
                #     dataset_dir / "prompt_result.txt", "w", encoding="utf-8"
                # ) as f:
                #     f.write(str(optimizer.llm_template))
                #     f.write("\nCompletion:" + completion + "\n")
                #     f.write(f"\n{type(e).__name__}, {str(e)}")
            else:
                with open(
                    dataset_dir / "prompt_result.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(str(optimizer.llm_template))
            if not cfg.experiment.test_mode:
                sleep(15)  # to avoid openai api limit


def run_feature_generation_experiment_genetic(cfg, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    optimizer = FeatureOptimizerGenetic(cfg)

    for num, dataset in enumerate(optimizer.dataset_loader):
        #     # create all dirs to save dataset results
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
        metrics_history = []
        for iteration in range(cfg.experiment.num_iterations):
            message = optimizer.get_message()
            completion = optimizer.get_completion(message=message, population=10)
            # try:
            pipeline_str, metric = optimizer.fit_model(
                dataset, completion=completion, plot_path=dataset_dir
            )
            metrics_history.append(metric["accuracy"])
            optimizer.llm_template.update_evaluations(
                f"Top pipeline on iteration {iteration}", metric, pipeline_str
            )
            logger.info(
                r"Iteration %s/%s metrics: %s",
                iteration + 1,
                cfg.experiment.num_iterations,
                metric["accuracy"],
            )
            # except (KeyError, ValueError) as e:
            #     logger.error(r"Error: %s. Completion: %s",
            #                  e,
            #                  completion)
            # else:
            with open(dataset_dir / "prompt_result.txt", "w", encoding="utf-8") as f:
                f.write(str(optimizer.llm_template))
            if not cfg.experiment.test_mode:
                sleep(15)
        plot_metrics_history(metrics_history, dataset_dir / "metrics_history.png")


@hydra.main(version_base=None, config_path="D:/PhD/LAAFE/cfg", config_name="cfg")
def main(cfg: DictConfig):
    # run_feature_generation_experiment(cfg, log_level=logging.DEBUG)
    run_feature_generation_experiment_genetic(cfg, log_level=logging.DEBUG)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
