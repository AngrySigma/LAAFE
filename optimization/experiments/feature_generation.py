import logging
import os
from time import sleep

from omegaconf import DictConfig

from optimization.data_operations.dataset_loaders import DatasetLoader, OpenMLDataset
from optimization.optimizers.feature_optimizer import (
    FeatureOptimizer,
    FeatureOptimizerGenetic,
)
from optimization.utils.visualisation import plot_metrics_history


def run_feature_generation_experiment(cfg: DictConfig, log_level: int) -> None:
    results_dir = FeatureOptimizer.init_save_folder(cfg)
    if results_dir is not None and cfg.experiment.save_results:
        logging.basicConfig(filename=results_dir / ".log", level=log_level)
    logger = logging.getLogger(__name__)
    # logger.setLevel(log_level)

    dataset_loader = DatasetLoader(dataset_ids=cfg[cfg.problem_type].datasets)
    dataset: OpenMLDataset
    for num, dataset in enumerate(dataset_loader):
        dataset.data.rename(columns=lambda x: x.lower(), inplace=True)
        optimizer = FeatureOptimizer(dataset, cfg, results_dir=results_dir)

        # create all dirs to save dataset results
        dataset_dir = (
            optimizer.results_dir / dataset.name
            if optimizer.results_dir is not None
            else None
        )
        if dataset_dir is not None and cfg.experiment.save_results:
            os.makedirs(dataset_dir)
        logger.info(
            r"Processing dataset %s/%s: %s",
            num + 1,
            len(dataset_loader),
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
        metrics_history = [metrics["accuracy"]] * (cfg.experiment.num_iterations + 1)
        for iteration in range(cfg.experiment.num_iterations):
            message = optimizer.get_message()
            completion = optimizer.get_completion(message)
            try:
                pipeline_str, metrics = optimizer.fit_model(  # type: ignore
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
                metrics_history[iteration + 1] = metrics["accuracy"]
            except (
                KeyError,
                ValueError,
                TypeError,
                ZeroDivisionError,
                TimeoutError,
            ) as e:
                logger.error(
                    rf"Iteration {iteration + 1}: %s:%s. Completion: %s",
                    e.__class__.__name__,
                    e,
                    completion,
                )
                # with open(
                #     dataset_dir / "prompt_result.txt", "w", encoding="utf-8"
                # ) as f:
                #     f.write(str(optimizer.llm_template))
                #     f.write("\nCompletion:" + completion + "\n")
                #     f.write(f"\n{type(e).__name__}, {str(e)}")
            else:
                if dataset_dir is not None and cfg.experiment.save_results:
                    with open(
                        dataset_dir / "prompt_result.txt", "w", encoding="utf-8"
                    ) as f:
                        f.write(str(optimizer.llm_template))
            if not cfg.experiment.test_mode:
                sleep(cfg.llm.gpt.sleep_timeout)  # to avoid openai api limit
        if cfg.experiment.save_results:
            plot_metrics_history(metrics_history, dataset_dir)


def run_feature_generation_experiment_genetic(
    cfg: DictConfig, log_level: int = logging.INFO
) -> None:
    results_dir = (
        FeatureOptimizerGenetic.init_save_folder(cfg)
        if cfg.experiment.save_results
        else None
    )
    if results_dir is not None and cfg.experiment.save_results:
        logging.basicConfig(filename=results_dir / ".log", level=log_level)
    logger = logging.getLogger(__name__)
    # logger.setLevel(log_level)

    dataset_loader = DatasetLoader(dataset_ids=cfg[cfg.problem_type].datasets)
    dataset: OpenMLDataset
    for num, dataset in enumerate(dataset_loader):
        dataset.data.rename(columns=lambda x: x.lower(), inplace=True)
        optimizer = FeatureOptimizerGenetic(dataset, cfg, results_dir=results_dir)

        dataset_dir = (
            optimizer.results_dir / dataset.name
            if optimizer.results_dir is not None
            else None
        )
        if dataset_dir is not None and cfg.experiment.save_results:
            os.makedirs(dataset_dir)
        logger.info(
            r"Processing dataset %s/%s: %s",
            num + 1,
            len(dataset_loader),
            dataset.name,
        )
        optimizer.write_dataset_name(dataset.name)
        metric, operations_pipeline = optimizer.train_initial_model(
            dataset=dataset, dataset_dir=dataset_dir
        )
        logger.info("Initial 0: %s", metric["accuracy"])
        optimizer.llm_template.generate_initial_llm_message(
            dataset, metric, operations_pipeline
        )
        metrics_history = [metric["accuracy"]] * (cfg.experiment.num_iterations + 1)

        for iteration in range(cfg.experiment.num_iterations):
            message = optimizer.get_message()
            completion = optimizer.get_completion(message=message, population=10)
            try:
                pipeline_str, metric = optimizer.fit_model(  # type: ignore
                    dataset, completion=completion, plot_path=dataset_dir
                )
                metrics_history[iteration + 1] = metric["accuracy"]
                optimizer.llm_template.update_evaluations(
                    f"Top pipeline on iteration {iteration}", metric, pipeline_str
                )
                logger.info(
                    r"Iteration %s/%s metrics: %s",
                    iteration + 1,
                    cfg.experiment.num_iterations,
                    metric["accuracy"],
                )

            except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
                logger.error(
                    rf"Iteration {iteration + 1}: %s:%s. Completion: %s",
                    e.__class__.__name__,
                    e,
                    completion,
                )
            if dataset_dir is not None and cfg.experiment.save_results:
                with open(
                    dataset_dir / "prompt_result.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(str(optimizer.llm_template))
            if not cfg.experiment.test_mode:
                sleep(cfg.llm.gpt.sleep_timeout)  # to avoid openai api limit
        if cfg.experiment.save_results:
            plot_metrics_history(metrics_history, dataset_dir)
