import logging
import os
import time
from pathlib import Path
from time import sleep

import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.optimization import MODELS
from src.optimization.data_operations import OPERATIONS
from src.optimization.data_operations.dataset_loaders import DatasetLoader
from src.optimization.data_operations.operation_pipeline import OperationPipeline
from src.optimization.llm.gpt import ChatBot, ChatMessage
from src.optimization.llm.llm_templates import LLMTemplate


def run_feature_generation_experiment(cfg: DictConfig) -> None:
    np.random.seed(42)
    # init save folder
    results_dir: Path = init_save_folder(cfg.experiment.root_path)

    # init llm
    chatbot = ChatBot(
        api_key=cfg.llm.gpt.openai_api_key,
        api_organization=cfg.llm.gpt.openai_api_organization,
        model=cfg.llm.gpt.model_name,
    )
    llm_template = LLMTemplate(operators=cfg.llm.operators)

    model = MODELS[cfg.model_type]

    dataset_ids = cfg[cfg.problem_type].datasets
    dataset_loader = DatasetLoader(dataset_ids=dataset_ids)
    for dataset in dataset_loader:
        # create all dirs to save dataset results
        dataset_dir = results_dir / dataset.name
        os.makedirs(dataset_dir)
        logging.info(f"Processing dataset {1}/{len(dataset_loader)}: {dataset.name}")
        write_dataset_name(dataset.name, results_dir)

        metrics, operations_pipeline = train_initial_model(
            cfg, model, dataset, dataset_dir
        )
        write_model_evaluation(metrics, results_dir)
        logging.info(f'Initial 0: {metrics["accuracy"]}')

        llm_template.generate_llm_messages(dataset, metrics, operations_pipeline)

        for iteration in range(cfg.experiment.num_iterations):
            np.random.seed(42)
            message = ChatMessage("".join(llm_template.messages))
            completion = chatbot.get_completion(messages=message)
            # completion = cfg.llm.operation_split.join(
            #     (
            #         "FillnaMean(pclass)",
            #         "Drop(name)",
            #         "LabelEncoding(sex)",
            #         "FillnaMean(sex)",
            #         "FillnaMean(age)",
            #         "FillnaMean(sibsp)",
            #         "FillnaMean(parch)",
            #         "Drop(ticket)",
            #         "FillnaMean(fare)",
            #         "Binning(fare)",
            #         "Pca(test)",
            #         "Drop(cabin)",
            #         "LabelEncoding(embarked)",
            #         "FillnaMean(embarked)",
            #         "Drop(boat)",
            #         "FillnaMean(body)",
            #         "Drop(home.dest)",
            #     )
            # )

            try:
                data_train, data_test, target_train, target_test = train_test_split(
                    dataset.data.copy(), dataset.target.copy(), test_size=0.2
                )

                operations_pipeline = OperationPipeline(
                    OPERATIONS, split_by=cfg.llm.operation_split
                )
                operations_pipeline.parse_pipeline(completion)
                pipeline_str = str(operations_pipeline)
                operations_pipeline.fit_transform(data_train)
                operations_pipeline.transform(data_test)
                metrics = model(random_state=42, pipeline=operations_pipeline).train(
                    data_train, target_train, data_test, target_test
                )
                operations_pipeline.draw_pipeline(
                    dataset_dir / f"pipeline_{iteration + 1}.png"
                )

                write_model_evaluation(metrics, results_dir)
                llm_template.messages[-2] += (
                    f"\nIteration {iteration + 1}: {metrics['accuracy']}, "
                    f"Pipeline: \n{pipeline_str}"
                )
                if operations_pipeline.errors:
                    error_msg = "\n".join(operations_pipeline.errors)
                    llm_template.messages[-2] += f"\nErrors: {error_msg}\n"
                logging.info(
                    f"Iteration {iteration + 1}/"
                    f"{cfg.experiment.num_iterations} "
                    f'metrics: {metrics["accuracy"]}'
                )
            except KeyError as e:
                with open(
                    dataset_dir / "prompt_result.txt", "w", encoding="utf-8"
                ) as f:
                    f.write("\n".join(llm_template.messages))
                    f.write("\nCompletion:" + completion + "\n")
                    f.write(f"\n{type(e).__name__}, {str(e)}")
            else:
                with open(
                    dataset_dir / "prompt_result.txt", "w", encoding="utf-8"
                ) as f:
                    f.write("\n".join(llm_template.messages))
            sleep(15)  # to avoid openai api limit


def write_model_evaluation(metrics, results_dir: Path):
    with open(results_dir / "metric_results.txt", "a", encoding="utf-8") as file:
        file.write(f"\t{metrics}")


def write_dataset_name(dataset_name, results_dir: Path):
    with open(results_dir / "metric_results.txt", "a", encoding="utf-8") as file:
        file.write(f"\n{dataset_name}")


def init_save_folder(root, path: Path | str | None = None):
    path = path if path is not None else time.strftime("%Y%m%d-%H%M%S")
    path, root = Path(path), Path(root)
    os.makedirs(root / "results" / path)
    results_dir: Path = root / "results" / path
    return results_dir


def train_initial_model(cfg, model, dataset, dataset_dir: Path):
    operations_pipeline = OperationPipeline(
        operations=OPERATIONS, split_by=cfg.llm.operation_split
    )
    data_train, data_test, target_train, target_test = train_test_split(
        dataset.data.copy(), dataset.target.copy(), test_size=0.2
    )
    operations_pipeline.build_default_pipeline(data_train)
    operations_pipeline.draw_pipeline(dataset_dir / "pipeline_0.png")
    data_train = operations_pipeline.fit_transform(data_train)
    data_test = operations_pipeline.transform(data_test)

    metrics = model(random_state=42, pipeline=operations_pipeline).train(
        data_train, target_train, data_test, target_test
    )
    return metrics, operations_pipeline


@hydra.main(version_base=None, config_path="D:/PhD/LAAFE/cfg", config_name="cfg")
def main(cfg: DictConfig):
    run_feature_generation_experiment(cfg)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
