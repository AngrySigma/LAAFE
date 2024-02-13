import logging
import os
import time
from pathlib import Path
from time import sleep

import hydra
import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier
from fedot.api.main import Fedot
from omegaconf import DictConfig
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm, trange

from src.optimization import MODELS
from src.optimization.data_operations import OPERATIONS
from src.optimization.data_operations.dataset_loaders import DatasetLoader
from src.optimization.data_operations.operation_pipeline import (
    OperationPipeline,
    apply_pipeline,
    parse_data_operation_pipeline,
)
from src.optimization.llm.gpt import ChatBot, ChatMessage, MessageHistory
from src.optimization.llm.llm_templates import LLMTemplate
from src.optimization.models.model_training import (
    CatboostClassifierModel,
    LinearClassifierModel,
    RandomForestClassifierModel,
    SVMClassifierModel,
)

np.random.seed(42)


def write_model_evaluation(metrics, results_dir: Path):
    with open(results_dir / "metric_results.txt", "a") as file:
        file.write(f"\t{metrics}")


def write_dataset_name(dataset_name, results_dir: Path):
    with open(results_dir / "metric_results.txt", "a") as file:
        file.write(f"\n{dataset_name}")


def run_feature_generation_experiment(cfg):
    # this all should be in a config file somewhere
    time_now = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(f"D:/PhD/LAAFE/results/{time_now}")
    results_dir: Path = Path("D:/PhD/LAAFE/results/") / time_now
    chatbot = ChatBot(
        api_key=cfg.llm.gpt.openai_api_key,
        api_organization=cfg.llm.gpt.openai_api_organization,
        model=cfg.llm.gpt.model_name,
    )
    llm_template = LLMTemplate(operators=cfg.llm.operators)
    model = MODELS[cfg.model_type]

    dataset_ids = cfg[cfg.problem_type].datasets  # take first dataset for now
    dataset_loader = DatasetLoader(dataset_ids=dataset_ids)
    counter = 1
    for dataset in dataset_loader:
        dataset_dir = results_dir / dataset.name
        os.makedirs(dataset_dir)
        logging.info(f"Processing dataset {1}/{len(dataset_loader)}: {dataset.name}")
        counter += 1
        np.random.seed(42)
        write_dataset_name(dataset.name, results_dir)
        operations_pipeline = OperationPipeline(
            OPERATIONS, split_by=cfg.llm.operation_split
        )

        data_train, data_test, target_train, target_test = train_test_split(
            dataset.data.copy(), dataset.target.copy(), test_size=0.2
        )
        operations_pipeline.build_default_pipeline(data_train)
        plt.close()
        operations_pipeline.draw_pipeline()
        plt.savefig(dataset_dir / f"pipeline_0.png")
        data_train = operations_pipeline.fit_transform(data_train)
        data_test = operations_pipeline.transform(data_test)

        metrics = model().train(data_train, target_train, data_test, target_test)
        write_model_evaluation(metrics, results_dir)
        logging.info(f'Initial 0: {metrics["accuracy"]}')

        llm_template.messages.append(llm_template.initial_template())
        llm_template.messages.append(str(dataset))
        llm_template.messages.append(llm_template.previous_evaluations_template())
        llm_template.messages.append(llm_template.instruction_template())
        llm_template.messages[-2] += (
            f"\nInitial evaluation: {metrics['accuracy']}, "
            f"Pipeline: {operations_pipeline}"
        )

        for iteration in range(cfg.experiment.num_iterations):
            np.random.seed(42)
            message = ChatMessage("".join(llm_template.messages))
            completion = chatbot.get_completion(messages=message)
            # completion = cfg.llm.operation_split.join(('FillnaMean(pclass)',
            #               'Drop(name)',
            #               'LabelEncoding(sex)',
            #               'FillnaMean(sex)',
            #               'FillnaMean(age)',
            #               'FillnaMean(sibsp)',
            #               'FillnaMean(parch)',
            #               'Drop(ticket)',
            #               'FillnaMean(fare)',
            #               'Binning(fare)',
            #               'FillnaMean(fare)',
            #               'Pca(test)',
            #               'Drop(cabin)',
            #               'LabelEncoding(embarked)',
            #               'FillnaMean(embarked)',
            #               'Drop(boat)',
            #               'FillnaMean(body)',
            #               'Drop(home.dest)',))

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
                plt.close()
                operations_pipeline.draw_pipeline()
                plt.savefig(dataset_dir / f"pipeline_{iteration + 1}.png")

                metrics = model(random_state=42).train(
                    data_train, target_train, data_test, target_test
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
            # but would be better to use token counter


def run_dataset_description_experiment(cfg):
    pass


@hydra.main(version_base=None, config_path="D:/PhD/LAAFE/cfg", config_name="cfg")
def main(cfg: DictConfig):
    run_feature_generation_experiment(cfg)


if __name__ == "__main__":
    main()
