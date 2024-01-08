import logging
import os
import time
from pathlib import Path
from time import sleep

import hydra
import numpy as np
from catboost import CatBoostClassifier
from fedot.api.main import Fedot
from omegaconf import DictConfig
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm, trange

from src.optimization import MODELS
from src.optimization.data_operations.dataset_loaders import DatasetLoader
from src.optimization.data_operations.operation_pipeline import (
    apply_pipeline, parse_data_operation_pipeline)
from src.optimization.llm.gpt import ChatBot, ChatMessage, MessageHistory
from src.optimization.llm.llm_templates import LLMTemplate
from src.optimization.models.model_training import (
    CatboostClassifierModel, LinearClassifierModel,
    RandomForestClassifierModel, SVMClassifierModel)

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

    dataset_ids = cfg[cfg.problem_type].datasets[:1]  # take first dataset for now
    dataset_loader = DatasetLoader(dataset_ids=dataset_ids)
    counter = 1
    for dataset in dataset_loader:
        logging.info(f"Processing dataset {1}/{len(dataset_loader)}: {dataset.name}")
        counter += 1
        np.random.seed(42)
        write_dataset_name(dataset.name, results_dir)
        data_train, data_test, target_train, target_test = train_test_split(
            np.array(dataset.data), np.array(dataset.target), test_size=0.2
        )
        metrics = model().train(data_train, target_train, data_test, target_test)
        write_model_evaluation(metrics, results_dir)
        logging.info(f'Initial 0: {metrics["accuracy"]}')

        messages = MessageHistory()
        messages.add_message(llm_template.initial_template())
        messages.add_message(str(dataset))
        messages.add_message(llm_template.previous_evaluations_template())
        messages.add_message(llm_template.instruction_template())
        messages.add_pipeline_evaluation("Initial evaluation", metrics)

        for _ in range(cfg.experiment.num_iterations):
            np.random.seed(42)
            # completion = chatbot.get_completion(messages=messages)
            completion = "std(V1), fillna_median(V2), std(V3), fillna_mean(V4)"
            try:
                pipeline = parse_data_operation_pipeline(completion)
                dataset_mod = dataset.data.copy()
                dataset_mod = apply_pipeline(dataset_mod, pipeline)

                data_train, data_test, target_train, target_test = train_test_split(
                    np.array(dataset_mod), np.array(dataset.target), test_size=0.2
                )
                # TODO: fix data leak

                metrics = model(random_state=42).train(
                    data_train, target_train, data_test, target_test
                )
                write_model_evaluation(metrics, results_dir)
                messages.add_pipeline_evaluation(completion, metrics)
                logging.info(
                    f'Iteration {_ + 1}/{cfg.experiment.num_iterations} metrics: {metrics["accuracy"]}'
                )
            except KeyError as e:
                with open(results_dir / f"results_{dataset.name}.txt", "w") as f:
                    f.write(str(messages))
                    f.write(completion)
                    f.write(f"\n{type(e).__name__}, {str(e)}")
            else:
                with open(results_dir / f"results_{dataset.name}.txt", "w") as f:
                    f.write(str(messages))
            sleep(15)  # to avoid openai api limit
            # but would be better to use token counter


def run_dataset_description_experiment(cfg):
    pass


@hydra.main(version_base=None, config_path="D:/PhD/LAAFE/", config_name="cfg")
def main(cfg: DictConfig):
    run_feature_generation_experiment(cfg)


if __name__ == "__main__":
    main()
