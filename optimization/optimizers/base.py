import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from optimization.data_operations.dataset_loaders import DatasetLoader
from optimization.llm.gpt import ChatBot, ChatMessage
from optimization.llm.llm_templates import LLMTemplate


class BaseOptimizer(ABC):
    def __init__(self, cfg: DictConfig) -> None:
        np.random.seed(42)
        self.results_dir: Path = self.init_save_folder(cfg.experiment.root_path)
        # init llm
        self.chatbot = ChatBot(
            api_key=cfg.llm.gpt.openai_api_key,
            api_organization=cfg.llm.gpt.openai_api_organization,
            model=cfg.llm.gpt.model_name,
        )
        self.llm_template = LLMTemplate(
            operators=cfg.llm.operators,
            experiment_description=cfg.llm.experiment_description,
            output_format=cfg.llm.output_format,
            available_nodes_description=cfg.llm.available_nodes_description,
            instruction=cfg.llm.instruction,
            message_order=cfg.llm.message_order,
        )
        self.dataset_loader = DatasetLoader(dataset_ids=cfg[cfg.problem_type].datasets)
        self.cfg = cfg

    @staticmethod
    def init_save_folder(root, path: Path | str | None = None):
        path = path if path is not None else time.strftime("%Y%m%d-%H%M%S")
        path, root = Path(path), Path(root)
        os.makedirs(root / "results" / path)
        results_dir: Path = root / "results" / path
        return results_dir

    @abstractmethod
    def get_completion(self, message: ChatMessage) -> str:
        pass

    @abstractmethod
    def get_message(self) -> str:
        pass

    def write_model_evaluation(self, metrics):
        with open(
            self.results_dir / "metric_results.txt", "a", encoding="utf-8"
        ) as file:
            file.write(f"\t{metrics}")

    def write_dataset_name(self, dataset_name):
        with open(
            self.results_dir / "metric_results.txt", "a", encoding="utf-8"
        ) as file:
            file.write(f"\n{dataset_name}")
