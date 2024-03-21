import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from optimization.llm.gpt import ChatBot, ChatMessage


class BaseOptimizer(ABC):
    def __init__(
        self,
        dataset,
        cfg: DictConfig,
        results_dir=None,
    ) -> None:
        np.random.seed(42)
        self.results_dir = results_dir
        # init llm
        self.chatbot = ChatBot(
            api_key=cfg.llm.gpt.openai_api_key,
            api_organization=cfg.llm.gpt.openai_api_organization,
            model=cfg.llm.gpt.model_name,
        )
        self.dataset = dataset
        self.cfg = cfg

    @staticmethod
    def init_save_folder(cfg: DictConfig) -> Path | None:
        if cfg.experiment.save_results:
            path = Path(time.strftime("%Y-%m-%d__%H-%M_%S"))
            root = Path(cfg.experiment.root_path)
            os.makedirs(root / "results" / path)
            results_dir = root / "results" / path
            return results_dir
        return None

    @abstractmethod
    def train_initial_model(self, dataset, dataset_dir: Path):
        pass

    @abstractmethod
    def get_completion(self, message: ChatMessage) -> str:
        pass

    @abstractmethod
    def get_message(self, llm_template=None) -> ChatMessage:
        pass

    def write_model_evaluation(self, metrics):
        with open(
            self.results_dir / "metric_results.txt", "a", encoding="utf-8"
        ) as file:
            file.write(f"\t{metrics}")

    def write_dataset_name(self, dataset_name):
        if self.results_dir is None:
            return
        with open(
            self.results_dir / "metric_results.txt", "a", encoding="utf-8"
        ) as file:
            file.write(f"\n{dataset_name}")
