import os
import time
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from hydra.utils import instantiate
from langchain_core.messages import HumanMessage
from omegaconf import DictConfig

from optimization.data_operations.dataset_loaders import OpenMLDataset
from optimization.data_operations.operation_pipeline import OperationPipeline
from optimization.llm.gpt import ChatMessage
from optimization.llm.llm_templates import BaseLLMTemplate


class BaseOptimizer(ABC):
    def __init__(
        self,
        dataset: OpenMLDataset,
        cfg: DictConfig,
        results_dir: Path | None = None,
    ) -> None:
        np.random.seed(42)
        self.results_dir = results_dir
        # init llm
        self.chatbot = instantiate(cfg.llm.groq)
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
    def train_initial_model(
        self, dataset: OpenMLDataset, dataset_dir: Path
    ) -> tuple[dict[str, float], OperationPipeline]:
        pass

    @abstractmethod
    def get_candidate(self, message: HumanMessage) -> str:
        pass

    @abstractmethod
    def get_message(self, llm_template: BaseLLMTemplate | None = None) -> ChatMessage:
        pass

    def write_model_evaluation(self, metrics: dict[str, float]) -> None:
        if self.results_dir is not None:
            with open(
                self.results_dir / "metric_results.txt", "a", encoding="utf-8"
            ) as file:
                file.write(f"\t{metrics}")

    def write_dataset_name(self, dataset_name: str) -> None:
        if self.results_dir is None:
            return
        with open(
            self.results_dir / "metric_results.txt", "a", encoding="utf-8"
        ) as file:
            file.write(f"\n{dataset_name}")
