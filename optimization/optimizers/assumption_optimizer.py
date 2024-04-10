# pylint: skip-file
# type: ignore
import logging
from pathlib import Path

import numpy as np
import pyperclip
from fedot.api.main import Fedot
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from optimization.llm.gpt import ChatMessage
from optimization.optimizers.base import BaseOptimizer

# TODO
TEST_ASSUMPTION_PIPELINES = ["rf, pca(rf), scaling(pca)", "pca, scaling(pca)"]

logger = logging.getLogger(__name__)


class AssumptionOptimizer(BaseOptimizer):
    def __init__(self, dataset, cfg: DictConfig) -> None:
        super().__init__(dataset, cfg)

    def get_completion(self, message):
        # message = ChatMessage("\n".join(self.llm_template.messages))
        #
        # np.random.seed()
        # if self.cfg.experiment.test_mode:
        #     # if self.cfg.experiment.interactive_mode:
        #     # pyperclip.copy(message.content)
        #     completion = input("Enter completion: ")
        # else:
        #     completion = self.chatbot.get_completion(chat_messages=message)
        # return completion

        if not self.cfg.experiment.test_mode:
            return self.chatbot.get_completion(chat_messages=message)

        if not self.cfg.experiment.interactive_mode:
            return str(self._get_test_pipeline())

        pyperclip.copy(message.content)
        return input("Enter completion: ")

    @staticmethod
    def _get_test_pipeline():
        np.random.seed()
        return np.random.choice(TEST_ASSUMPTION_PIPELINES)

    def get_message(self):
        return ChatMessage(str(self.llm_template))

    def train_initial_model(self, dataset, dataset_dir: Path):
        data_train, data_test, target_train, target_test = train_test_split(
            dataset.data.copy(), dataset.target.copy(), test_size=0.2
        )
        # TODO: fix model params, from cfg
        model = Fedot(
            problem="classification",
            timeout=1,
            preset="fast_train",
            with_tuning=True,
            n_jobs=4,
            logging_level=logger.level,
            seed=42,
        )
        model.fit(features=data_train, target=target_train, predefined_model="auto")
        prediction = model.predict(data_test)  # noqa: F841
        metrics = model.get_metrics(target_test)
        self.write_model_evaluation(metrics)
        # TODO: wrong return of nodes (no parents).
        #  need to transform to pipeline format
        return metrics, model
