import logging
from pathlib import Path

import numpy as np
import pyperclip
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from optimization import MODELS
from optimization.data_operations import OPERATIONS
from optimization.data_operations.operation_pipeline import (
    OperationPipeline,
    OperationPipelineGenetic,
)
from optimization.llm.gpt import ChatMessage
from optimization.optimizers.base import BaseOptimizer

logger = logging.getLogger(__name__)

TEST_OPERATION_PIPELINE = np.array(
    [
        "FillnaMean(pclass)",
        "Drop(name)",
        "LabelEncoding(sex)",
        "FillnaMean(sex)",
        "FillnaMean(age)",
        "FillnaMean(sibsp)",
        "FillnaMean(parch)",
        "Drop(ticket)",
        "FillnaMean(fare)",
        "Binning(fare)",
        "Pca(test)",
        "Drop(cabin)",
        "LabelEncoding(embarked)",
        "FillnaMean(embarked)",
        "Drop(boat)",
        "FillnaMean(body)",
        "Drop(home.dest)",
    ]
)


class FeatureOptimizer(BaseOptimizer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.model = MODELS[cfg.model_type]

    def train_initial_model(self, dataset, dataset_dir: Path):
        operations_pipeline = OperationPipeline(
            operations=OPERATIONS, split_by=self.cfg.llm.operation_split
        )
        data_train, data_test, target_train, target_test = train_test_split(
            dataset.data.copy(), dataset.target.copy(), test_size=0.2
        )
        operations_pipeline.build_default_pipeline(data_train)
        operations_pipeline.draw_pipeline(dataset_dir / "pipeline_0.png")
        data_train = operations_pipeline.fit_transform(data_train)
        data_test = operations_pipeline.transform(data_test)

        metrics = self.model(random_state=42, pipeline=operations_pipeline).train(
            data_train, target_train, data_test, target_test
        )
        self.write_model_evaluation(metrics)
        return metrics, operations_pipeline

    def get_message(self):
        return ChatMessage(str(self.llm_template))

    def get_completion(self, message):
        if not self.cfg.experiment.test_mode:
            return self.chatbot.get_completion(chat_messages=message)

        if not self.cfg.experiment.interactive_mode:
            return str(self._get_test_pipeline())

        pyperclip.copy(message.content)
        return input("Enter completion: ")

    def fit_model(self, dataset, completion, plot_path: str | None = None):
        np.random.seed(42)
        (data_train, data_test, target_train, target_test) = train_test_split(
            dataset.data.copy(), dataset.target.copy(), test_size=0.2
        )
        operations_pipeline = OperationPipeline(
            OPERATIONS, split_by=self.cfg.llm.operation_split
        )
        operations_pipeline.parse_pipeline(completion)
        pipeline_str = str(operations_pipeline)
        operations_pipeline.fit_transform(data_train)
        operations_pipeline.transform(data_test)
        metrics = self.model(random_state=42, pipeline=operations_pipeline).train(
            data_train, target_train, data_test, target_test
        )
        if plot_path is not None:
            operations_pipeline.draw_pipeline(plot_path)
        self.write_model_evaluation(metrics)
        return pipeline_str, metrics

    def _get_test_pipeline(self):
        np.random.seed()
        operations_test = TEST_OPERATION_PIPELINE[
            np.random.randint(0, 2, len(TEST_OPERATION_PIPELINE)).astype(bool)
        ].tolist()
        return self.cfg.llm.operation_split.join(operations_test)


class FeatureOptimizerGenetic(FeatureOptimizer):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.model = MODELS[cfg.model_type]

    def fit_model(self, dataset, completion, plot_path: str | None = None):
        metrics = []
        operations_pipelines = OperationPipelineGenetic(
            operations=OPERATIONS, split_by=self.cfg.llm.operation_split
        )
        operations_pipelines.parse_pipeline(completion)
        # pipeline_str = str(operations_pipelines)
        pipeline_strs = []
        # TODO: make this a user list class
        for pipeline in operations_pipelines.operations_pipelines:
            np.random.seed(42)
            try:
                (data_train, data_test, target_train, target_test) = train_test_split(
                    dataset.data.copy(), dataset.target.copy(), test_size=0.2
                )
                pipeline.fit_transform(data_train)
                pipeline.transform(data_test)
                metric = self.model(random_state=42, pipeline=pipeline).train(
                    data_train, target_train, data_test, target_test
                )
                metrics.append(metric)
                pipeline_strs.append(str(pipeline))
            except (KeyError, ValueError) as e:
                logger.error(r"Error: %s. Completion: %s", e, str(pipeline))

        # return only the results for the best pipeline
        top_pipeline_idx = np.argmax([metric["accuracy"] for metric in metrics])
        top_pipeline = pipeline_strs[top_pipeline_idx]
        top_metric = metrics[top_pipeline_idx]
        if plot_path is not None:
            operations_pipelines.operations_pipelines[top_pipeline_idx].draw_pipeline(
                plot_path
            )
        self.write_model_evaluation(top_metric)
        return top_pipeline, top_metric

    def get_completion(self, message, population=1):
        if not self.cfg.experiment.test_mode:
            return self.chatbot.get_completion(chat_messages=message)

        if not self.cfg.experiment.interactive_mode:
            return "\n".join([self._get_test_pipeline() for _ in range(population)])

        pyperclip.copy(message.content)
        return input("Enter completion: ")
