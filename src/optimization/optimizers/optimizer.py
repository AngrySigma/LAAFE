from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from src.optimization import MODELS
from src.optimization.data_operations import OPERATIONS
from src.optimization.data_operations.operation_pipeline import OperationPipeline
from src.optimization.llm.gpt import ChatMessage
from src.optimization.optimizers.base import BaseOptimizer


class Optimizer(BaseOptimizer):
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

    def get_completion(self):
        message = ChatMessage("\n".join(self.llm_template.messages))
        np.random.seed()
        operations_test = np.array(
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
        if self.cfg.experiment.test_mode:
            if self.cfg.experiment.interactive_mode:
                # pyperclip.copy(message.content)
                completion = input("Enter completion: ")
            else:
                operations_test = operations_test[
                    np.random.randint(0, 2, len(operations_test)).astype(bool)
                ].tolist()
                completion = self.cfg.llm.operation_split.join(operations_test)
        else:
            completion = self.chatbot.get_completion(chat_messages=message)
        return completion

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
