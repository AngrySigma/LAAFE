import logging
from pathlib import Path
from time import sleep

import numpy as np
import pyperclip
from langchain_core.messages import HumanMessage
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from optimization import MODELS
from optimization.data_operations import OPERATIONS
from optimization.data_operations.dataset_loaders import OpenMLDataset
from optimization.data_operations.operation_pipeline import (
    OperationPipeline,
    OperationPipelineGenetic,
)
from optimization.llm.llm_templates import BaseLLMTemplate, LLMTemplate
from optimization.optimizers.base import BaseOptimizer
from optimization.utils.function_tools import timeout

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
    def __init__(
        self,
        dataset: OpenMLDataset,
        cfg: DictConfig,
        initial_advice: str | None = None,
        results_dir: Path | None = None,
    ) -> None:
        super().__init__(dataset, cfg, results_dir=results_dir)
        if cfg.experiment.give_initial_advice and initial_advice is None:
            initial_advice_template = BaseLLMTemplate(
                message_order=tuple(cfg.dataset_advice.message_order)
            )
            initial_advice_template.generate_initial_llm_message(
                **dict(cfg.dataset_advice.messages)
            )
            message = self.get_message(llm_template=initial_advice_template)
            initial_advice = self.get_initial_advice_completion(message)
        self.llm_template = LLMTemplate(
            operator_types=cfg.llm.operators,
            experiment_description=cfg.llm.experiment_description,
            output_format=cfg.llm.output_format,
            available_nodes_description=cfg.llm.available_nodes_description,
            instruction=cfg.llm.instruction,
            message_order=cfg.llm.message_order,
            initial_advice=initial_advice if initial_advice is not None else "",
        )
        self.model = MODELS[cfg.model_type]

    def train_initial_model(
        self, dataset: OpenMLDataset, dataset_dir: Path | None = None
    ) -> tuple[dict[str, float], OperationPipeline]:
        operations_pipeline = OperationPipeline(
            operations=OPERATIONS, split_by=self.cfg.llm.operation_split
        )
        data_train, data_test, target_train, target_test = train_test_split(
            dataset.data.copy(), dataset.target.copy(), test_size=0.2
        )
        operations_pipeline.build_default_pipeline(data_train)
        if dataset_dir is not None:
            operations_pipeline.draw_pipeline(dataset_dir / "pipeline_0.png")
        data_train = operations_pipeline.fit_transform(data_train)
        data_test = operations_pipeline.transform(data_test)

        metrics = self.model(random_state=42, pipeline=operations_pipeline).train(
            data_train, target_train, data_test, target_test
        )
        if self.results_dir is not None:
            self.write_model_evaluation(metrics)
        return metrics, operations_pipeline

    # def get_message(self):
    #     return ChatMessage(str(self.llm_template))

    def get_message(self, llm_template: BaseLLMTemplate | None = None) -> HumanMessage:
        return HumanMessage(
            content=str(
                llm_template if llm_template is not None else self.llm_template
            ),
            role="user",
        )

    def get_candidate(self, message: HumanMessage) -> str:
        if not self.cfg.experiment.test_mode:
            return self.chatbot.invoke([message]).content
        match self.cfg.experiment.test_mode:
            case "interactive":
                pyperclip.copy(message.content)
                return input("Enter completion: ")
            case "random_predefined":
                return str(self._get_test_pipeline())
            case "random_search":
                return self._get_random_search_pipeline()
            case _:
                return ""

    def get_initial_advice_completion(self, message: HumanMessage) -> str:
        if not self.cfg.experiment.test_mode:
            sleep(self.cfg.llm.sleep_timeout)
            response = self.chatbot.invoke([message]).content
            sleep(self.cfg.llm.sleep_timeout)
            return response

        return (
            "INITIAL ADVICE TEST RUN: here will be the instruction from the first step"
        )

    @timeout(120)
    def fit_model(
        self,
        dataset: OpenMLDataset,
        completion: str,
        plot_path: str | None = None,
    ) -> tuple[str, dict[str, float]]:
        np.random.seed(42)
        (data_train, data_test, target_train, target_test) = train_test_split(
            dataset.data.copy(), dataset.target.copy(), test_size=0.2
        )
        operations_pipeline = OperationPipeline(
            OPERATIONS, split_by=self.cfg.llm.operation_split
        )
        operations_pipeline.parse_pipeline(completion)
        pipeline_str = str(operations_pipeline)
        data_train = operations_pipeline.fit_transform(data_train)
        data_test = operations_pipeline.transform(data_test)
        metrics = self.model(random_state=42, pipeline=operations_pipeline).train(
            data_train, target_train, data_test, target_test
        )
        if plot_path is not None:
            operations_pipeline.draw_pipeline(plot_path)
        if self.results_dir is not None:
            self.write_model_evaluation(metrics)
        return pipeline_str, metrics

    def _get_test_pipeline(self) -> str:
        np.random.seed()
        operations_test = TEST_OPERATION_PIPELINE[
            np.random.randint(0, 2, len(TEST_OPERATION_PIPELINE)).astype(bool)
        ].tolist()
        splitter: str = self.cfg.llm.operation_split
        return splitter.join(operations_test)

    def _get_random_search_pipeline(self) -> str:
        splitter: str = self.cfg.llm.operation_split
        return splitter.join(
            self._get_random_node() for _ in range(np.random.randint(1, 10))
        )

    def _get_random_node(self) -> str:
        np.random.seed()
        operator: str = np.random.choice(self.cfg.llm.operators)
        return (
            operator
            + "("
            + ",".join(
                np.random.choice(
                    self.dataset.data.columns,
                    size=np.random.randint(1, 3),
                    replace=False,
                )
            )
            + ")"
        )


class FeatureOptimizerGenetic(FeatureOptimizer):
    def __init__(
        self,
        dataset: OpenMLDataset,
        cfg: DictConfig,
        initial_advice: str | None = None,
        results_dir: Path | None = None,
    ) -> None:
        super().__init__(
            dataset, cfg, initial_advice=initial_advice, results_dir=results_dir
        )
        self.model = MODELS[cfg.model_type]

    @timeout(300)
    def fit_model(
        self,
        dataset: OpenMLDataset,
        completion: str,
        plot_path: str | None = None,
    ) -> tuple[str, dict[str, float]]:
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
                    dataset.data.copy(),
                    dataset.target.copy(),
                    test_size=0.2,
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

    def get_candidate(self, message: HumanMessage, population: int = 1) -> str:
        if not self.cfg.experiment.test_mode:
            return self.chatbot.invoke([message]).content

        match self.cfg.experiment.test_mode:
            case "interactive":
                pyperclip.copy(message.content)
                return input("Enter completion: ")
            case "random_predefined":
                return "\n".join([self._get_test_pipeline() for _ in range(population)])
            case "random_search":
                return "\n".join(
                    [self._get_random_search_pipeline() for _ in range(population)]
                )
            case "genetic_search":
                # get scores???
                # selection
                # crossover
                # mutation
                return ...
            case _:
                return ""
