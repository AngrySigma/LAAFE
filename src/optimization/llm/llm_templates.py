from dataclasses import dataclass

import hydra

from src.optimization.data_operations import OPERATIONS
from src.optimization.data_operations.dataset_loaders import (
    DatasetLoader,
    OpenMLDataset,
)
from src.optimization.data_operations.operation_aliases import PipelineNode


@dataclass
class BaseLLMTemplate:
    nodes: tuple[PipelineNode]
    experiment_description: str = (
        "You should optimize directed acyclic graph structure to improve the metric."
    )
    output_format: str = (
        "The output is a directed acyclic graph structure written as in"
        "EXAMPLE:"
        "\n\tNode1, Node2(Node1), Node3(Node1), Node4(Node2, Node3)"
        "Which means node 1 is the first node, nodes 2 and 3"
        " depend on the first node,"
        " and node 4 depends on nodes 2 and 3."
        "Please, do not add anything else to the answer"
    )
    available_nodes_description: str = "Available nodes are the following:"
    instruction: str = (
        "Please, choose the best nodes for the graph"
        " based on the information provided"
    )
    message_order: tuple[str, ...] = (
        "experiment_description",
        "output_format",
        "available_nodes",
        "dataset",
        "previous_evaluations",
        "instruction",
    )

    def __init__(self, operators, previous_evaluations=None):
        self.operators = [OPERATIONS[operator] for operator in operators]
        self.previous_evaluations = (
            previous_evaluations
            if (previous_evaluations is not None)
            else "Previous pipeline evaluations and corresponding metrics:"
        )
        self.messages = {}


@dataclass
class LLMTemplate:
    operators: list[str]
    experiment_description: str = (
        "You should perform a feature engineering for the provided "
        "dataset to improve the performance of the model. "
    )
    output_format: str = (
        "The output is an operation pipeline written as in "
        "PIPELINE EXAMPLE:"
        "\n\toperation1(df_column)"
        "->operation2(df_column_1, df_column_2)"
        "->operationN()"
        "\nEmpty brackets mean that operation is applied "
        "to all columns of the dataset."
        "\nPlease, don't use spaces between operations and inputs. "
        "Name operations exactly as they are listed in initial message."
        " Do not add any other information to the output."
    )
    available_nodes_description: str = "Available data operations" " are the following"
    instruction: str = (
        "Based on the information provided, please, "
        "choose the operations you want to use in your pipeline "
        "and write them in the output format. "
        "Operation inputs have to match the columns of the dataset. "
        "Avoid repeating operation pipelines."
    )
    message_order: tuple[str, ...] = (
        "experiment_description",
        "output_format",
        "available_nodes",
        "dataset",
        "previous_evaluations",
        "instruction",
    )

    def __post_init__(self):
        self.operators: list[PipelineNode] = [
            OPERATIONS[operator] for operator in self.operators
        ]
        self.previous_evaluations = (
            "Previous pipeline evaluations and corresponding metrics:"
        )
        self.messages = {}

    def generate_initial_llm_message(
        self,
        dataset: OpenMLDataset,
        metrics: dict[str, float] | None = None,
        operations_pipeline: str | None = None,
    ):
        # self.messages['initial_template'] = self.initial_template()
        self.messages["experiment_description"] = self.experiment_description
        self.messages["output_format"] = self.output_format
        self.messages["available_nodes_description"] = self.available_nodes_description
        self.messages["available_nodes"] = "\t" + "\n\t".join(
            [
                operator.__name__ + ": " + operator.description()
                for operator in self.operators
            ]
        )
        self.messages["dataset"] = str(dataset)
        self.messages["previous_evaluations"] = ""
        if metrics is not None and operations_pipeline is not None:
            self.messages["previous_evaluations"] = self.previous_evaluations_template()
            self.messages["previous_evaluations"] += (
                f"\nInitial evaluation: {metrics['accuracy']}, "
                f"Pipeline: {operations_pipeline}"
            )
        self.messages["instruction"] = self.instruction

    def update_evaluations(
        self, label: str, metrics: dict[str, float], operations_pipeline: str
    ):
        if not self.messages["previous_evaluations"]:
            self.messages["previous_evaluations"] = self.previous_evaluations_template()
        self.messages["previous_evaluations"] += (
            f"\n{label}: {metrics['accuracy']}, " f"Pipeline: {operations_pipeline}"
        )

    def previous_evaluations_template(self):
        template = self.previous_evaluations
        return template

    def __str__(self):
        message = [self.messages[paragraph] for paragraph in self.message_order]
        return "\n".join(message)


@hydra.main(version_base=None, config_path="D:/PhD/LAAFE/cfg/", config_name="cfg")
def main(cfg):
    dataset_ids = [
        40945,
    ]
    dataset_loader = DatasetLoader(dataset_ids=dataset_ids)
    dataset = dataset_loader[0]
    llm_template = LLMTemplate(operators=cfg.llm.operators)
    llm_template.generate_initial_llm_message(dataset)
    llm_template.update_evaluations(
        "Evaluation 1",
        {"accuracy": 0.9},
        "Add(age, fare)->Sub(sibsp, parch)->"
        "FillnaMean(age)->FillnaMean(fare)->Drop(name)",
    )
    llm_template.update_evaluations(
        "Evaluation 2",
        {"accuracy": 1},
        "Add(age, fare)->Sub(sibsp, parch)->"
        "FillnaMean(age)->FillnaMean(fare)->Drop(name)",
    )
    print(llm_template)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
