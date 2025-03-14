import glob
import logging
import os.path
import re
from copy import deepcopy
from pathlib import Path

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype

from optimization.data_operations import OPERATIONS
from optimization.data_operations.operation_aliases import (
    Drop,
    FillnaMean,
    LabelEncoding,
    Operation,
)

logger = logging.getLogger(__name__)


class OperationPipeline:
    def __init__(
        self, operations: dict[str, type[Operation]], split_by: str = "\n"
    ) -> None:
        self.operations_pipeline: list[Operation] = []
        self.operations = operations
        self.errors = None
        self.split_by = split_by

    def add_operation(
        self, operation: type[Operation], inp: list[str] | None = None
    ) -> None:
        self.operations_pipeline.append(operation(inp if inp else None))

    def validate_pipeline(self, df: pd.DataFrame) -> bool:
        test_df = deepcopy(df)
        drop_operations = []
        error_flag = False
        for operation in self.operations_pipeline:
            try:
                test_df = operation.fit_transform(test_df)
            except AssertionError as e:
                logging.debug(
                    r"Error in %s with input %s: %s",
                    operation,
                    operation.inp,
                    e,
                )
                drop_operations.append(operation)
                error_flag = True

        for operation in drop_operations:
            self.operations_pipeline.remove(operation)
        return error_flag

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # drop_operations = []
        # self.errors = []
        self.validate_pipeline(df)
        # self.build_default_pipeline(df)
        for operation in self.operations_pipeline:
            df = operation.fit_transform(df)
        # TODO: here too add some LLM invocation to fix the error
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for operation in self.operations_pipeline:
            df = operation.transform(df)
        return df

    def parse_pipeline(self, completion: str) -> None:
        completion = completion.strip().replace(" ", "")
        operations_pipeline = [
            operation.strip(r")").split("(")
            for operation in completion.split(self.split_by)
        ]
        parsed_pipeline = []
        for operation in operations_pipeline:
            try:
                inp = None
                op = self.operations[re.sub("[_-]", "", operation[0].lower())]
                inp = operation[1].lower().replace(" ", "").split(",")
                parsed_pipeline.append((op, inp))
            except (KeyError, IndexError) as e:
                logger.error(
                    r"Error in %s with %s: %s. Completion: %s",
                    operation,
                    inp,
                    e,
                    completion,
                )
        for operation_node, inp in parsed_pipeline:
            self.add_operation(operation_node, inp if inp != [""] else None)

    def draw_pipeline(self, save_path: Path | str | None = None) -> nx.DiGraph:
        graph = nx.DiGraph()
        operation_names = [
            "\n".join([operation.__class__.__name__] + operation.inp)
            for operation in self.operations_pipeline
        ]
        # convert list to list of pairs
        # operation_names = list(zip(operation_names, operation_names[1:]))
        # graph.add_edges_from(operation_names)

        edges = list(
            zip(
                list(range(len(operation_names))),
                list(range(1, len(operation_names))),
            )
        )
        labels = dict(zip(list(range(len(operation_names))), operation_names))

        if labels:
            graph.add_node(0)
        graph.add_edges_from(edges)

        plt.close()
        nx.draw_kamada_kawai(
            graph,
            labels=labels,
            with_labels=True,
            font_weight="bold",
            node_size=1000,
            font_size=10,
        )
        if not save_path:
            return graph
        save_path = Path(save_path)
        if not os.path.isdir(save_path):
            plt.savefig(save_path)
            return graph
        num_pipelines = len(glob.glob(str(save_path / "pipeline_*.png")))
        plt.savefig(save_path / f"pipeline_{str(num_pipelines)}")
        return graph

    def build_default_pipeline(self, df: pd.DataFrame) -> None:
        for column in df.columns:
            if is_numeric_dtype(df[column]):
                self.add_operation(FillnaMean, [column])
            else:
                if (
                    df[column].dtype.name in ("object", "category")
                    and df[column].nunique() < 10
                ):
                    self.add_operation(LabelEncoding, [column])
                else:
                    self.add_operation(Drop, [column])

    def __str__(self) -> str:
        return self.split_by.join(
            [
                operation.__class__.__name__
                + "("
                + (",".join(operation.inp) if operation.inp is not None else "")
                + ")"
                for operation in self.operations_pipeline
            ]
        )


class OperationPipelineGenetic:
    def __init__(
        self, operations: dict[str, type[Operation]], split_by: str = "\n"
    ) -> None:
        self.operations_pipelines: list[OperationPipeline] = []
        self.operations = operations
        self.errors = None
        self.split_by = split_by

    def parse_pipeline(self, completion: str) -> None:
        pipelines = completion.strip().split("\n")
        for proposal_pipeline in pipelines:
            pipeline = OperationPipeline(self.operations, split_by=self.split_by)
            pipeline.parse_pipeline(proposal_pipeline)
            self.operations_pipelines.append(pipeline)

    def __str__(self) -> str:
        return "\n".join([str(pipeline) for pipeline in self.operations_pipelines])

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for pipeline in self.operations_pipelines:
            df = pipeline.fit_transform(df)
        return df


if __name__ == "__main__":
    test_data = pd.DataFrame({"a": [5, 2, 2, 4, 5], "b": [1, 2, None, 4, 5]})
    PROMPT = (
        "fillna_mean(b)->"
        "pca(b)->"
        "drop(a)->"
        "std(b)->"
        "minmax()->"
        "onehotencoding(pca_0, b)->"
        "drop(pca_0_0.0)->"
        "add(b_0.75, b_1.0)->"
        "sub(b_0.75, b_1.0)->"
        "mul(b_0.75, b_1.0)->"
        "div(b_0.75, b_1.0)"
    )

    operation_pipeline = OperationPipeline(OPERATIONS, split_by="->")
    operation_pipeline.parse_pipeline(PROMPT)
    test_data = operation_pipeline.fit_transform(test_data)
    operation_pipeline.draw_pipeline()
    plt.show()
    print(test_data)
