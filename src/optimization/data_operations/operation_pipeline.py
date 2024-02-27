import logging
import re
from copy import deepcopy
from pathlib import Path

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from pandas.core.dtypes.common import is_numeric_dtype

from src.optimization.data_operations import OPERATIONS
from src.optimization.data_operations.operation_aliases import (
    Drop,
    FillnaMean,
    LabelEncoding,
)


def parse_data_operation_pipeline(data_operation_pipeline):
    parsed_data_operation_pipeline = []
    operations = re.findall(r"[a-zA-Z_]*\(.*?\)", data_operation_pipeline)
    operations = [operation.split("(") for operation in operations]
    for operation, args in operations:
        args = args.strip(")")
        inp = [arg.strip() for arg in args.split(",") if arg.strip()]
        parsed_data_operation_pipeline.append(
            (operation.lower().replace("_", ""), inp if inp else None)
        )
        # parsed_data_operation_pipeline[operation.lower()] = inp if inp else None
    return parsed_data_operation_pipeline


def apply_pipeline(df, parsed_data_operation_pipeline):
    for operation, inp in parsed_data_operation_pipeline:
        function = OPERATIONS[operation]
        # inp = parsed_data_operation_pipeline[operation]
        try:
            df = function(df, inp)
            # df[f'{operation}_{"_".join(inp)}'] = function(df, inp[0])
        except (KeyError, TypeError) as e:
            logging.error(f"Error in {operation} with {inp}: {e}")
    return df


class OperationPipeline:
    def __init__(self, operations, split_by="\n"):
        self.operations_pipeline = []
        self.operations = operations
        self.errors = None
        self.split_by = split_by

    def add_operation(self, operation, inp=None):
        self.operations_pipeline.append(operation(inp if inp else None))

    def validate_pipeline(self, df):
        test_df = deepcopy(df)
        drop_operations = []
        error_flag = False
        for operation in self.operations_pipeline:
            try:
                test_df = operation.fit_transform(test_df)
            except (KeyError, ValueError, TypeError) as e:
                logging.debug(f"Error in {operation} with input {operation.inp}: {e}")
                drop_operations.append(operation)
                error_flag = True

        for operation in drop_operations:
            self.operations_pipeline.remove(operation)
        return error_flag

    def fit_transform(self, df):
        # drop_operations = []
        # self.errors = []
        self.validate_pipeline(df)
        # self.build_default_pipeline(df)
        for operation in self.operations_pipeline:
            df = operation.fit_transform(df)
            # except (KeyError, ValueError) as e:
            # drop_operations.append(operation)
            # self.errors.append(
            #     f"{operation.__class__.__name__}{e.__class__.__name__}: {e}"
            # )
        # for operation in drop_operations:
        #     self.operations_pipeline.remove(operation)
        # TODO: here too add some LLM invocation to fix the error
        return df

    def transform(self, df):
        for operation in self.operations_pipeline:
            df = operation.transform(df)
        return df

    def parse_pipeline(self, prompt):
        operations_pipeline = [
            operation.strip(r"\w)").split("(")
            for operation in prompt.strip().split(self.split_by)
        ]
        operations_pipeline = [
            (
                self.operations[re.sub("[_-]", "", operation[0].lower())],
                operation[1].lower().replace(" ", "").split(","),
            )
            for operation in operations_pipeline
        ]
        for operation, inp in operations_pipeline:
            self.add_operation(operation, inp if inp != [""] else None)

    def draw_pipeline(self, save_path: Path | str | None = None):
        graph = nx.DiGraph()
        operation_names = [
            "\n".join([operation.__class__.__name__] + operation.inp)
            for operation in self.operations_pipeline
        ]
        # convert list to list of pairs
        # operation_names = list(zip(operation_names, operation_names[1:]))
        # graph.add_edges_from(operation_names)

        edges = list(
            zip(list(range(len(operation_names))), list(range(1, len(operation_names))))
        )
        graph.add_edges_from(edges)

        labels = dict(zip(list(range(len(operation_names))), operation_names))
        plt.close()
        nx.draw_kamada_kawai(
            graph,
            labels=labels,
            with_labels=True,
            font_weight="bold",
            node_size=1000,
            font_size=10,
        )
        if save_path:
            plt.savefig(save_path)
        return graph

    def build_default_pipeline(self, df):
        for column in df.columns:
            if is_numeric_dtype(df[column]):
                self.add_operation(FillnaMean, [column])
            else:
                if df[column].dtype.name == "object" and df[column].nunique() < 10:
                    self.add_operation(LabelEncoding, [column])
                else:
                    self.add_operation(Drop, [column])

    def __str__(self):
        return self.split_by.join(
            [
                operation.__class__.__name__ + "(" + ",".join(operation.inp) + ")"
                for operation in self.operations_pipeline
            ]
        )


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
