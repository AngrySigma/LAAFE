import re
from collections import OrderedDict

import pandas as pd

from src.optimization.data_operations.operation_aliases import OPERATIONS


# TODO: add unary, binary, n-ary operations and in-place operations
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
        except KeyError:
            pass
    return df


class OperationPipeline:
    def __init__(self):
        self.operations = []

    def add_operation(self, operation, inp=None):
        self.operations.append(operation(inp))

    def fit_transform(self, df):
        for operation in self.operations:
            df = operation.fit_transform(df)
        return df

    def transform(self, df):
        for operation in self.operations:
            df = operation.transform(df)
        return df


if __name__ == "__main__":
    data_operation_pipeline = "pca(a)"
    parsed_data_operation_pipeline = parse_data_operation_pipeline(
        data_operation_pipeline
    )
    test_data = pd.DataFrame({"a": [5, 2, 2, 4, 5], "b": [1, 2, None, 4, 5]})
    # for operation in parsed_data_operation_pipeline:
    #     function = OPERATIONS[operation]
    #     test_data[operation] = function(test_data,
    #                                     parsed_data_operation_pipeline[
    #                                         operation])
    # test_data = apply_pipeline(test_data, parsed_data_operation_pipeline)
    operation_pipeline = OperationPipeline()
    operation_pipeline.add_operation(OPERATIONS["fillnamean"], "b")
    operation_pipeline.add_operation(OPERATIONS["pca"], "b")
    operation_pipeline.add_operation(OPERATIONS["drop"], "a")
    operation_pipeline.add_operation(OPERATIONS["std"], "b")
    operation_pipeline.add_operation(OPERATIONS["minmax"], "pca_0")
    operation_pipeline.add_operation(OPERATIONS["onehotencoding"], "pca_0")
    test_data = operation_pipeline.fit_transform(test_data)

    print(test_data)
    # print(operation_pipeline.transform(pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})))
