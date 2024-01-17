import re
from collections import OrderedDict

import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt

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
    def __init__(self, operations):
        self.operations_pipeline = []
        self.operations = operations

    def add_operation(self, operation, inp=None):
        self.operations_pipeline.append(operation(inp))

    def fit_transform(self, df):
        for operation in self.operations_pipeline:
            df = operation.fit_transform(df)
        return df

    def transform(self, df):
        for operation in self.operations_pipeline:
            df = operation.transform(df)
        return df

    def parse_pipeline(self, prompt):
        operations_pipeline = [operation.strip('\w)').split('(') for operation in
                      prompt.strip().split('\n')]
        operations_pipeline = {
            self.operations[re.sub("[_-]", "", operation[0].lower())]: operation[1].split(',')
            for operation in operations_pipeline}
        for operation, inp in operations_pipeline.items():
            self.add_operation(operation, inp)

    def draw_pipeline(self):
        graph = nx.DiGraph()
        operation_names = ['\n'.join([operation.__class__.__name__] + operation.inp) for operation in self.operations_pipeline]
        # convert list to list of pairs
        operation_names = list(zip(operation_names, operation_names[1:]))
        graph.add_edges_from(operation_names)

        nx.draw_spectral(graph, with_labels=True, font_weight='bold', node_size=1000, font_size=10)
        return graph



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
    prompt = ('fillna_mean(b)\n'
              'pca(b)\n'
              'drop(a)\n'
                'std(b)\n'
                'minmax(pca_0)\n'
                'onehotencoding(pca_0)\n')



    operation_pipeline = OperationPipeline(OPERATIONS)
    operation_pipeline.parse_pipeline(prompt)
    test_data = operation_pipeline.fit_transform(test_data)
    operation_pipeline.draw_pipeline()
    plt.show()
    print(test_data)
    # print(operation_pipeline.transform(pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 2, 3, 4, 5]})))
