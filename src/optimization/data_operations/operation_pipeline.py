import re
from collections import OrderedDict

import pandas as pd

from src.optimization.data_operations.operation_aliases import OPERATIONS


def parse_data_operation_pipeline(data_operation_pipeline):
    parsed_data_operation_pipeline = OrderedDict()
    operations = re.findall(r'[a-zA-Z_]*\(.*?\)', data_operation_pipeline)
    operations = [operation.split('(') for operation in operations]
    for operation, args in operations:
        args = args.strip(')').split('->')
        inp = [arg.strip() for arg in args[0].split(',')]
        parsed_data_operation_pipeline[operation] = inp
    return parsed_data_operation_pipeline


if __name__ == '__main__':
    data_operation_pipeline = 'add(a, b) , pca(a, b), fillna_mean(a), std(a)'
    parsed_data_operation_pipeline = parse_data_operation_pipeline(
        data_operation_pipeline)
    test_data = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [1, 2, 3, 4, 5]})
    for operation in parsed_data_operation_pipeline:
        function = OPERATIONS[operation]
        test_data[operation] = function(test_data,
                                        parsed_data_operation_pipeline[
                                            operation])
    print(test_data)
