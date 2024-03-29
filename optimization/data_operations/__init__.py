from optimization.data_operations.operation_aliases import (
    Add,
    Binning,
    Div,
    Drop,
    FillnaMean,
    FillnaMedian,
    FrequencyEncoding,
    LabelEncoding,
    Minmax,
    Mul,
    OneHotEncoding,
    Pca,
    Std,
    Sub,
)

OPERATIONS = {
    "add": Add,
    "sub": Sub,
    "mul": Mul,
    "div": Div,
    "pca": Pca,
    "fillnamean": FillnaMean,
    "fillnamedian": FillnaMedian,
    "std": Std,
    "minmax": Minmax,
    "drop": Drop,
    "frequencyencoding": FrequencyEncoding,
    "binning": Binning,
    "labelencoding": LabelEncoding,
    "onehotencoding": OneHotEncoding,
}
