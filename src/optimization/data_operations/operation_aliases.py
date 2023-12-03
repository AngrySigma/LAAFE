import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Operation(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, df, inp):
        pass

    def __str__(self):
        return self.description

    def __repr__(self):
        return self.description


class Add(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Add two input columns together to a new column "add"'

    def __call__(self, df, inp):
        return df[inp[0]].add(df[inp[1]])


class Sub(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Subtract two input columns together to a new column "sub"'

    def __call__(self, df, inp):
        return df[inp[0]].sub(df[inp[1]])


class Mul(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Multiply two input columns together to a new column "mul"'

    def __call__(self, df, inp):
        return df[inp[0]].mul(df[inp[1]])


class Div(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Divide two input columns together to a new column "div"'

    def __call__(self, df, inp):
        return df[inp[0]].div(df[inp[1]])


class Pca(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'PCA on two input columns to a new column "pca"'

    def __call__(self, df, inp):
        return pd.DataFrame(PCA(1).fit_transform(df[inp]))


class Fillna_mean(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Fill missing values with mean'

    def __call__(self, df, inp):
        return df[inp].fillna(df[inp].mean())


class Fillna_median(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Fill missing values with median'

    def __call__(self, df, inp):
        return df[inp].fillna(df[inp].median())


class Std(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Standardize input columns'

    def __call__(self, df, inp):
        return pd.DataFrame(StandardScaler().fit_transform(df[inp]),
                            columns=inp)


class Minmax(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Minmax input columns'

    def __call__(self, df, inp):
        return pd.DataFrame(MinMaxScaler().fit_transform(df[inp]), columns=inp)


# operation types: unary, binary, n-ary
OPERATIONS = {
    'add': Add(),
    'sub': Sub(),
    'mul': Mul(),
    'div': Div(),
    'pca': Pca(),
    'fillna_mean': Fillna_mean(),
    'fillna_median': Fillna_median(),
    'std': Std(),
    'minmax': Minmax(),
}
