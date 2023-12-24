import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Operation(ABC):
    def __init__(self, description=None, *args, **kwargs):
        self.description = description if description is not None else ""

    @abstractmethod
    def __call__(self, df, inp):
        pass

    def __str__(self):
        return self.description

    def __repr__(self):
        return self.description


class InplaceOperation(Operation, ABC):
    def __init__(self, description=None, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def __call__(self, df, inp):
        return df


class ReturnOperation(Operation, ABC):
    def __init__(self, description=None, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def __call__(self, df, inp):
        return df[inp]


class Drop(InplaceOperation):
    def __init__(self):
        super().__init__()
        self.description = "Drop input columns inplace"

    def __call__(self, df, inp):
        df.drop(columns=inp, inplace=True)
        return df


# old
class Add(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Add two input columns together to a new column "add"'

    def __call__(self, df, inp):
        return df[inp[0]].add(df[inp[1]])


# old
class Sub(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Subtract two input columns together to a new column "sub"'

    def __call__(self, df, inp):
        return df[inp[0]].sub(df[inp[1]])


# old
class Mul(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Multiply two input columns together to a new column "mul"'

    def __call__(self, df, inp):
        return df[inp[0]].mul(df[inp[1]])


# old
class Div(Operation):
    def __init__(self):
        super().__init__()
        self.description = 'Divide two input columns together to a new column "div"'

    def __call__(self, df, inp):
        return df[inp[0]].div(df[inp[1]])


class Pca(InplaceOperation):
    def __init__(self):
        super().__init__()
        self.pca = None
        self.description = "Create new column from PCA on input columns"

    def __call__(self, df, inp):
        inp = inp if inp else df.columns
        # if self.pca is None:
        self.pca = PCA(1)
        df["pca"] = self.pca.fit_transform(df[inp])
        # df.drop(columns=inp, inplace=True)
        return df
        # df['pca'] = self.pca.transform(df[inp])
        # df.drop(columns=inp, inplace=True)
        # return df
        # TODO: fix pca initialization. now it uses the same pca if not initialized anew each time


class FillnaMean(InplaceOperation):
    def __init__(self):
        super().__init__()
        self.mean = None
        self.description = "Fill missing values with mean inplace"

    def __call__(self, df, inp=None):
        inp = inp if inp else df.columns
        if self.mean is None:
            self.mean = df[inp].mean()
        df[inp] = df[inp].fillna(self.mean)
        return df


class FillnaMedian(InplaceOperation):
    def __init__(self):
        super().__init__()
        self.median = None
        self.description = "Fill missing values with median inplace"

    def __call__(self, df, inp=None):
        inp = inp if inp else df.columns
        if self.median is None:
            self.median = df[inp].median()
        df[inp] = df[inp].fillna(self.median)
        return df


class Std(InplaceOperation):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()
        self.description = "Inplace standard scaling of input columns"

    def __call__(self, df, inp=None):
        inp = inp if inp else df.columns
        df[inp] = self.scaler.fit_transform(df[inp])
        return df


class Minmax(InplaceOperation):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()
        self.description = "Inplace minmax scaling of input columns"

    def __call__(self, df, inp=None):
        inp = inp if inp else df.columns
        df[inp] = self.scaler.fit_transform(df[inp])
        return df


# operation types: unary, binary, n-ary
OPERATIONS = {
    "add": Add(),
    "sub": Sub(),
    "mul": Mul(),
    "div": Div(),
    "pca": Pca(),
    "fillnamean": FillnaMean(),
    "fillnamedian": FillnaMedian(),
    "std": Std(),
    "minmax": Minmax(),
    "drop": Drop(),
}
