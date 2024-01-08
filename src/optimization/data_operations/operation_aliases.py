from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class Operation(ABC):
    def __init__(self, inp=None, *args, **kwargs):
        self.inp = [inp] if isinstance(inp, str) else inp

    @abstractmethod
    def __call__(self, df):
        pass

    def fit_transform(self, df):
        self.inp = df.columns if self.inp is None else self.inp

    @abstractmethod
    def transform(self, df):
        pass

    @classmethod
    @abstractmethod
    def description(cls):
        pass

    def __str__(self):
        return self.description()

    def __repr__(self):
        return self.description()


# TODO: check this. Unnecessary, can be deleted
class InplaceOperation(Operation, ABC):
    def __init__(self, inp=None, *args, **kwargs):
        super().__init__(inp, *args, **kwargs)

    @abstractmethod
    def __call__(self, df) -> None:
        pass


# TODO: update this
class ReturnOperation(Operation, ABC):
    def __init__(self, inp=None, *args, **kwargs):
        super().__init__(inp, *args, **kwargs)

    @abstractmethod
    def __call__(self, df) -> pd.DataFrame:
        return df


class Drop(InplaceOperation):
    def __init__(self, inp=None):
        super().__init__(inp)

    @classmethod
    def description(cls):
        return "Drop input columns inplace"

    def __call__(self, df):
        return self.fit_transform(df)

    def transform(self, df):
        df.drop(columns=self.inp, inplace=True)
        return df

    def fit_transform(self, df):
        self.transform(df)
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


# TODO: add train and test transformations
class Pca(InplaceOperation):
    def __init__(self, inp=None):
        super().__init__(inp)
        self.pca = PCA(0.95)

    @classmethod
    def description(cls):
        return "Create new column from PCA on input columns"

    def __call__(self, df):
        return self.fit_transform(df)

    def fit_transform(self, df):
        super().fit_transform(df)
        # TODO: fill na if there are any. moreover, this should be added to pipeline
        self.inp = df.columns if self.inp is None else self.inp
        pca_result = self.pca.fit_transform(df[self.inp])
        pca_columns = [f"pca_{i}" for i in range(pca_result.shape[1])]
        df[pca_columns] = pca_result
        return df

    def transform(self, df):
        pca_result = self.pca.transform(df[self.inp])
        pca_columns = [f"pca_{i}" for i in range(pca_result.shape[1])]
        df[pca_columns] = pca_result
        return df


class FillnaMean(InplaceOperation):
    def __init__(self, inp=None):
        super().__init__(inp)
        self.mean = None

    @classmethod
    def description(cls):
        return "Fill missing values with mean inplace"

    def __call__(self, df, inp=None):
        return self.fit_transform(df, inp)

    def fit_transform(self, df):
        super().fit_transform(df)
        self.inp = df.columns if self.inp is None else self.inp
        # TODO: [col for col in inp if col in df.columns]
        self.mean = df[self.inp].mean()
        df[self.inp] = df[self.inp].fillna(self.mean)
        return df

    def transform(self, df):
        df[self.inp] = df[self.inp].fillna(self.mean)
        return df


class FillnaMedian(InplaceOperation):
    def __init__(self, inp=None):
        super().__init__(inp)
        self.median = None

    @classmethod
    def description(cls):
        return "Fill missing values with median inplace"

    def __call__(self, df, inp=None):
        return self.fit_transform(df, inp)

    def fit_transform(self, df):
        super().fit_transform(df)
        self.median = df[self.inp].median()
        df[self.inp] = df[self.inp].fillna(self.median)
        return df

    def transform(self, df):
        df[self.inp] = df[self.inp].fillna(self.median)
        return df


class Std(InplaceOperation):
    def __init__(self, inp=None):
        super().__init__(inp)
        self.scaler = StandardScaler()

    @classmethod
    def description(cls):
        return "Inplace standard scaling of input columns"

    def __call__(self, df, inp=None):
        return self.fit_transform(df)

    def fit_transform(self, df):
        super().fit_transform(df)
        df[self.inp] = self.scaler.fit_transform(df[self.inp])
        return df

    def transform(self, df):
        df[self.inp] = self.scaler.transform(df[self.inp])
        return df


# TODO: here we can create scaler subclass
class Minmax(InplaceOperation):
    def __init__(self, inp=None):
        super().__init__(inp)
        self.scaler = MinMaxScaler()

    @classmethod
    def description(cls):
        return "Inplace minmax scaling of input columns"

    def __call__(self, df, inp=None):
        return self.fit_transform(df)

    def fit_transform(self, df):
        super().fit_transform(df)
        df[self.inp] = self.scaler.fit_transform(df[self.inp])
        return df

    def transform(self, df):
        df[self.inp] = self.scaler.transform(df[self.inp])
        return df


class FrequencyEncoding(InplaceOperation):
    def __init__(self, inp=None):
        super().__init__(inp)

    @classmethod
    def description(cls):
        return "Frequency encoding of categorical features"

    def __call__(self, df):
        return self.fit_transform(df)

    def fit_transform(self, df):
        super().fit_transform(df)
        for col in self.inp:
            df[col] = df.groupby(col)[col].transform("count")
        return df

    def transform(self, df):
        # TODO: here, for shorter test set, we will get something else.
        #  also, if there are matching entities, no info will be captured.
        #  need to be fixed
        for col in self.inp:
            df[col] = df.groupby(col)[col].transform("count")
        return df


class Binning(InplaceOperation):
    def __init__(self, inp=None, bins=10):
        super().__init__(inp)
        self.bins = bins

    @classmethod
    def description(cls):
        return "Binning of numerical features"

    def __call__(self, df):
        return self.fit_transform(df)

    def fit_transform(self, df):
        super().fit_transform(df)
        for col in self.inp:
            df[col] = pd.qcut(df[col], 10, duplicates="drop")
        return df

    def transform(self, df):
        # TODO: here, for test set, binning will be different. need to be fixed
        for col in self.inp:
            df[col] = pd.qcut(df[col], 10, duplicates="drop")
        return df


class LabelEncoding(InplaceOperation):
    def __init__(self, inp=None):
        super().__init__(inp)
        self.label_encoders = []

    @classmethod
    def description(cls):
        return "Label encoding of categorical features"

    def __call__(self, df):
        return self.fit_transform(df)

    def fit_transform(self, df):
        super().fit_transform(df)
        for col in self.inp:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            self.label_encoders.append(encoder)
        return df

    def transform(self, df):
        for col, encoder in zip(self.inp, self.label_encoders):
            df[col] = encoder.transform(df[col])
        return df


class OneHotEncoding(InplaceOperation):
    def __init__(self, inp=None):
        super().__init__(inp)

    @classmethod
    def description(cls):
        return "One hot encoding of categorical features"

    def __call__(self, df):
        return self.fit_transform(df)

    def fit_transform(self, df):
        super().fit_transform(df)
        for col in self.inp:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
            df.drop(columns=[col], inplace=True)
        return df

    def transform(self, df):
        for col in self.inp:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
            df.drop(columns=[col], inplace=True)
        return df


# operation types: unary, binary, n-ary
# here wrong stuff happens: we initialize all classes and all inner variables are always the same
# TODO: fix this
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
