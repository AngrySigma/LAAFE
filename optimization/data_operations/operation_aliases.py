from abc import ABC, abstractmethod

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


class PipelineNode(ABC):
    @classmethod
    @abstractmethod
    def description(cls):
        pass

    def __str__(self):
        return self.description()

    def __repr__(self):
        return self.description()


class Operation(PipelineNode, ABC):
    def __init__(self, inp=None):
        self.inp = [inp] if isinstance(inp, str) else inp

    @abstractmethod
    def __call__(self, df):
        pass

    def fit_transform(self, df):
        self.inp = list(df.columns) if self.inp is None else list(self.inp)

    @abstractmethod
    def transform(self, df):
        pass


class Drop(Operation):
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


# TODO: add subclass arithmetic
class Add(Operation):
    def __call__(self, df):
        return self.fit_transform(df)

    @classmethod
    def description(cls):
        return 'Add two input columns together to a new column "add"'

    def fit_transform(self, df):
        super().fit_transform(df)
        df["+".join(self.inp)] = df[self.inp].sum(axis=1)
        return df

    def transform(self, df):
        return self.fit_transform(df)


class Sub(Operation):
    def __call__(self, df):
        return self.fit_transform(df)

    @classmethod
    def description(cls):
        return 'Subtract two input columns together to a new column "sub"'

    def fit_transform(self, df):
        super().fit_transform(df)
        df["-".join(self.inp)] = df[self.inp[0]].sub([self.inp[1]])
        return df

    def transform(self, df):
        return self.fit_transform(df)


class Mul(Operation):
    def __call__(self, df):
        return self.fit_transform(df)

    @classmethod
    def description(cls):
        return 'Multiply two input columns together to a new column "mul"'

    def fit_transform(self, df):
        super().fit_transform(df)
        df["*".join(self.inp)] = df[self.inp].cumprod(axis=1)[self.inp[-1]]
        return df

    def transform(self, df):
        return self.fit_transform(df)


class Div(Operation):
    def __call__(self, df):
        return self.fit_transform(df)

    @classmethod
    def description(cls):
        return 'Divide two input columns together to a new column "div"'

    def fit_transform(self, df):
        super().fit_transform(df)
        df["/".join(self.inp)] = df[self.inp[0]].div(df[self.inp[1]], fill_value=0)
        return df

    def transform(self, df):
        return self.fit_transform(df)


class Pca(Operation):
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
        # self.inp = df.columns if self.inp is None else self.inp
        pca_result = self.pca.fit_transform(df[self.inp])
        pca_columns = [f"pca_{i}" for i in range(pca_result.shape[1])]
        df[pca_columns] = pca_result
        return df

    def transform(self, df):
        pca_result = self.pca.transform(df[self.inp])
        pca_columns = [f"pca_{i}" for i in range(pca_result.shape[1])]
        df[pca_columns] = pca_result
        return df


class FillnaMean(Operation):
    def __init__(self, inp=None):
        super().__init__(inp)
        self.mean = None

    @classmethod
    def description(cls):
        return "Fill missing values with mean inplace"

    def __call__(self, df):
        return self.fit_transform(df)

    def fit_transform(self, df):
        super().fit_transform(df)
        # self.inp = df.columns if self.inp is None else self.inp
        # TODO: [col for col in inp if col in df.columns]
        self.mean = df[self.inp].mean()
        df[self.inp] = df[self.inp].fillna(self.mean)
        return df

    def transform(self, df):
        df[self.inp] = df[self.inp].fillna(self.mean)
        return df


class FillnaMedian(Operation):
    def __init__(self, inp=None):
        super().__init__(inp)
        self.median = None

    @classmethod
    def description(cls):
        return "Fill missing values with median inplace"

    def __call__(self, df, inp=None):
        return self.fit_transform(df)

    def fit_transform(self, df):
        super().fit_transform(df)
        self.median = df[self.inp].median()
        df[self.inp] = df[self.inp].fillna(self.median)
        return df

    def transform(self, df):
        df[self.inp] = df[self.inp].fillna(self.median)
        return df


class Std(Operation):
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
class Minmax(Operation):
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


class FrequencyEncoding(Operation):
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


class Binning(Operation):
    def __init__(self, inp=None, bins_num=10):
        super().__init__(inp)
        self.bins_num = bins_num
        self.bins = {}
        self.label_encoders = {}

    @classmethod
    def description(cls):
        return "Binning of numerical features"

    def __call__(self, df):
        return self.fit_transform(df)

    def fit_transform(self, df):
        super().fit_transform(df)
        for col in self.inp:
            df[col], self.bins[col] = pd.qcut(
                df[col], 10, duplicates="drop", retbins=True, labels=False
            )
            # label_encoder = LabelEncoder()
            # df[col] = label_encoder.fit_transform(df[col])
            # self.label_encoders[col] = label_encoder
        return df

    def transform(self, df):
        # TODO: here, for test set, binning will be different. need to be fixed
        for col in self.inp:
            df[col] = pd.cut(df[col], self.bins[col], duplicates="drop", labels=False)
            # df[col] = self.label_encoders[col].transform(df[col])
        return df


class LabelEncoding(Operation):
    def __init__(self, inp=None):
        super().__init__(inp)
        self.label_encoders = {}

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
            self.label_encoders[col] = encoder
        return df

    def transform(self, df):
        for col in self.inp:
            df[col] = self.label_encoders[col].transform(df[col])
        return df


class OneHotEncoding(Operation):
    @classmethod
    def description(cls):
        return "One hot encoding of categorical features"

    def __call__(self, df):
        return self.fit_transform(df)

    def fit_transform(self, df):
        super().fit_transform(df)
        for col in self.inp:
            df = pd.concat(
                [df, pd.get_dummies(df[col], prefix=col).astype(int)], axis=1
            )
            df.drop(columns=[col], inplace=True)
        return df

    def transform(self, df):
        # if there are new features in the test set, it will fail likely.
        # solution: save the dummies values and use them for test set
        return self.fit_transform(df)
