from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype
from pandas.core.dtypes.common import is_bool_dtype
from sklearn.decomposition import PCA
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)


class PipelineNode(ABC):
    @classmethod
    @abstractmethod
    def description(cls) -> str:
        pass

    def __str__(self) -> str:
        return self.description()

    def __repr__(self) -> str:
        return self.description()


class Operation(PipelineNode, ABC):
    def __init__(self, inp: str | list[str] | None = None) -> None:
        # self.inp: list[str] | None = [inp] if isinstance(inp, str) else inp
        self.inp: list[str] = (
            inp if isinstance(inp, list) else [inp] if isinstance(inp, str) else []
        )

    @abstractmethod
    def __call__(self, df: DataFrame) -> DataFrame:
        pass

    def bools_to_numeric(self, df):
        for col in self.inp:
            if is_bool_dtype(df[col]):
                df[col] = int(df.col)
        return df

    @abstractmethod
    def fit_transform(self, df: DataFrame) -> DataFrame:
        assert all(
            [e in df.columns for e in self.inp]
        ), f"{self.__class__.__name__}: Columns are not in df"
        self.set_inp(df)
        return df

    def set_inp(self, df: DataFrame) -> None:
        # inp = self.inp
        if not self.inp:
            self.inp = list(df.columns)
        # return inp

    @abstractmethod
    def transform(self, df: DataFrame) -> DataFrame:
        assert all(
            [e in df.columns for e in self.inp]
        ), f"{self.__class__.__name__}: Columns are not in df"


class Drop(Operation):
    @classmethod
    def description(cls) -> str:
        return "Drop input columns inplace"

    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    def transform(self, df: DataFrame) -> DataFrame:
        super().transform(df)
        df.drop(columns=self.inp, inplace=True)
        return df

    def fit_transform(self, df: DataFrame) -> DataFrame:
        self.transform(df)
        return df


# TODO: add subclass arithmetic
class Add(Operation):
    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    @classmethod
    def description(cls) -> str:
        return (
            "Add two input columns together to a new column "
            '"add_{number of previous add operations + 1}"'
        )

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        df = self.bools_to_numeric(df)
        assert is_numeric_dtype(df[self.inp]), "Summation error: column is not numeric"
        num = len(df.filter(regex=r"^add_[\d]+").columns)
        df[f"add_{num}"] = df[self.inp].sum(axis=1)
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)


class Sub(Operation):
    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    @classmethod
    def description(cls) -> str:
        return (
            "Subtract two input columns together to a new column"
            ' "sub_{number of previous sub operations + 1}"'
        )

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        df = self.bools_to_numeric(df)
        assert is_numeric_dtype(df[self.inp]), "Summation error: column is not numeric"
        if isinstance(self.inp, str):
            self.inp = [self.inp]
        if self.inp is None:
            self.inp = list(df.columns)
        num = len(df.filter(regex=r"^sub_[\d]+").columns)
        df[f"sub_{num}"] = df[self.inp[0]].sub([self.inp[1]])
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)


class Mul(Operation):
    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    @classmethod
    def description(cls) -> str:
        return (
            "Multiply two input columns together to a new column"
            ' "mul_number of previous mul operations + 1"'
        )

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        df = self.bools_to_numeric(df)
        assert is_numeric_dtype(df[self.inp]), "Summation error: column is not numeric"
        num = len(df.filter(regex=r"^mul_[\d]+").columns)
        df[f"mul_{num}"] = df[self.inp].cumprod(axis=1)[self.inp[-1]]
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)


class Div(Operation):
    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    @classmethod
    def description(cls) -> str:
        return (
            "Divide two input columns together to a new column"
            ' "div_{number of previous div operations + 1}"'
        )

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        assert len(self.inp) > 1, "Too few input columns passed, at least 2 required"
        assert is_numeric_dtype(df[self.inp]), "Column values are not numeric"
        num = len(df.filter(regex=r"^div_[\d]+").columns)
        df[f"div_{num}"] = df[self.inp[0]].div(df[self.inp[1]], fill_value=0)
        df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)


class Pca(Operation):
    def __init__(self, inp: str | list[str] | None = None) -> None:
        super().__init__(inp)
        self.pca = PCA(0.95)

    @classmethod
    def description(cls) -> str:
        return "Create new columns pca_0, pca_1 ... from PCA on input columns"

    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        assert is_numeric_dtype(df[self.inp]), "PCA error: column is not numeric"
        # TODO: fill na if there are any. moreover, this should be added to pipeline
        # self.inp = df.columns if self.inp is None else self.inp
        pca_result = self.pca.fit_transform(df[self.inp])
        pca_columns = [f"pca_{i}" for i in range(pca_result.shape[1])]
        df[pca_columns] = pca_result
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        pca_result = self.pca.transform(df[self.inp])
        pca_columns = [f"pca_{i}" for i in range(pca_result.shape[1])]
        df[pca_columns] = pca_result
        return df


class FillnaMean(Operation):
    def __init__(self, inp: str | list[str] | None = None) -> None:
        super().__init__(inp)
        self.mean: "pd.Series[float]" | None = None

    @classmethod
    def description(cls) -> str:
        return "Fill missing values with mean inplace"

    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        df = self.bools_to_numeric(df)
        assert is_numeric_dtype(
            df[self.inp]
        ), "Std scaling error: column is not numeric"
        mean = df[self.inp].mean().astype("float")
        self.mean = mean
        df[self.inp] = df[self.inp].infer_objects(copy=False).fillna(self.mean)
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        df[self.inp] = df[self.inp].infer_objects(copy=False).fillna(self.mean)
        return df


class FillnaMedian(Operation):
    def __init__(self, inp: str | list[str] | None = None) -> None:
        super().__init__(inp)
        self.median: "pd.Series[float]" | None = None

    @classmethod
    def description(cls) -> str:
        return "Fill missing values with median inplace"

    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        df = self.bools_to_numeric(df)
        assert is_numeric_dtype(df[self.inp]), "Median error: column is not numeric"
        self.median = df[self.inp].median().astype("float")
        df[self.inp] = df[self.inp].infer_objects(copy=False).fillna(self.median)
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        df[self.inp] = df[self.inp].infer_objects(copy=False).fillna(self.median)
        return df


class Std(Operation):
    def __init__(self, inp: str | list[str] | None = None) -> None:
        super().__init__(inp)
        self.scaler = StandardScaler()

    @classmethod
    def description(cls) -> str:
        return "Inplace standard scaling of input columns"

    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        df = self.bools_to_numeric(df)
        assert is_numeric_dtype(
            df[self.inp]
        ), "Std scaling error: column is not numeric"
        df[self.inp] = self.scaler.fit_transform(df[self.inp])
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        df[self.inp] = self.scaler.transform(df[self.inp])
        return df


# TODO: here we can create scaler subclass
class Minmax(Operation):
    def __init__(self, inp: str | list[str] | None = None) -> None:
        super().__init__(inp)
        self.scaler = MinMaxScaler()

    @classmethod
    def description(cls) -> str:
        return "Inplace minmax scaling of input columns"

    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        assert is_numeric_dtype(
            df[self.inp]
        ), "Minmax scaling error: column is not numeric"
        df[self.inp] = self.scaler.fit_transform(df[self.inp])
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        df[self.inp] = self.scaler.transform(df[self.inp])
        return df


class FrequencyEncoding(Operation):
    @classmethod
    def description(cls) -> str:
        return "Frequency encoding of categorical features"

    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        for col in self.inp:
            df[col] = df.groupby(col)[col].transform("count")
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        # TODO: here, for shorter test set, we will get something else.
        #  also, if there are matching entities, no info will be captured.
        #  need to be fixed
        for col in self.inp:
            df[col] = df.groupby(col)[col].transform("count")
        return df


class Binning(Operation):
    def __init__(self, inp: list[str] | None = None, bins_num: int = 10) -> None:
        super().__init__(inp)
        self.bins_num = bins_num
        self.bins: dict[str, Any] = {}

    @classmethod
    def description(cls) -> str:
        return "Binning of numerical features. Inplace operation"

    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        assert is_numeric_dtype(df[self.inp]), "Binning error: column is not numeric"
        for col in self.inp:
            df[col], self.bins[col] = pd.qcut(
                df[col], 10, duplicates="drop", retbins=True, labels=False
            )
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        # TODO: here, for test set, binning will be different. need to be fixed
        for col in self.inp:
            df[col] = pd.cut(df[col], self.bins[col], duplicates="drop", labels=False)
        return df


class LabelEncoding(Operation):
    def __init__(self, inp: list[str] | None = None) -> None:
        super().__init__(inp)
        self.label_encoders: dict[str, LabelEncoder] = {}

    @classmethod
    def description(cls) -> str:
        return "Label encoding of categorical features. Inplace operation"

    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        for col in self.inp:
            encoder = LabelEncoder()
            df[col] = encoder.fit_transform(df[col])
            self.label_encoders[col] = encoder
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        for col in self.inp:
            if df[col].dtype.name == "category":
                df[col] = df[col].astype("object")
            present = df[col].isin(self.label_encoders[col].classes_)
            proxy = pd.Series(index=df[col].index, data=-1)
            proxy[present] = self.label_encoders[col].transform(df[col][present])
            df[col] = proxy
        return df


class OneHotEncoding(Operation):
    def __init__(self, inp):
        super().__init__(inp)
        self.encoder = OneHotEncoder(sparse_output=False)

    @classmethod
    def description(cls) -> str:
        return "One hot encoding of categorical features"

    def __call__(self, df: DataFrame) -> DataFrame:
        return self.fit_transform(df)

    def fit_transform(self, df: DataFrame) -> DataFrame:
        super().fit_transform(df)
        data = df[self.inp]
        self.encoder.fit(data)
        assert len(self.encoder.get_feature_names_out()) < 100, "Too many features"
        data = pd.DataFrame(
            columns=self.encoder.get_feature_names_out(),
            data=self.encoder.transform(data),
            index=df.index,
        )
        df.drop(columns=self.inp, inplace=True)
        df = pd.concat([df, data], axis=1)
        return df

    def transform(self, df: DataFrame) -> DataFrame:
        for col in self.inp:
            items = pd.Series(
                [col + "_" + item for item in df[col].unique().astype(str)]
            )
            assert all(
                items.isin(self.encoder.get_feature_names_out())
            ), "Some input features are not present in encoder"
        data = df[self.inp]
        data = pd.DataFrame(
            columns=self.encoder.get_feature_names_out(),
            data=self.encoder.transform(data),
            index=df.index,
        )
        df.drop(columns=self.inp, inplace=True)
        df = pd.concat([df, data], axis=1)
        return df
