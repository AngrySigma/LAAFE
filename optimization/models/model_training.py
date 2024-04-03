from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from optimization.data_operations import OPERATIONS
from optimization.data_operations.operation_pipeline import OperationPipeline


class Model(ABC):
    @abstractmethod
    def __init__(
        self,
        random_state: Optional[int | float] = None,
        pipeline: Optional[OperationPipeline] = None,
    ) -> None:
        self.random_state = random_state
        self.pipeline = pipeline

    @abstractmethod
    def train(
        self,
        data_train: pd.DataFrame,
        target_train: pd.DataFrame,
        data_test: pd.DataFrame,
        target_test: pd.DataFrame,
    ) -> dict[str, float]:
        pass


class CatboostClassifierModel(Model):
    def __init__(self, random_state: Optional[int | float] = None) -> None:
        if random_state is not None:
            self.model = CatBoostClassifier(random_seed=random_state)
        else:
            self.model = CatBoostClassifier()

    def train(
        self,
        data_train: pd.DataFrame,
        target_train: pd.DataFrame,
        data_test: pd.DataFrame,
        target_test: pd.DataFrame,
    ) -> dict[str, float]:
        self.model.fit(data_train, target_train)
        metrics = self.model.score(data_test, target_test)
        return {"accuracy": metrics}


class LinearClassifierModel(Model):
    def __init__(self, random_state: Optional[int | float] = None) -> None:
        if random_state is not None:
            self.model = SGDClassifier(random_state=random_state)
        else:
            self.model = SGDClassifier()

    def train(
        self,
        data_train: pd.DataFrame,
        target_train: pd.DataFrame,
        data_test: pd.DataFrame,
        target_test: pd.DataFrame,
    ) -> dict[str, float]:
        self.model.fit(data_train, target_train)
        metrics = self.model.score(data_test, target_test)
        return {"accuracy": metrics}


class SVMClassifierModel(Model):
    def __init__(self, random_state: Optional[int | float] = None) -> None:
        if random_state is not None:
            self.model = SVC(random_state=random_state)
        else:
            self.model = SVC()

    def train(
        self,
        data_train: pd.DataFrame,
        target_train: pd.DataFrame,
        data_test: pd.DataFrame,
        target_test: pd.DataFrame,
    ) -> dict[str, float]:
        self.model.fit(data_train, target_train)
        metrics = self.model.score(data_test, target_test)
        return {"accuracy": metrics}


class RandomForestClassifierModel(Model):
    def __init__(
        self,
        random_state: Optional[int | float] = None,
        pipeline: Optional[OperationPipeline] = None,
    ) -> None:
        super().__init__(random_state, pipeline=pipeline)
        self.model = RandomForestClassifier(
            random_state=random_state if random_state is not None else None
        )

    def train(
        self,
        data_train: pd.DataFrame,
        target_train: pd.DataFrame,
        data_test: pd.DataFrame,
        target_test: pd.DataFrame,
    ) -> dict[str, float]:
        try:
            self.model.fit(data_train, target_train)
            metrics = self.model.score(data_test, target_test)
        except ValueError:
            # self.pipeline.build_default_pipeline(data_train)
            # self.pipeline.fit_transform(data_train)
            # self.pipeline.transform(data_test)
            backup_pipeline = OperationPipeline(OPERATIONS)
            backup_pipeline.build_default_pipeline(data_train)
            backup_pipeline.fit_transform(data_train)
            backup_pipeline.transform(data_test)
            for column in data_train:
                if column not in data_test.columns:
                    data_test[column] = 0
            for column in data_test:
                if column not in data_train.columns:
                    data_test.drop(columns=column, inplace=True)
            data_test = data_test[data_train.columns]

            self.model.fit(data_train, target_train)
            metrics = self.model.score(data_test, target_test)
        # TODO: we can invoke next request for the LLM to fix the error
        return {
            "accuracy": metrics
            # if not err
            # else f"{metrics}, {err.__class__.__name__}: {err}"
        }
