from abc import ABC, abstractmethod
from typing import Optional

from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC

from src.optimization.data_operations import OPERATIONS
from src.optimization.data_operations.operation_pipeline import OperationPipeline


class Model(ABC):
    @abstractmethod
    def __init__(self, random_state: Optional[int | float] = None) -> None:
        pass

    @abstractmethod
    def train(
        self, data_train, target_train, data_test, target_test
    ) -> dict[str, float]:
        pass


class CatboostClassifierModel(Model):
    def __init__(self, random_state: Optional[int | float] = None) -> None:
        if random_state is not None:
            self.model = CatBoostClassifier(random_seed=random_state)
        else:
            self.model = CatBoostClassifier()

    def train(self, data_train, target_train, data_test, target_test):
        self.model.fit(data_train, target_train)
        metrics = self.model.score(data_test, target_test)
        return {"accuracy": metrics}


class LinearClassifierModel(Model):
    def __init__(self, random_state: Optional[int | float] = None) -> None:
        if random_state is not None:
            self.model = SGDClassifier(random_state=random_state)
        else:
            self.model = SGDClassifier()

    def train(self, data_train, target_train, data_test, target_test):
        self.model.fit(data_train, target_train)
        metrics = self.model.score(data_test, target_test)
        return {"accuracy": metrics}


class SVMClassifierModel(Model):
    def __init__(self, random_state: Optional[int | float] = None) -> None:
        if random_state is not None:
            self.model = SVC(random_state=random_state)
        else:
            self.model = SVC()

    def train(self, data_train, target_train, data_test, target_test):
        self.model.fit(data_train, target_train)
        metrics = self.model.score(data_test, target_test)
        return {"accuracy": metrics}


class RandomForestClassifierModel(Model):
    def __init__(self, random_state: Optional[int | float] = None) -> None:
        if random_state is not None:
            self.model = RandomForestClassifier(random_state=random_state)
        else:
            self.model = RandomForestClassifier()

    def train(self, data_train, target_train, data_test, target_test):
        err = False
        try:
            self.model.fit(data_train, target_train)
            metrics = self.model.score(data_test, target_test)
        except ValueError as e:
            # backup_pipeline = OperationPipeline(OPERATIONS)
            # backup_pipeline.build_default_pipeline(data_train)
            # backup_pipeline.fit_transform(data_train)
            # backup_pipeline.transform(data_test)
            # self.model.fit(data_train, target_train)
            # metrics = self.model.score(data_test, target_test)
            metrics = 0
            err = e
        # metrics = f'ValueError: {e}'
        # TODO: we can invoke next request for the LLM to fix the error
        return {
            "accuracy": metrics
            if not err
            else f"{metrics}, {err.__class__.__name__}: {err}"
        }
