from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import openml
import pandas as pd
from numpy import ndarray
from openml import OpenMLDataFeature


@dataclass
class OpenMLDataset:
    data: pd.DataFrame
    target: ndarray[Any, Any] | Any
    collection_date: str
    creator: str
    default_target_attribute: str
    description: str
    features: dict[int, OpenMLDataFeature]
    language: str
    name: str
    qualities: dict[str, float]

    def __repr__(self) -> str:
        return (
            f"Dataset description will be provided further."
            f"\nSTART DATASET DESCRIPTION"
            f"\nDataset name: {self.name}"
            f"\nDescription:\n{self.description}"
            f"\nData columns: {self.features}"
            f"\nData example:\n{self.data[:5]}"
            f"\nTarget:\n{self.target[:5]}"
            f"\nDataset qualities: {self.qualities}"
            f"\nEND DATASET DESCRIPTION"
        )

    def __str__(self) -> str:
        return self.__repr__()


class DatasetLoader(Iterable[OpenMLDataset]):
    def __init__(self, dataset_ids: list[int]) -> None:
        super().__init__()
        self.dataset_ids = dataset_ids

    @staticmethod
    def load_dataset(dataset_id: int) -> OpenMLDataset:
        dataset = openml.datasets.get_dataset(
            dataset_id,
            download_data=True,
            download_qualities=True,
            download_features_meta_data=True,
        )
        data, target, _, _ = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        collection_date = (
            dataset.collection_date if dataset.collection_date else "Unknown"
        )
        creator = dataset.creator if dataset.creator else "Unknown"
        default_target_attribute = (
            dataset.default_target_attribute
            if dataset.default_target_attribute
            else "Unknown"
        )
        description = dataset.description if dataset.description else "Unknown"
        features = dataset.features
        language = dataset.language if dataset.language else "Unknown"
        name = dataset.name if dataset.name else "Unknown"
        qualities = dataset.qualities if dataset.qualities else {}
        return OpenMLDataset(
            data,
            target,
            collection_date,
            creator,
            default_target_attribute,
            description,
            features,
            language,
            name,
            qualities,
        )

    def __getitem__(self, item: int) -> OpenMLDataset:
        return self.load_dataset(self.dataset_ids[item])

    def __iter__(self) -> Iterator[OpenMLDataset]:
        index = 0
        while index < len(self.dataset_ids):
            yield self.load_dataset(self.dataset_ids[index])
            index += 1

    def __len__(self) -> int:
        return len(self.dataset_ids)


if __name__ == "__main__":
    dataset_loader = DatasetLoader(dataset_ids=[61])
    dataset = dataset_loader[0]
    print(dataset)
