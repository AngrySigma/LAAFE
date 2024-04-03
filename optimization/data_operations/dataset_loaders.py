from dataclasses import dataclass
from typing import Iterable, Iterator

import openml
import pandas as pd
from pandera.typing import Series

@dataclass
class OpenMLDataset:
    data: pd.DataFrame
    target: Series[int | str]
    collection_date: str
    creator: str
    default_target_attribute: str
    description: str
    features: dict[str, dict[str, str]]
    language: str
    name: str
    qualities: dict[str, str]

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
        collection_date = dataset.collection_date
        creator = dataset.creator
        default_target_attribute = dataset.default_target_attribute
        description = dataset.description
        features = dataset.features
        language = dataset.language
        name = dataset.name
        qualities = dataset.qualities
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
