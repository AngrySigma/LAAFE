import openml
from dataclasses import dataclass

import pandas as pd


@dataclass
class OpenMLDataset:
    data: pd.DataFrame
    target: pd.Series
    collection_date: str
    creator: str
    default_target_attribute: str
    description: str
    features: dict
    language: str
    name: str
    qualities: dict

    def __repr__(self):
        return (f'\nDataset name: {self.name}'
                f'\nDescription:\n{self.description}'
                f'\nData example:\n{self.data[:5]}'
                f'\nTarget:\n{self.target[:5]}'
                f'\nCollection_date: {self.collection_date}'
                )


class DatasetLoader:
    def __init__(self, dataset_ids):
        self.dataset_ids = dataset_ids

    @staticmethod
    def load_dataset(dataset_id):
        dataset = openml.datasets.get_dataset(dataset_id, download_data=True,
                                              download_qualities=True,
                                              download_features_meta_data=True)
        data, target, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format='dataframe',
            target=dataset.default_target_attribute
        )
        collection_date = dataset.collection_date
        creator = dataset.creator
        default_target_attribute = dataset.default_target_attribute
        description = dataset.description
        features = dataset.features
        language = dataset.language
        name = dataset.name
        qualities = dataset.qualities
        return OpenMLDataset(data, target, collection_date, creator,
                             default_target_attribute, description, features,
                             language, name, qualities)

    def __getitem__(self, item):
        return self.load_dataset(self.dataset_ids[item])

    def __len__(self):
        return len(self.dataset_ids)


if __name__ == '__main__':
    dataset_loader = DatasetLoader(dataset_ids=[61])
    dataset = dataset_loader[0]
    print(dataset)
