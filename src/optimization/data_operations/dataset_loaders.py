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
        return f'OpenMLDataset(data={self.data}, target={self.target}, collection_date={self.collection_date}, ' \
               f'creator={self.creator}, default_target_attribute={self.default_target_attribute}, ' \
               f'description={self.description}, features={self.features}, language={self.language}, ' \
               f'name={self.name}, qualities={self.qualities})'

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

    # def __next__(self):
    #     for dataset_id in self.dataset_ids:
    #         yield self.load_dataset(dataset_id)

    def __getitem__(self, item):
        return self.load_dataset(self.dataset_ids[item])

    def __len__(self):
        return len(self.dataset_ids)


if __name__ == '__main__':
    dataset_loader = DatasetLoader(dataset_ids=[61])
    dataset = dataset_loader[0]
    print('\tData example: \n', dataset.data[:5])
    print('\tTarget example: ', dataset.target[:5])
    print('\tCollection date: ', dataset.collection_date)
    print('\tCreator: ', dataset.creator)
    print('\tDefault target attribute: ', dataset.default_target_attribute)
    print('\tDescription: ', dataset.description)
    print('\tFeature names: ', dataset.features)
    print('\tDataset language: ', dataset.language)
    print('\tDataset name: ', dataset.name)
    print('\tDataset qualities: ', dataset.qualities)
