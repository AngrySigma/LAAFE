import openml


def load_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=True, download_features_meta_data=True)
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
    return data, target, collection_date, creator, default_target_attribute, description, features, language, name, qualities


if __name__ == '__main__':
    data, target, collection_date, creator, default_target_attribute, description, features, language, name, qualities = load_dataset(61)
    print(data)
    print(target)
    print(collection_date)
    print(creator)
    print(default_target_attribute)
    print(description)
    print(features)
    print(language)
    print(name)
    print(qualities)