import importlib


def get_dataset(alias):
    if alias.lower() in ['kitti', 'ddad', 'nyu', 'tum', 'boon']:
        alias = "original_dataset"
    datamodule = importlib.import_module("datasets." + alias.lower())
    return datamodule.Dataset
