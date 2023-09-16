import importlib


def get_dataset(alias):
    datamodule = importlib.import_module("datasets." + alias.lower())
    return datamodule.Dataset
