import importlib


def get_model(alias, mode="train"):
    module = importlib.import_module("models." + alias)
    return module.Model
