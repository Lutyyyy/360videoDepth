from base_dataset import Dataset as base_dataset


class Dataset(base_dataset):
    @classmethod
    def add_argument(cls, parser):
        return parser, set()

    def __init__(self, opt, mode="train"):
        super().__init__()
