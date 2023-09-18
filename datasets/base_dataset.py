import numpy
import torch.utils.data


class Dataset(torch.utils.data.Dataset):
    @classmethod
    def add_arguments(cls, parser):
        return parser, set()

    def __init__(self, opt, mode="train"):
        super().__init__()
        self.opt = opt
        self.mode = mode

    @staticmethod
    def convert_to_torch(loaded_sample):
        for k, v in loaded_sample.items():
            if isinstance(v, numpy.ndarray):
                loaded_sample[k] = torch.from_numpy(v).float()
