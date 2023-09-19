def set_manual_seed(seed):
    import random

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError as err:
        print("Numpy not found. Random seed for numpy not set. ")
    try:
        import torch
        import torch.backends.cudnn as cudnn

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
    except ImportError as err:
        print("Pytorch not found. Random seed for pytorch not set. ")


def set_gpu(gpu, check=True):
    import torch
    import torch.backends.cudnn as cudnn

    cudnn.benchmark = False
    torch.cuda.set_device(int(gpu))
