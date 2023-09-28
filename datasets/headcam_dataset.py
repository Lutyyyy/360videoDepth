from path import Path
import os

from base_dataset import Dataset as base_dataset
import configs
from . import custom_transforms


class Dataset(base_dataset):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument("--dump_root", type=str, required=True,
                    help="Where to dump the data")
        parser.add_argument("--seq_length", type=int, required=True,
                            help="Length of each training sequence")
        parser.add_argument("--img_height", type=int, default=512,
                            help="Image height")
        parser.add_argument("--img_width", type=int, default=2048,
                            help="Image width")
        # TODO num_threads
        parser.add_argument("--val_frac", type=float, default=0.1,
                            help="Fraction of data to use for validation")

        parser.add_argument('--sequence_length', type=int, default=3,
                            help='number of images for training')
        parser.add_argument('--skip_frames', type=int, default=1,
                            help='jump sampling from video')
        # parser.add_argument('--use_frame_index', action='store_true',
        #                     help='filter out static-camera frames in video')  # already done by SfM
        return parser, set()

    def __init__(self, opt, mode="train"):
        super().__init__(opt, mode)
        self.mode = mode
        assert mode in ('train', 'vali')
        
        data_root = configs.dataset_root
        
        self.img_resize = [opt.img_height, opt.img_width]

        if mode == "train":
            self.train_transform = custom_transforms.Compose(
                [
                    custom_transforms.RandomHorizontalFlip(),
                    custom_transforms.RandomScaleCrop(),
                    custom_transforms.RescaleTo(self.img_resize),
                    custom_transforms.ArrayToTensor(),
                    custom_transforms.Normalize(),
                ]
            )
            # self.data_path = Path(data_root) / f"{opt.dataset}" / "training"
            # scene_list_path = self.data_path / "train.txt"
            self.scenes = [os.listdir(data_root)]
            self.k = opt.skip_frames
            # self.use_frame_index = opt.use_frame_index
            self.with_pseudo_depth = False  # if V3 else False
            self._crawl_train_folders(opt.sequence_length)
        else:
            self.valid_transform = custom_transforms.Compose(
                [
                    custom_transforms.RescaleTo(self.img_resize),
                    custom_transforms.ArrayToTensor(),
                    custom_transforms.Normalize(),
                ]
            )
            self.data_path = Path(data_root) / f"{opt.dataset}" / "training"
            scene_list_path = self.data_path / "val.txt"
            self.scenes = [
                self.data_path / folder[:-1] for folder in open(scene_list_path)
            ]
            if self.opt.val_mode == "photo":
                self.k = opt.skip_frames
                self.use_frame_index = opt.use_frame_index
                self.with_pseudo_depth = False
                self._crawl_train_folders(opt.sequence_length)
            else:
                self.imgs, self.depth = self._crawl_vali_folders(
                    self.scenes, opt.dataset
                )
                
    def __len__(self):
        pass
    
    def __getitem__(self, index):
        pass
    
    def _crawl_train_folders(self, sequence_length):
        imgs = sorted()
        pass