from path import Path
import os
import numpy as np
from imageio import imread

import torch

from .base_dataset import Dataset as base_dataset
import configs
from . import custom_transforms


class Dataset(base_dataset):
    @classmethod
    def add_arguments(cls, parser):
        parser.add_argument('--sequence_length', type=int, default=3,
                            help='number of images for training')
        parser.add_argument("--img_height", type=int, default=512,
                            help="Image height")
        parser.add_argument("--img_width", type=int, default=2048,
                            help="Image width")
        # TODO num_threads
        parser.add_argument("--val_frac", type=float, default=0.1,
                            help="Fraction of data to use for validation")

        parser.add_argument('--repeat', type=int, default=1,
                            help='number of repeatition')
        parser.add_argument('--skip_frames', type=int, default=1,
                            help='jump sampling from video')
        parser.add_argument('--use_frame_index', action='store_true',
                            help='filter out static-camera frames in video')  # already done by SfM
        return parser, set()

    def __init__(self, opt, mode="train"):
        super().__init__(opt, mode)
        self.mode = mode
        assert mode in ("train", "vali")

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
            self.data_path = Path(data_root) / "training"
            scene_list_path = self.data_path / "train.txt"
            self.scenes = [
                self.data_path / folder for folder in open(scene_list_path)
            ]
            self.k = opt.skip_frames
            self.use_frame_index = opt.use_frame_index
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
            self.data_path = Path(data_root) / "training"
            scene_list_path = self.data_path / "val.txt"
            self.scenes = [
                self.data_path / folder for folder in open(scene_list_path)
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
        if self.mode == "train":
            return len(self.samples) * self.opt.repeat
        elif self.mode == "vali" and self.opt.val_mode == "depth":
            return len(self.imgs)
        return len(self.samples)

    def __getitem__(self, index):
        sample_loaded = {}
        if self.mode == "vali" and self.opt.val_mode == "depth":
            raise NotImplementedError()
        else:
            sample = self.samples[index]
            tgt_img = imread(sample["tgt_img"]).astype(np.float32)
            ref_imgs = [
                imread(ref_img).astype(np.float32) for ref_img in sample["ref_imgs"]
            ]

            if self.mode == "train":
                data_transform = self.train_transform
            elif self.mode == "vali" and self.opt.val_mode == "photo":
                data_transform = self.valid_transform
            else:
                raise NotImplemented(f"Unknown transformation")

            if self.with_pseudo_depth:
                tgt_pseudo_depth = imread(sample["tgt_pseudo_depth"]).astype(np.float32)

            if data_transform is not None:
                if self.with_pseudo_depth:
                    imgs, intrinsics = data_transform(
                        [tgt_img, tgt_pseudo_depth] + ref_imgs,
                        np.copy(sample["intrinsics"]),
                    )
                    tgt_img = imgs[0]
                    tgt_pseudo_depth = imgs[1]
                    ref_imgs = imgs[2:]
                else:
                    imgs, intrinsics = data_transform(
                        [tgt_img] + ref_imgs, np.copy(sample["intrinsics"])
                    )
                    tgt_img = imgs[0]
                    ref_imgs = imgs[1:]
            else:
                intrinsics = np.copy(sample["intrinsics"])

            sample_loaded = {
                "tgt_img": tgt_img,
                "ref_imgs": ref_imgs,
                "intrinsics": intrinsics,
            }

            if self.with_pseudo_depth:
                sample_loaded.update({"tgt_pseudo_depth": tgt_pseudo_depth})

        self.convert_to_torch(sample_loaded)

        return sample_loaded

    def _crawl_train_folders(self, sequence_length):
        sequence_set = []
        for scene in self.scenes:
            imgs = sorted(scene.files("*.png"))
            intrinsics = (
                np.genfromtxt(scene / "cam.txt").astype(np.float32).reshape(3, 3)
            )

            if self.use_frame_index:
                frame_index = [int(index) for index in open(scene / "frame_index.txt")]
                imgs = [imgs[d] for d in frame_index]

            if self.with_pseudo_depth:
                pseudo_depths = sorted((scene / "leres_depth").files("*.png"))
                if self.use_frame_index:
                    pseudo_depths = [pseudo_depths[d] for d in frame_index]

            if len(imgs) < sequence_length:
                continue

            sample_index_list = _generate_sample_index(
                len(imgs), self.k, sequence_length
            )

            for sample_index in sample_index_list:
                sample = {
                    "intrinsics": intrinsics,
                    "tgt_img": imgs[sample_index["tgt_idx"]],
                }
                if self.with_pseudo_depth:
                    sample["tgt_pseudo_depth"] = pseudo_depths[sample_index["tgt_idx"]]

                sample["ref_imgs"] = []
                for j in sample_index["ref_idx"]:
                    sample["ref_imgs"].append(imgs[j])
                sequence_set.append(sample)

        self.samples = sequence_set
    
    def _crawl_vali_folders(self, folders_list, dataset):
        raise NotImplementedError()


def _generate_sample_index(num_frames, skip_frames, sequence_length):
    sample_index_list = []
    k = skip_frames
    demi_length = (sequence_length - 1) // 2
    shifts = list(range(-demi_length * k, demi_length * k + 1, k))
    shifts.pop(demi_length)

    if num_frames > sequence_length:
        for i in range(demi_length * k, num_frames - demi_length * k):
            sample_index = {"tgt_idx": i, "ref_idx": []}
            for j in shifts:
                sample_index["ref_idx"].append(i + j)
            sample_index_list.append(sample_index)

    return sample_index_list
