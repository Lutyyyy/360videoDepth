import os

import torch
from kornia.geometry.depth import depth_to_normals

from models.NetInterface import NetInterface
from networks.DepthNet import DepthNet
from networks.PoseNet import PoseNet
import losses.loss_functions as LossF


class Model(NetInterface):
    @classmethod
    def add_arguments(cls, parser):
        # model options
        parser.add_argument('--resnet_layers', type=int, default=18, choices=[18, 34, 50, 101, 152],
                            help='number of resnet layers for depth network')
        parser.add_argument('--ImageNet_pretrained', default=True,
                            help='use pretrained weights for depth net')
        
        # training options
        parser.add_argument('--pose_lr_mul', type=float, default=1,
                            help='lr multiplier for pose network')
        parser.add_argument('--val_mode', type=str, default='depth', choices=['photo', 'depth'],
                            help='how to run validation')
        
        # loss options
        parser.add_argument('--photo_weight', type=float, default=1.0,
                            help='photometric loss weight')
        parser.add_argument('--geometry_weight', type=float, default=0.1,
                            help='geometry loss weight')
        parser.add_argument('--mask_rank_weight', type=float, default=0.1,
                            help='ranking loss with dynamic mask sampling')
        parser.add_argument('--normal_matching_weight', type=float, default=0.1,
                            help='weight for normal L1 loss')
        parser.add_argument('--normal_rank_weight', type=float, default=0.1,
                            help='edge-guided sampling for normal ranking loss')
        
        # for ablation study
        parser.add_argument('--no_ssim', action='store_true',
                            help='use ssim in photometric loss')
        parser.add_argument('--no_auto_mask', action='store_true',
                            help='masking invalid static points')
        parser.add_argument('--no_dynamic_mask',
                            action='store_true', help='masking dynamic regions')
        parser.add_argument('--no_min_optimize', action='store_true',
                            help='optimize the minimum loss')
            
        return parser, set()

    def __init__(self, opt, logger):
        super().__init__(opt, logger)

        self.input_names = [
            "tgt_img",
            "tgt_pseudo_img",
            "ref_imgs",
            "intrinsics",
            "gt_depth",
            "tgt_pseudo_depth",
        ]  # TODO
        self.gt_names = []
        self.requires = list(set().union(self.input_names, self.gt_names))

        img_resize = list()
        if opt.dataset == "kitti":
            img_resize = [256, 832]
        elif opt.dataset == "ddad":
            img_resize = [384, 640]
        elif opt.dataset in ["nyu", "tum", "bonn"]:
            img_resize = [256, 320]
        else:
            # TODO
            img_resize = [None, None]

        self.depth_net = DepthNet(opt.resnet_layers, opt.ImageNet_pretrained)
        self.pose_net = PoseNet()
        self._nets = [self.depth_net, self.pose_net]

        self.global_rank = opt.global_rank

        self.optimizer_depth = self.optim(
            self.depth_net.parameters(), lr=opt.lr, **self.optim_params
        )
        self.optimizer_pose = self.optim(
            self.pose_net.parameters(), lr=opt.lr * opt.pose_lr_mul, **self.optim_params
        )
        self._optimizers = [self.optimizer_depth, self.optimizer_pose]
        self._metrics = [
            "total_loss",
            "photo_loss",
            "geometry_loss",
            "normal_l1_loss",
            "mask_ranking_loss",
            "normal_ranking_loss",
        ]

        self.init_vars(add_path=False)

    def _train_on_batch(self, epoch, batch_idx, batch):
        for n in self._nets:
            n.zero_grad()

        self.load_batch(batch)

        pred = {}
        loss = 0
        loss_data = {}

        pred = self._predict_on_batch()
        loss, loss_data = self._calc_loss(pred)

        loss.backward()

        for k, v in pred.items():
            if torch.is_tensor(v):
                pred[k] = v.data.cpu().numpy()

        for optimizer in self._optimizers:
            optimizer.step()

        # TODO virtualization

        batch_log = {"size": self.opt.batch_size, "loss": loss.item(), **loss_data}

        return batch_log

    def _vali_on_batch(self, epoch, batch_idx, batch):
        for n in self._nets:
            n.eval()

        self.load_batch(batch)

        with torch.no_grad():
            pred = self._predict_on_batch(is_train=False)

        if self.opt.val_mode == "depth":
            errs = LossF.compute_errors(
                batch["gt_depth"], pred["tgt_depth"], self.opt.dataset
            )
            errs = {
                "abs_diff": errs[0],
                "abs_rel": errs[1],
                "a1": errs[6],
                "a2": errs[7],
                "a3": errs[8],
            }
        elif self.opt.val_mode == "photo":
            loss_1, loss_2 = LossF.photo_and_geometry_loss(
                self._input.tgt_img,
                self._input.ref_imgs,
                pred["tgt_depth"],
                pred["ref_depths"],
                self._input.intrinsics,
                pred["poses"],
                pred["poses_inv"],
                self.opt,
            )
            errs = {"photo_loss": loss_1.item(), "geometry_loss": loss_2.item()}
        else:
            raise NotImplementedError(
                f"validation mode {self.opt.val_mod} not supported"
            )

        # TODO vitualization

        batch_size = batch["tgt_img"].shape[0]
        batch_log = {"size": batch_size, **errs}

        return batch_log

    def _calc_loss(self, pred):
        photo_loss, geometry_loss, dynamic_mask = LossF.photo_and_geometry_loss(
            self._input.tgt_img,
            self._input.ref_imgs,
            pred["tgt_depth"],
            pred["ref_depths"],
            self._input.intrinsics,
            pred["poses"],
            pred["poses_inv"],
            self.opt,
        )

        normal_l1_loss = (pred["tgt_normal"] - pred["tgt_pseudo_normal"]).abs().mean()

        mask_ranking_loss = LossF.mask_ranking_loss(
            pred["tgt_depth"], self._input.tgt_pseudo_depth, dynamic_mask
        )

        normal_ranking_loss = LossF.normal_ranking_loss(
            self._input.tgt_pseudo_depth,
            self._input.tgt_img,
            pred["tgt_normal"],
            pred["tgt_pseudo_normal"],
        )

        w1 = self.opt.photo_weight
        w2 = self.opt.geometry_weight
        w3 = self.opt.normal_matching_weight
        w4 = self.opt.mask_rank_weight
        w5 = self.opt.normal_rank_weight

        total_loss = (
            w1 * photo_loss
            + w2 * geometry_loss
            + w3 * normal_l1_loss
            + w4 * mask_ranking_loss
            + w5 * normal_ranking_loss
        )

        loss_data = {
            "total_loss": total_loss.item(),
            "photo_loss": photo_loss.item(),
            "geometry_loss": geometry_loss.item(),
            "normal_l1_loss": normal_l1_loss.item(),
            "mask_ranking_loss": mask_ranking_loss.item(),
            "normal_ranking_loss": normal_ranking_loss.item(),
        }

        return total_loss, loss_data

    def _predict_on_batch(self, is_train=True):
        if is_train:
            target_depth = self.depth_net(self._input.tgt_img)
            result = {
                "tgt_depth": target_depth,
                "ref_depths": [self.depth_net(im) for im in self._input.ref_imgs],
                "poses": [
                    self.pose_net(self._input.tgt_img, im)
                    for im in self._input.ref_imgs
                ],
                "poses_inv": [
                    self.pose_net(im, self._input.tgt_img)
                    for im in self._input.ref_imgs
                ],
                "tgt_normal": depth_to_normals(
                    target_depth, self._input.intrinsics
                ),
                "tgt_pseudo_normal": depth_to_normals(
                    self._input.tgt_pseudo_depth, self._input.intrinsics
                ),
            }
        else:
            if self.opt.val_mode == "depth":
                result = {"tgt_depth": self.depth_net(self._input.tgt_img)}
            elif self.opt.val_mode == "photo":
                result = {
                    "tgt_depth": self.depth_net(self._input.tgt_img),
                    "ref_depths": [self.depth_net(im) for im in self._input.ref_imgs],
                    "poses": [
                        self.pose_net(self._input.tgt_img, im)
                        for im in self._input.ref_imgs
                    ],
                    "poses_inv": [
                        self.pose_net(im, self._input.tgt_img)
                        for im in self._input.ref_imgs
                    ],
                }
            else:
                raise NotImplementedError(
                    f"validation mode {self.opt.val_mod} not supported"
                )

        return result

    def test_on_batch(self, batch_idx, batch):
        for n in self._nets:
            n.eval()

        self.load_batch(batch)

        with torch.no_grad():
            pred = self._predict_on_batch(is_train=False)

        for k, v in pred.items():
            pred[k] = v.cpu().numpy()

        epoch_string = "best" if self.opt.epoch < 0 else "%04d" % self.opt.epoch
        outdir = os.path.join(self.opt.output_dir, "epoch%s_test" % epoch_string)
        if not hasattr(self, "outdir"):
            self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
