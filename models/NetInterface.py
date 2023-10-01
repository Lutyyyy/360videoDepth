import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import FloatTensor
import torch.distributed as dist
from torch.nn import init

from util.util_print import str_warning
from loggers.loggers import _LogCumulator


class NetInterface(object):
    """base class of all Model Interface
    all derived classes should overwrite __init__ and _train_on_batch(),
    and defines variables '_moveable_vars', '_nets', '_metrics' and '_optimizers'.

    Requirements for derived class:
    Variables:
        '_moveable_vars' for cuda();
        '_nets' and '_optimizers' for saving;
        '_metrics' for logging
    batch_log: all _train_on_batch() and _eval_on_batch() should return a batch_log for logging.
               All values in the batch_log are considerred as sample-wise mean.
    This log must have the key:
        'loss': the standard loss for choosing best performing eval model
        'size': the batch size for the batch
    """

    @classmethod
    def add_arguments(cls, parser):
        unique_params = set()
        return parser, unique_params

    def __init__(self, opt, logger):
        self.optim_params = dict()
        if opt.optim == "adam":
            self.optim = optim.Adam
            self.optim_params["betas"] = (opt.adam_beta1, opt.adam_beta2)
        elif opt.optim == "sgd":
            self.optim = optim.SGD
            self.optim_params["momentum"] = opt.sgd_momentum
            self.optim_params["dampening"] = opt.sgd_dampening
            self.optim_params["weight_decay"] = opt.wdecay
        else:
            raise NotImplementedError("optimizer %s has not been added yet" % opt.optim)

        self._internal_logger = _LogCumulator()
        logger.add_logger(self._internal_logger)
        self._logger = logger

        self.opt = opt
        self.full_logdir = opt.full_logdir
        self._nets = []
        self._moveable_vars = []
        self._optimizers = []

        self.input_names = []
        self._input = lambda: None
        self.gt_names = []
        self._gt = lambda: None

    def init_vars(self, add_path=True):
        # for net in self._nets:
        #    self._moveable_vars.append(net)
        for name in self.input_names:
            setattr(self._input, name, FloatTensor())
            self._moveable_vars.append("_input." + name)
            if add_path:
                setattr(self._input, name + "_path", None)
        for name in self.gt_names:
            setattr(self._gt, name, FloatTensor())
            self._moveable_vars.append("_gt." + name)
            if add_path:
                setattr(self._gt, name + "_path", None)

    def init_weight(
        self,
        net=None,
        init_type="kaiming",
        init_param=0.02,
        a=0,
        turnoff_tracking=False,
    ):
        """
        This is borrowed from Junyan
        """

        def init_func(m, init_type=init_type):
            classname = m.__class__.__name__
            if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
            ):
                if init_type == "normal":
                    init.normal_(m.weight.data, 0.0, init_param)
                elif init_type == "xavier":
                    init.xavier_normal_(m.weight.data, gain=init_param)
                elif init_type == "kaiming":
                    init.kaiming_normal_(m.weight.data, a=a, mode="fan_in")
                elif init_type == "orth":
                    init.orthogonal_(m.weight.data, gain=init_param)
                else:
                    raise NotImplementedError(
                        "initialization method [%s] is not implemented" % init_type
                    )
                if hasattr(m, "bias") and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find("BatchNorm") != -1:
                if m.affine:
                    init.normal_(m.weight.data, 1.0, init_param)
                    init.constant_(m.bias.data, 0.0)
                if turnoff_tracking:
                    m.track_running_stats = False

        if net is not None:
            net.apply(init_func)
        else:
            self.net.apply(init_func)

    def move_vars_to(self, device, non_blocking=False):
        for v in self._moveable_vars:
            if isinstance(v, str):
                if "." in v:
                    var_type, var_name = v.split(".")
                    var = getattr(getattr(self, var_type), var_name)
                    if type(var) == list:
                        # currently ignore lists.
                        mvar = [v.to(device, non_blocking=non_blocking) for v in var]
                    else:
                        mvar = var.to(device, non_blocking=non_blocking)
                    setattr(getattr(self, var_type), var_name, mvar)
                else:
                    setattr(
                        self, v, getattr(self, v).to(device, non_blocking=non_blocking)
                    )
            else:
                v.to(device, non_blocking=non_blocking)

    def load_batch(self, batch, include_gt=True):
        for name in self.input_names:
            if name in batch.keys():
                var = batch[name]
                setattr(self._input, name, var)
            if name + "_path" in batch.keys():
                setattr(self._input, name + "_path", batch[name + "_path"])
        if include_gt:
            for name in self.gt_names:
                if name in batch.keys():
                    var = batch[name]
                    setattr(self._gt, name, var)
                if name + "_path" in batch.keys():
                    setattr(self._gt, name + "_path", batch[name + "_path"])
        self.move_vars_to(self.device, True)

    def _train_on_batch(self, epoch, batch_idx, batch):
        """function that trains the model over one batch and return batch_log (including size and loss)"""
        raise NotImplementedError

    def _vali_on_batch(self, epoch, batch_idx, batch):
        """function that trains the model over one batch and return batch_log (including size and loss)"""
        raise NotImplementedError

    def test_on_batch(self, batch_idx, batch):
        raise NotImplementedError

    def _register_tensorboard(self, tblogger):
        self.tensorboard_logger = tblogger

    def train_epoch(
        self,
        dataloader,
        *,
        dataloader_vali=None,
        max_batches_per_train=None,
        max_batches_per_vali=None,
        epochs=1,
        initial_epoch=1,
        verbose=1,
        vali_at_start=False,
        global_rank=0,
        train_epoch_callback=None
    ):
        """
        Train the model with given dataloader and run evaluation with the given dataloader_vali
        max_batches_per_train: limit the number of batches for each epoch
        """
        logger = self._logger
        steps_per_epoch = len(dataloader)
        samples_per_epoch = _get_num_samples(dataloader)
        if max_batches_per_train is not None:
            steps_per_epoch = min(max_batches_per_train, steps_per_epoch)
            samples_per_epoch = min(
                samples_per_epoch, steps_per_epoch * dataloader.batch_sampler.batch_size
            )
        self.opt.epoch_batches = steps_per_epoch
        if dataloader_vali is not None:
            steps_per_eval = len(dataloader_vali)
            samples_per_eval = _get_num_samples(dataloader_vali)
            if max_batches_per_vali is not None:
                steps_per_eval = min(steps_per_eval, max_batches_per_vali)
                samples_per_eval = min(
                    samples_per_eval,
                    steps_per_eval * dataloader_vali.batch_sampler.batch_size,
                )
        else:
            steps_per_eval = 0
            samples_per_eval = 0

        logger.set_params(
            {
                "epochs": epochs + initial_epoch - 1,
                "steps": steps_per_epoch,
                "steps_eval": steps_per_eval,
                "samples": samples_per_epoch,
                "samples_eval": samples_per_eval,
                "verbose": 1,
                "metrics": self._metrics,
            }
        )

        logger.set_model(self)
        logger.on_train_begin()

        # define train closure
        def _train(epoch):
            nonlocal steps_per_epoch
            nonlocal samples_per_epoch
            nonlocal steps_per_eval
            nonlocal samples_per_eval

            self.train()  # Set Net to train mode
            logger.train()
            logger.on_epoch_begin(epoch)

            end = time.time()
            for i, data in enumerate(dataloader):
                if i >= steps_per_epoch:
                    break

                start_time = time.time()
                data_time = time.time() - end

                logger.on_batch_begin(i)

                batch_log = self._train_on_batch(epoch, i, data)
                if batch_log is None:
                    raise ValueError(
                        "Batch log is not returned by _train_on_batch method. Aborting."
                    )

                batch_log["batch"] = i
                batch_log["epoch"] = epoch
                batch_log["data_time"] = data_time
                batch_log["batch_time"] = time.time() - start_time
                logger.on_batch_end(i, batch_log)
                end = time.time()

            epoch_log = self._internal_logger.get_epoch_log()
            if self.opt.multiprocess_distributed:
                sync_log = {}
                for k, v in epoch_log.items():
                    v = torch.tensor(v).cuda()
                    dist.reduce(v, 0, async_op=False)
                    v = v.cpu().numpy() / (self.opt.world_size * self.opt.ngpus)
                    sync_log[k] = v
                logger.on_epoch_end(epoch, sync_log)
            else:
                logger.on_epoch_end(epoch, epoch_log)

        # define vali closure
        def _vali(epoch):
            self.eval()
            logger.eval()
            dataiter = iter(dataloader_vali)
            logger.on_epoch_begin(epoch)

            for i in range(steps_per_eval):
                start_time = time.time()
                data = next(dataiter)
                data_time = time.time() - start_time
                logger.on_batch_begin(i)
                batch_log = self._vali_on_batch(epoch, i, data)
                batch_log["batch"] = i
                batch_log["epoch"] = epoch
                batch_log["data_time"] = data_time
                batch_log["batch_time"] = time.time() - start_time
                logger.on_batch_end(i, batch_log)

            epoch_log = self._internal_logger.get_epoch_log()
            if self.opt.multiprocess_distributed:
                sync_log = {}
                for k, v in epoch_log.items():
                    v = torch.tensor(v).cuda()
                    dist.reduce(v, 0, async_op=False)
                    v = v.cpu().numpy() / (self.opt.world_size * self.opt.ngpus)
                    sync_log[k] = v
                logger.on_epoch_end(epoch, sync_log)
            else:
                logger.on_epoch_end(epoch, epoch_log)

        # Run actual training
        if vali_at_start:
            if dataloader_vali is None:
                raise ValueError(
                    "eval_at_beginning is set to True but no eval data is given."
                )
            _vali(initial_epoch - 1)
        for epoch in range(initial_epoch, initial_epoch + epochs):
            _train(epoch)
            if train_epoch_callback is not None:
                train_epoch_callback(epoch)
            if dataloader_vali is not None:
                _vali(epoch)

        logger.on_train_end()

    def to(self, device):
        for net in self._nets:
            net.to(device)
        self.device = device

    def cpu(self):
        for net in self._nets:
            net.cpu()

    def cuda(self):
        for net in self._nets:
            net.cuda()

    def train(self):
        for m in self._nets:
            m.train()

    def eval(self):
        for m in self._nets:
            m.eval()

    def num_parameters(self, return_list=False):
        nparams_list = list()
        for net in self._nets:
            parameters = list(net.parameters())
            nparams_list.append(sum([x.numel() for x in parameters]))
        if return_list:
            return nparams_list
        return sum(nparams_list)

    def save_state_dict(self, filepath, *, save_optimizer=False, additional_values={}):
        state_dicts = dict()
        state_dicts["nets"] = [net.state_dict() for net in self._nets]
        if save_optimizer:
            state_dicts["optimizers"] = [
                optimizer.state_dict() for optimizer in self._optimizers
            ]
        for k, v in additional_values.items():
            state_dicts[k] = v
        torch.save(state_dicts, filepath)

    def load_state_dict(self, filepath, *, load_optimizer="auto"):
        state_dicts = torch.load(filepath)

        if load_optimizer == "auto":
            load_optimizer = "optimizers" in state_dicts
            if not load_optimizer:
                print(str_warning, "Model loaded without optimizer states. ")

        assert len(self._nets) == len(state_dicts["nets"])
        for i in range(len(self._nets)):
            self._nets[i].load_state_dict(state_dicts["nets"][i])

        if load_optimizer:
            assert len(self._optimizers) == len(state_dicts["optimizers"])
            for i in range(len(self._optimizers)):
                optimizer = self._optimizers[i]
                state = state_dicts["optimizers"][i]

                # load optimizer state without overwriting training hyper-parameters, e.g. lr
                _optimizer_load_state_dict(optimizer, state, keep_training_params=True)

        additional_values = {
            k: v for k, v in state_dicts.items() if k not in ("optimizers", "nets")
        }
        return additional_values

    def pack_output(self, pred, batch):
        output = {"batch_size": len(batch["tgt_img"])}

        for k, v in pred.items():
            if torch.is_tensor(v):
                output.update({v.cpu().numpy()})
            elif isinstance(v, list):
                val_ = [x.cpu().detach().numpy() for x in v]
                output.update({k: val_})
            else:
                output.update({k: v})

        return output


def _get_num_samples(dataloader):
    # import torch.utils.data.sampler as samplers
    batch_sampler = dataloader.batch_sampler
    if batch_sampler.drop_last:
        return (
            len(batch_sampler.sampler)
            // batch_sampler.batch_size
            * batch_sampler.batch_size
        )
    return len(batch_sampler.sampler)


def _optimizer_load_state_dict(optimizer, state, keep_training_params=False):
    if keep_training_params:
        oldstate = optimizer.state_dict()
        assert len(oldstate["param_groups"]) == len(state["param_groups"])
        # use oldstate to override this state
        for oldg, g in zip(oldstate["param_groups"], state["param_groups"]):
            for k, v in oldg.items():
                if k != "params":  # parameter id not overwriten
                    g[k] = v
    optimizer.load_state_dict(state)
