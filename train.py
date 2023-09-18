import sys
import os
import time
import pandas as pd

import torch
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torch.utils.data._utils

import datasets
import models
from options import options_train
from util.util_print import str_error, str_stage, str_verbose, str_warning
from util import util_loadlib as loadlib
from loggers import loggers


def main():
    # Option Parsing
    print(str_stage, "Parsing arguments")
    opt, unique_opt_params = options_train.parse()
    # Get all parse done, including subparsers
    print(opt)

    # Setting up log directory
    print(str_stage, "Setting up logging directory")
    if opt.exprdir_no_prefix:
        exprdir = ""
        exprdir += ("" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
    else:
        exprdir = "{}_{}_{}".format(opt.net, opt.dataset, opt.lr)
        exprdir += ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
    opt.exprdir = exprdir

    if opt.full_logdir is None:
        logdir = os.path.join(opt.logdir, exprdir, str(opt.expr_id))
    else:
        logdir = opt.full_logdir

    if opt.resume == 0:
        # train from scratch
        if os.path.isdir(logdir):
            # force to overwrite previous experiments, without keyborad confirmations
            if opt.force_overwrite:
                print(
                    str_warning,
                    ("removing Experiment %d at\n\t%s\n") % (opt.expr_id, logdir),
                )
                os.system("rm -rf " + logdir)
            elif opt.expr_id <= 0:
                print(
                    str_warning,
                    (
                        "Will remove Experiment %d at\n\t%s\n"
                        "Do you want to continue? (y/n)"
                    )
                    % (opt.expr_id, logdir),
                )
                need_input = True
                while need_input:
                    response = input().lower()
                    if response in ("y", "n"):
                        need_input = False
                if response == "n":
                    print(str_stage, "User decides to quit")
                    sys.exit()
                os.system("rm -rf " + logdir)
            else:
                raise ValueError(str_error + " Refuse to remove positive expr_id")
        os.system("mkdir -p " + logdir)
    else:
        if not os.path.isdir(logdir):
            print(str_warning, "training from scratch...")
            os.system("mkdir -p " + logdir)
        else:
            opt_f_old = os.path.join(logdir, "opt.pt")
            opt = options_train.overwrite(opt, opt_f_old, unique_opt_params)

    # Save opt
    if os.path.exists(os.path.join(logdir, "opt.pt")) and opt.pt_no_overwrite:
        print(
            str_warning,
            "not overwriting previous opt.pt, this should only be set when doing on the fly eval.",
        )
        pass
    else:
        torch.save(vars(opt), os.path.join(logdir, "opt.pt"))
        with open(os.path.join(logdir, "opt.txt"), "w") as fout:
            for k, v in vars(opt).items():
                fout.write("%20s\t%-20s\n" % (k, v))

    opt.full_logdir = logdir
    print(str_verbose, "Logging directory set to: %s" % logdir)

    # Multiprocess distributed training
    if opt.multiprocess_distributed:
        print(str_stage, "using multiprocessing distributed parallel model")
        if opt.gpu != "none":
            print(
                str_warning,
                f"ignoring the gpu set up in opt: {opt.gpu}. Will use all gpus in each node.",
            )
        ngpus = torch.cuda.device_count()
        opt.ngpus = ngpus
        opt.world_size = opt.ngpus * opt.world_size
        print(f"using {ngpus} gpus")
        mp.spawn(main_worker, nprocs=ngpus, args=(ngpus, opt))
    else:
        opt.global_rank = 0
        main_worker(None, 1, opt=opt)


def main_worker(local_rank, ngpus, opt):
    def _safe_print(*args, **kargs):
        if global_rank == 0:
            print(*args, **kargs, flush=True)

    if local_rank is None:
        print(str_stage, "Setting device")
        # Single GPU training case
        if opt.gpu == "-1":
            device = torch.device("cpu")
        else:
            loadlib.set_gpu(opt.gpu)
            device = torch.device("cuda")
        global_rank = 0  # as if this is the master node
    else:
        if opt.multiprocess_distributed:
            opt.global_rank = opt.node_rank * opt.ngpus + local_rank
            if opt.global_rank == 0:
                print(str_stage, "setting up devices...")
            loadlib.set_gpu(str(local_rank))
            device = torch.device("cuda")
            if opt.global_rank == 0:
                print(str_stage, "Setting up process groups....")
            dist.init_process_group(
                backend=opt.dist_backend,
                init_method=opt.init_url,
                world_size=opt.world_size,
                rank=opt.global_rank,
            )
            global_rank = opt.global_rank

    if opt.manual_seed is not None:
        loadlib.set_manual_seed(opt.manual_seed)

    # Setting up loggers
    logdir = opt.full_dir
    _safe_print(str_stage, "Setting up loggers")
    if opt.resume != 0 and os.path.isfile(os.path.join(logdir, "best.pt")):
        try:
            prev_best_data = torch.load(os.path.join(logdir, "best.pt"))
            prev_best = prev_best_data["loss_eval"]
            del prev_best_data
        except KeyError:
            prev_best = None
    else:
        prev_best = None

    if global_rank == 0:
        best_model_logger = loggers.ModelSaveLogger(
            os.path.join(logdir, "best.pt"),
            period=1,
            save_optimizer=True,
            save_best=True,
            prev_best=prev_best,
        )
        logger_list = [
            loggers.TerminateOnNaN(),
            loggers.ProgbarLogger(
                allow_unused_fields="all",
                interval=opt.progbar_interval,
                no_accum=opt.no_accum,
            ),
            loggers.CsvLogger(
                os.path.join(logdir, "epoch_loss.csv"), allow_unused_fields="all"
            ),
            loggers.ModelSaveLogger(
                os.path.join(logdir, "nets", "{epoch:04d}.pt"),
                period=opt.save_net,
                save_optimizer=opt.save_net_opt,
            ),
            loggers.ModelSaveLogger(
                os.path.join(logdir, "checkpoint.pt"), period=1, save_optimizer=True
            ),
            best_model_logger,
        ]

        if opt.log_batch:
            logger_list.append(
                loggers.BatchCsvLogger(
                    os.path.join(logdir, "batch_loss.csv"), allow_unused_fields="all"
                )
            )

        if opt.tensorboard:
            if opt.tensorboard_keyword != "none":
                [parent_dir, sub_dir] = logdir.split(f"/{opt.tensorboard_keyword}/")
                tf_logdir = os.path.join(
                    parent_dir, opt.tensorboard_keyword, "tensorboard", sub_dir
                )
            else:
                tf_logdir = os.path.join(
                    opt.logdir, "tensorboard", opt.exprdir, str(opt.expr_id)
                )

            if os.path.exists(tf_logdir) and opt.resume == 0:
                os.system("rm -r " + tf_logdir)

            tensorboard_logger = loggers.TensorBoardLogger(
                tf_logdir, opt.html_logger, allow_unused_fields="all"
            )
            logger_list.append(tensorboard_logger)
        logger = loggers.ComposeLogger(logger_list)

    if opt.html_logger:
        html_summary_filepath = os.path.join(opt.full_logdir, "summary")
        html_logger = loggers.HtmlLogger(html_summary_filepath)
        logger_list.append(html_logger)
    elif global_rank:
        # other procs do no log.
        logger = loggers.ComposeLogger([loggers.TerminateOnNaN()])

    # setting up models
    _safe_print(str_stage, "Setting up models")
    Model = models.get_model(opt.net)
    model = Model(opt, logger)
    if global_rank == 0 and opt.tensorboard:
        model._register_tensorboard(tensorboard_logger)

    _safe_print(str_stage, "Setting up data loaders")
    if global_rank == 0:
        start_time = time.time()

    dataset = datasets.get_dataset(opt.dataset)
    dataset_train = dataset(opt, mode="train")
    dataset_vali = dataset(opt, mode="vali")

    if opt.multiprocess_distributed:
        dist.barrier()
    _safe_print(str_stage, "data loaders set")

    initial_epoch = 1
    if opt.resume:
        if opt.resume == -1:
            # resume latest model
            net_filename = os.path.join(logdir, "checkpoint.pt")
        elif opt.resume == -2:
            # resume best model
            net_filename = os.path.join(logdir, "best.pt")
        else:
            # resume specified model
            net_filename = os.path.join(logdir, "nets", "{epoch:04d}.pt").format(
                epoch=opt.resume
            )

        if not os.path.isfile(net_filename):
            _safe_print(
                str_warning,
                ("Network file not found for opt.resume=%d. " "Starting from scratch")
                % opt.resume,
            )
        else:
            # if global_rank == 0:
            additional_values = model.load_state_dict(
                net_filename, load_optimizer="auto"
            )
            try:
                initial_epoch += additional_values["epoch"]
            except KeyError as err:
                # Old saved model does not have epoch as additional values
                print(str(err))
                epoch_loss_csv = os.path.join(logdir, "epoch_loss.csv")
                if opt.resume == -1:
                    try:
                        initial_epoch += pd.read_csv(epoch_loss_csv)["epoch"].max()
                    except pd.errors.ParserError:
                        with open(epoch_loss_csv, "r") as f:
                            lines = f.readlines()
                        initial_epoch += max([int(l.split(",")[0]) for l in lines[1:]])
                else:
                    initial_epoch += opt.resume

    # wait until model is loaded.
    if opt.multiprocess_distributed:
        dist.barrier()

    model.to(device)
    _safe_print(model)
    _safe_print("# model parameters: {:,d}".format(model.num_parameters()))

    # convert to DDP and sync params
    if opt.multiprocess_distributed:
        for net in model._nets:
            net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
            net = torch.nn.parallel.DistributedDataParallel(net)
        # sync parameters before training:
        _safe_print(str_stage, "syncing parameters....")
        for net in model._nets:
            for p in net.parameters():
                dist.broadcast(p, 0, async_op=False)

    # Setting up data loaders

    # Get custom collate function to deal with variable size inputs.
    if hasattr(Model, "collate_fn"):
        collate_fn = Model.collate_fn
    else:
        # use default collate function instead.
        collate_fn = torch.utils.data._utils.collate.default_collate

    # Setting up data sampler
    if opt.multiprocess_distributed:
        training_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_train
        )
        call_back = training_sampler.set_epoch
        training_sampler.set_epoch(opt.epoch)
    else:
        training_sampler = None
        call_back = None

    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=opt.batch_size,
        shuffle=(training_sampler is None),
        sampler=training_sampler,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    dataloader_vali = torch.utils.data.DataLoader(
        dataset_vali,
        batch_size=opt.batch_size,
        num_workers=opt.workers,
        pin_memory=True,
        drop_last=True,
        shuffle=False,
        collate_fn=collate_fn,
    )

    if global_rank == 0:
        _safe_print(
            str_verbose,
            "Time spent in data IO initialization: %.2fs" % (time.time() - start_time),
        )
        _safe_print(
            str_verbose,
            "# training points: " + str(len(dataset_train))
        )
        _safe_print(
            str_verbose,
            "# training batches per epoch: " + str(len(dataloader_train))
        )
        _safe_print(
            str_verbose,
            "# test batches: " + str(len(dataloader_vali))
        )

    # Training
    if opt.epoch > 0:
        _safe_print(str_stage, "Training")
        model.train_epoch(
            dataloader_train,
            dataloader_vali=dataloader_vali,
            max_batches_per_train=opt.epoch_batches,
            max_batches_per_vali=opt.vali_batches,
            epochs=opt.epoch,
            initial_epoch=initial_epoch,
            vali_at_start=opt.vali_at_start,
            train_epoch_callback=call_back,
        )

    if opt.test_template is not None:
        del model
        torch.cuda.empty_cache()
        with open(opt.test_template) as f:
            cmd = f.readlines()[0]
        cmd = cmd.format(suffix_expand=opt.suffix.format(**vars(opt)), **vars(opt))
        with open(os.path.join(opt.full_logdir, "test_cmd.sh"), "w") as f:
            f.write(cmd)

        from subprocess import call

        call(cmd, shell=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
