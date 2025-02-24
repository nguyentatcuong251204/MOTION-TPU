# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Wrapper to train and test a video classification model."""
from timesformer.utils.misc import launch_job
from timesformer.utils.parser import load_config, parse_args
# import torch.distributed as dist
from tools.test_net import test
from tools.train_net import train #,_mp_fn
# import torch_xla as xla
import timesformer.utils.logging as logging
logger = logging.get_logger(__name__)

def get_func(cfg):
    if(cfg.TRAIN.TPU_ENABLE == False):
        train_func = train
    else:
        train_func = _mp_fn
    test_func = test
    return train_func, test_func

def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    if args.num_shards > 1:
       args.output_dir = str(args.job_dir)

    # logger.info("LOAD CONFIG")
    cfg = load_config(args)

    logger.info("LOAD TRAIN TEST FUNC")
    train, test = get_func(cfg)

    logger.info("START TRAINING")
    # Perform training.
    if cfg.TRAIN.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=train)

    # Perform multi-clip testing.
    if cfg.TEST.ENABLE:
        launch_job(cfg=cfg, init_method=args.init_method, func=test)

    # Perform model visualization.
    if cfg.TENSORBOARD.ENABLE and (
        cfg.TENSORBOARD.MODEL_VIS.ENABLE
        or cfg.TENSORBOARD.WRONG_PRED_VIS.ENABLE
    ):
        launch_job(cfg=cfg, init_method=args.init_method, func=visualize)


if __name__ == "__main__":
    main()
    # args = parse_args()
    # if args.num_shards > 1:
    #    args.output_dir = str(args.job_dir)

    # print("LOAD CONFIG in run_net.py")
    # cfg = load_config(args)

    # print("LOAD TRAIN TEST FUNC in run_net.py")
    # train, test = get_func(cfg)

    # print("START TRAINING in run_net.py")
    # # Perform training.
    # if cfg.TRAIN.ENABLE:
    #     # launch_job(cfg=cfg, init_method=args.init_method, func=train)
    #     xla.launch(
    #                 train,
    #                 args=(cfg,),
    #                 debug_single_process=1
    #             )
