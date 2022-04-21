
import torch
from torch import nn, optim
from torch.utils import data

import os
import numpy as np
import argparse

from trainer import UncertaintyTrainer
from module.config import checkOutputDirectoryAndCreate, loadConfig, dumpConfig, showConfig

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

def main(config, device):
    trainer = UncertaintyTrainer(config, device)
    trainer.fit()
    trainer.save(f'result/{config["output_folder"]}/flow_last.pt')

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description="Uncertainty trainer")
    parser.add_argument("--config", type=str)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--output_folder", type=str, default="")
    args = parser.parse_args()

    # config
    config = loadConfig(args.config)
    if (args.output_folder != ""):
        config["output_folder"] = args.output_folder

    # save config
    checkOutputDirectoryAndCreate(config["output_folder"])
    dumpConfig(config)
    showConfig(config)

    # wandb
    os.environ["WANDB_WATCH"] = "false"
    if (wandb != None):
        wandb.init(project="UncertaintyFlow", entity="andy-su", name=config["output_folder"])
        wandb.config.update(config)
        wandb.define_metric("loss", summary="min")

    if (args.gpu != ""):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "-1" else "cpu")

    # main
    main(config, device)
    