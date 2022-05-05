
import torch
from torch import nn, optim
from torch.utils import data

import os
import numpy as np
import argparse

from trainer import UncertaintyTrainer
from module.config import checkOutputDirectoryAndCreate, loadConfig, dumpConfig, showConfig
from module.resnet import MyResNet
from tqdm import tqdm

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

def pre_encoder(path):
    return path

def sample_condition(train_loader, encoder):
    x = []
    for data in tqdm(train_loader):
        condition_X_feature = encoder(data[0])
        x.append(condition_X_feature)

    x = torch.cat(x, dim=0)
    return x.detach().cpu().numpy()

def main(config, device):
    trainer = UncertaintyTrainer(config, device)
    # condition_sampler
    if ("condition_sampler" in config.keys() and config["condition_sampler"]):
        print("condition_sampler")
        trainer.load_encoder(config["pre_train_model_path"])
        trainer.load_sampler(config["sampler_path"])
        trainer.loadSamplerDataset(config["sampler_dataset"])
        trainer.fit_sampler()
    else:
        trainer.fit()
    trainer.save(f'result/{config["output_folder"]}/flow_last.pt')
    trainer.save_encoder(f'result/{config["output_folder"]}/encoder_last.pt')

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
    