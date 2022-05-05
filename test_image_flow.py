import os
import numpy as np
import argparse

import pandas
import torch

from trainer import UncertaintyTrainer
from module.config import loadConfig, showConfig

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

def main(config, device, model_path, encoder_path):
    trainer = UncertaintyTrainer(config, device)
    csv_path = "./" + config["output_folder"] + ".csv"
    trainer.setDataFrame(csv_path)
    trainer.load(model_path)
    trainer.load_encoder(encoder_path)
    trainer.loadValImageDataset()
    # trainer.sampleImageAcc()
    
    corruptions = {
    "CIFAR10": [(0, cor) for cor in range(1, 6)],
    # "CIFAR100": [(0, cor) for cor in range(1, 6)],
    "MNIST": [],
    # "Fashion": [],
    # "SVHN": []
    }

    rotations = {
        "CIFAR10": [],
        # "CIFAR100": [],
        "MNIST": [(rot, 0) for rot in range(15, 181, 15)],
        # "Fashion": [],
        # "SVHN": []
    }

    target_datasets = {
    "MNIST": ["Fashion"],
    # "Fashion": ["MNIST", "KMNIST"],
    "CIFAR10": ["SVHN"],
    # "CIFAR100": ["SVHN"],
    # "SVHN": ["CIFAR10"]
    }
    for i in range(10):
        err_list, ll_list = trainer.rot_measurements(rotations, corruptions)
        err_props = trainer.rejection_measurements(target_datasets)
    

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description="Uncertainty trainer")
    parser.add_argument("--modelPath", type=str)
    parser.add_argument("--encoderPath", type=str)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    # config
    if (args.config == ""):
        args.config = os.path.dirname(args.modelPath) + "/config.yaml"
    config = loadConfig(args.config)
    showConfig(config)

    if (args.gpu != ""):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "-1" else "cpu")
    # main
    main(config, device, args.modelPath, args.encoderPath)
    