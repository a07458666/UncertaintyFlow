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
    acc_all_list = []
    with torch.no_grad():
        trainer = UncertaintyTrainer(config, device)
        csv_path = "./" + config["output_folder"] + ".csv"
        trainer.setDataFrame(csv_path)
        # trainer.loadValImageDataset()
        trainer.loadNoiseDataset()
        for epoch in range(config["epochs"]):
            model_path = f'result/{config["output_folder"]}/flow_{str(epoch).zfill(2)}.pt'
            encoder_path = f'result/{config["output_folder"]}/encoder_{str(epoch).zfill(2)}.pt'
            trainer.load(model_path)
            trainer.load_encoder(encoder_path)
            test_acc = trainer.sampleImageAcc(MC_sample = 10, mean = 0, std = 0.2)
            acc_all_list.extend([test_acc])
    npy_path = "./log/" + config["output_folder"] + ".npy"
    np.save(npy_path, np.array(acc_all_list))

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description="Uncertainty trainer")
    parser.add_argument("--modelPath", type=str)
    parser.add_argument("--encoderPath", type=str)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    if (args.gpu != ""):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # config
    if (args.config == ""):
        args.config = os.path.dirname(args.modelPath) + "/config.yaml"
    config = loadConfig(args.config)
    showConfig(config)

    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "-1" else "cpu")
    # main
    main(config, device, args.modelPath, args.encoderPath)
    