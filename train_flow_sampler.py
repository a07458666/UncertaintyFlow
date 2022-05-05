
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
from module.condition_sampler.flow_sampler import FlowSampler

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

def pre_encoder(path):
    return path

def sample_condition(train_loader, encoder, save_dir):
    x = []
    y = []
    for data in tqdm(train_loader):
        condition_X_feature = encoder.forward_flatten(data[0])
        x.append(condition_X_feature)
        y.append(data[1])

    x = torch.cat(x, dim=0).detach().cpu().numpy()
    y = torch.cat(y, dim=0)
    file_path = os.path.join(save_dir, 'dataset.npz')
    np.savez(file_path, x=x, y=y, mean = x.mean(), var = x.var())
    return x

def main(config, device):
    trainer = UncertaintyTrainer(config, device)
    # condition_sampler
    if ("condition_sampler" in config.keys() and config["condition_sampler"]):
        encoder = MyResNet(in_channels = trainer.input_channels, out_features = trainer.N_classes)
        encoder.load_state_dict(torch.load(config["pre_train_model_path"]))
        encoder.eval()
        save_dir = f'result/{config["output_folder"]}'
        x = sample_condition(trainer.train_loader, encoder, save_dir)
        x_sampler = FlowSampler((trainer.cond_size, 1), '128-128', 1)
        loss = x_sampler.fit(x, batch=trainer.batch_size, lr=5e-3, epoch=50, save_dir=save_dir, save_model=True)
        import matplotlib.pyplot as plt
        # draw loss
        print('draw loss')
        plt.plot([i for i in range(len(loss))], loss)
        plt.savefig(os.path.join(save_dir, 'loss.png'))
        plt.clf()

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
    