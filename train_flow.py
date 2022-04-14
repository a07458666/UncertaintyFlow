
import torch
from torch import nn, optim
from torch.utils import data

import torchvision.datasets as dset
import torchvision.transforms as transforms

import os
import numpy as np
import random
import argparse
from math import log, pi
from tqdm import tqdm

from module.flow import cnf
from module.dataset.loader import loadDataset, MyDataset
from module.config import checkOutputDirectoryAndCreate, loadConfig, dumpConfig

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2

def position_encoding(x, m):
    x_p_list = [x]
    for i in range(m):
        x_p_list.append(np.sin((2**(i+1)) * x))
        x_p_list.append(np.cos((2**(i+1)) * x))
    x = np.concatenate(x_p_list, axis=1)
    return x

def main(config, device):
    torch.manual_seed(0)

    X_train, y_train = loadDataset(config["dataset"])
    if (config["add_uniform"]):
        print("add_uniform data")
        print("uniform count : ", int(X_train.shape[0] * config["uniform_rate"]))
        print("uniform X rane : ", X_train.max(), X_train.min())
        print("uniform y rane : ", y_train.max(), y_train.min())
        print("X_train.shape", X_train.shape)
        print("X_train[:10]", X_train[:10])
        print("y_train.shape", y_train.shape)
        print("y_train[:10]", y_train[:10])
        uniform_count = int(X_train.shape[0] * config["uniform_rate"])
        x_max = X_train.max() * config["uniform_scale"]
        x_min = X_train.min() * config["uniform_scale"]
        y_max = y_train.max() * config["uniform_scale"]
        y_min = y_train.min() * config["uniform_scale"]
        X_uniform = np.random.uniform(x_min, x_max, uniform_count).reshape(-1, 1)
        y_uniform = np.random.uniform(y_min, y_max, uniform_count).reshape(-1, 1)
        X_train = np.concatenate([X_uniform, X_train])
        y_train = np.concatenate([y_uniform, y_train])

    if config["position_encoding"]:
        X_train = position_encoding(X_train, config["position_encoding_m"])
    
    if config["condition_scale"] != 1:
        X_train = X_train * config["condition_scale"]

    cond_size = config["cond_size"]
    if config["position_encoding"]:
        cond_size += (config["position_encoding_m"] * 2)
    prior = cnf(config["inputDim"], config["flow_modules"], cond_size, 1)

    print("shape = ", X_train.shape, y_train.shape)
    trainset = MyDataset(torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device), transform=None)
    train_loader = data.DataLoader(trainset, shuffle=True, batch_size=config["batch"], drop_last = True)
    optimizer = optim.Adam(prior.parameters(), lr=config["lr"])

    with tqdm(range(config["epochs"])) as pbar:
        for epoch in pbar:
            for i, x in enumerate(train_loader):
                input_x = x[1].unsqueeze(1)
                condition_y = x[0].unsqueeze(1)
                delta_p = torch.zeros(x[1].size()[0], config["inputDim"], 1).to(x[1])

                approx21, delta_log_p2 = prior(input_x, condition_y, delta_p)

                approx2 = standard_normal_logprob(approx21).view(x[1].size()[0], -1).sum(1, keepdim=True)
              
                delta_log_p2 = delta_log_p2.view(x[1].size()[0], config["inputDim"], 1).sum(1)
                log_p2 = (approx2 - delta_log_p2)

                loss = -log_p2.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_description(
                    f'logP: {loss:.5f}')

                if (wandb != None):
                    logMsg = {}
                    logMsg["epoch"] = epoch
                    logMsg["loss"] = loss
                    wandb.log(logMsg)
                    wandb.watch(prior,log = "all", log_graph=True)  

            torch.save(prior.state_dict(), f'result/{config["output_foloder"]}/flow_{str(epoch).zfill(2)}.pt')
        torch.save(prior.state_dict(), f'result/{config["output_foloder"]}/flow_last.pt')

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description="Uncertainty trainer")
    parser.add_argument("--config", type=str)
    parser.add_argument("--gpu", type=str, default="0")
    args = parser.parse_args()

    # config
    config = loadConfig(args.config)
    checkOutputDirectoryAndCreate(config["output_foloder"])
    dumpConfig(config)
    print("config : ", config)

    # wandb
    os.environ["WANDB_WATCH"] = "false"
    if (wandb != None):
        wandb.init(project="UncertaintyFlow", entity="andy-su", name=config["output_foloder"])
        wandb.config.update(config)
        wandb.define_metric("loss", summary="min")

    if (args.gpu != ""):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "-1" else "cpu")
    # main
    main(config, device)
    