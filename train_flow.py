
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
from module.utils import standard_normal_logprob, position_encode
from module.dun_datasets.loader import loadDataset, MyDataset
from module.config import checkOutputDirectoryAndCreate, loadConfig, dumpConfig, showConfig

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

def addUniform(input_y, condition_X, uniform_count, X_mean, y_mean, X_var, y_var, config):
    var_scale = config["var_scale"]
    X_uniform = np.random.uniform(X_mean - (var_scale * X_var), X_mean + (var_scale * X_var), uniform_count).reshape(-1, 1, 1)
    y_uniform = np.random.uniform(y_mean - (var_scale * y_var), y_mean + (var_scale * y_var), uniform_count).reshape(-1, 1, 1)
    if config["position_encode"]:
        X_uniform = position_encode(X_uniform, config["position_encode_m"]).reshape(-1, 1, 1 + (config["position_encode_m"] * 2))
    X_uniform = torch.Tensor(X_uniform).to(device)
    y_uniform = torch.Tensor(y_uniform).to(device)

    # print("y size ", y_uniform.size())
    # print("x size ", X_uniform.size())
    # print("input_y size ", input_y.size())
    # print("condition_X size ", condition_X.size())
    condition_X = torch.cat((X_uniform, condition_X), 0)
    input_y = torch.cat((y_uniform, input_y), 0)
    return input_y, condition_X

def main(config, device):
    torch.manual_seed(config["time_seed"])
    batch_size = config["batch"]
    X_train, y_train = loadDataset(config["dataset"])
    if (config["add_uniform"]):
        X_mean = X_train.mean()
        y_mean = y_train.mean()
        X_var = X_train.var()
        y_var = y_train.var()
        uniform_count = int(config["batch"] * config["uniform_rate"])
        batch_size -= uniform_count
        print("X mean :", X_mean, "y mean :", y_mean,"X var :", X_var,"y var :", y_var)
        
    if config["position_encode"]:
        X_train = position_encode(X_train, config["position_encode_m"])
    
    if config["condition_scale"] != 1:
        X_train = X_train * config["condition_scale"]

    cond_size = config["cond_size"]
    if config["position_encode"]:
        cond_size += (config["position_encode_m"] * 2)
    prior = cnf(config["inputDim"], config["flow_modules"], cond_size, 1)

    print("shape = ", X_train.shape, y_train.shape)
    trainset = MyDataset(torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device), transform=None)
    train_loader = data.DataLoader(trainset, shuffle=True, batch_size=batch_size, drop_last = True)
    optimizer = optim.Adam(prior.parameters(), lr=config["lr"])

    with tqdm(range(config["epochs"])) as pbar:
        for epoch in pbar:
            for i, x in enumerate(train_loader):
                input_y = x[1].unsqueeze(1)
                condition_X = x[0].unsqueeze(1)
                if (config["add_uniform"]):
                    input_y, condition_X = addUniform(input_y, condition_X, uniform_count, X_mean, y_mean, X_var, y_var, config)

                delta_p = torch.zeros(input_y.size()[0], config["inputDim"], 1).to(input_y)

                approx21, delta_log_p2 = prior(input_y, condition_X, delta_p)

                approx2 = standard_normal_logprob(approx21).view(input_y.size()[0], -1).sum(1, keepdim=True)
              
                delta_log_p2 = delta_log_p2.view(input_y.size()[0], config["inputDim"], 1).sum(1)
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

            torch.save(prior.state_dict(), f'result/{config["output_folder"]}/flow_{str(epoch).zfill(2)}.pt')
        torch.save(prior.state_dict(), f'result/{config["output_folder"]}/flow_last.pt')

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
    