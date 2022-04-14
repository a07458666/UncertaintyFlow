
import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.data import Dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms

import os
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
from math import log, pi
from tqdm import tqdm

from module.flow import cnf
from module.dataset.loader import loadDataset, MyDataset
from module.config import loadConfig
from module.visualize import visualize_uncertainty

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

def sortData(x, y):
    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    return np.asarray(x_sorted), np.asarray(y_sorted)

def position_encoding(x, m):
    x_p_list = [x]
    for i in range(m):
        x_p_list.append(np.sin((2**(i+1)) * x))
        x_p_list.append(np.cos((2**(i+1)) * x))
    x = np.concatenate(x_p_list, axis=1)
    return x

def main(config, device, model_path):
    torch.manual_seed(0)

    gt_X, gt_y = loadDataset(config["dataset"])
    gt_X, gt_y = sortData(gt_X, gt_y)

    X_eval = np.linspace(config["eval_data"]["min"], config["eval_data"]["max"], config["eval_data"]["count"]).reshape(-1, 1)
    y_eval = np.linspace(0, 0, config["eval_data"]["count"]).reshape(-1, 1)
    
    if config["position_encoding"]:
        X_eval = position_encoding(X_eval, config["position_encoding_m"])

    if config["condition_scale"] != 1:
        X_eval = X_eval * config["condition_scale"]
        gt_X = gt_X * config["condition_scale"]

    cond_size = config["cond_size"]
    if config["position_encoding"]:
        cond_size += (config["position_encoding_m"] * 2)

    prior = cnf(config["inputDim"], config["flow_modules"], cond_size, 1)
    prior.load_state_dict(torch.load(model_path))
    prior.eval()
    

    evalset = MyDataset(torch.Tensor(X_eval).to(device), torch.Tensor(y_eval).to(device), transform=None)
    print("shape (gtX, gtY, evalX, eval Y) = ", gt_X.shape, gt_y.shape, X_eval.shape, y_eval.shape)
    
    loss_fn = nn.MSELoss()

    mse_list = []
    mean_list = []
    var_list = []
    x_list = []

    for i, x in tqdm(enumerate(evalset)):
        input_x = torch.normal(mean = 0.0, std = 1.0, size=(config["sample_count"] ,1)).unsqueeze(1).to(device)
        condition_y = x[0].expand(config["sample_count"], -1).unsqueeze(1) 
        delta_p = torch.zeros(config["sample_count"], config["inputDim"], 1).to(x[0])

        approx21, delta_log_p2 = prior(input_x, condition_y, delta_p, reverse=True)
        # mseLoss = loss_fn(x[0].expand(config["sample_count"], config["inputDim"]).unsqueeze(1), approx21)
        # np_mse = mseLoss.item()
        np_x = float(x[0].detach().cpu().numpy()[0])
        np_var = float(torch.var(approx21).detach().cpu().numpy())
        np_mean = float(torch.mean(approx21).detach().cpu().numpy())
        # mse_list.append(np_mse)
        x_list.append(np_x)
        var_list.append(np_var)
        mean_list.append(np_mean)
        # if (i > 50):
        #     break
    
    # mseMean = sum(mse_list) / len(mse_list)
    # print("mseMean :", mseMean)
    savePath = os.path.splitext(args.modelPath)[0] + "_var.png"
    visualize_uncertainty(savePath, gt_X.reshape(-1), gt_y.reshape(-1), x_list, mean_list, var_list)

if __name__ == '__main__':
    # args
    parser = argparse.ArgumentParser(description="Uncertainty trainer")
    parser.add_argument("--modelPath", type=str)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--config", type=str, default="")
    args = parser.parse_args()

    # config
    if (args.config == ""):
        args.config = os.path.dirname(args.modelPath) + "/config.yaml"
    config = loadConfig(args.config)
    print("config : ", config)

    if (args.gpu != ""):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "-1" else "cpu")
    # main
    main(config, device, args.modelPath)
    