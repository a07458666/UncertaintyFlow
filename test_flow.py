
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
from module.utils import standard_normal_logprob, position_encode
from module.dun_datasets.loader import loadDataset, MyDataset
from module.config import loadConfig, showConfig
from module.visualize import visualize_uncertainty

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


def sortData(x, y):
    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    return np.asarray(x_sorted), np.asarray(y_sorted)

def main(config, device, model_path):
    show_range = 5 * config["condition_scale"]
    torch.manual_seed(0)
    var_scale = config["var_scale"]
    gt_X, gt_y = loadDataset(config["dataset"])
    gt_X, gt_y = sortData(gt_X, gt_y)
    
    # X_eval = np.linspace(gt_X.mean() - (var_scale * gt_X.var()), gt_X.mean() + (var_scale * gt_X.var()), config["eval_data"]["count"]).reshape(-1, 1)
    config["eval_data"]["count"] *= config["condition_scale"]
    X_eval = np.linspace(-show_range, show_range, config["eval_data"]["count"]).reshape(-1, 1)
    y_eval = np.linspace(0, 0, config["eval_data"]["count"]).reshape(-1, 1)
    
    if config["position_encode"]:
        X_eval = position_encode(X_eval, config["position_encode_m"])

    if config["condition_scale"] != 1:
        X_eval = X_eval * config["condition_scale"]
        gt_X = gt_X * config["condition_scale"]

    cond_size = config["cond_size"]
    if config["position_encode"]:
        cond_size += (config["position_encode_m"] * 2)

    if config["linear_encode"]:
        prior = cnf(config["inputDim"], config["flow_modules"], cond_size, 1, config["linear_encode_m"])
    else:
        prior = cnf(config["inputDim"], config["flow_modules"], cond_size, 1)
    prior.load_state_dict(torch.load(model_path))
    prior.eval()
    

    evalset = MyDataset(torch.Tensor(X_eval).to(device), torch.Tensor(y_eval).to(device), transform=None)
    print("shape (gtX, gtY, evalX, eval Y) = ", gt_X.shape, gt_y.shape, X_eval.shape, y_eval.shape)
    
    loss_fn = nn.MSELoss()

    mean_list = []
    var_list = []
    x_list = []

    for i, x in tqdm(enumerate(evalset)):
        input_x = torch.normal(mean = 0.0, std = 1.0, size=(config["sample_count"] ,1)).unsqueeze(1).to(device)
        condition_y = x[0].expand(config["sample_count"], -1).unsqueeze(1) 
        delta_p = torch.zeros(config["sample_count"], config["inputDim"], 1).to(x[0])

        approx21, delta_log_p2 = prior(input_x, condition_y, delta_p, reverse=True)
        np_x = float(x[0].detach().cpu().numpy()[0])
        np_var = float(torch.var(approx21).detach().cpu().numpy())
        np_mean = float(torch.mean(approx21).detach().cpu().numpy())
        x_list.append(np_x)
        var_list.append(np_var)
        mean_list.append(np_mean)
    
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
    showConfig(config)

    if (args.gpu != ""):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "-1" else "cpu")
    # main
    main(config, device, args.modelPath)
    