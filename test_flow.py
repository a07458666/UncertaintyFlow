
# import dnnlib
from torch import nn, optim
import torch
import numpy as np
from torch.utils import data
from module.flow import cnf
from math import log, pi
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision.datasets as dset
import argparse
from module.dataset.loader import load_my_1d, load_wiggle
try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

def standard_normal_logprob(z):
    dim = z.size(-1)
    log_z = -0.5 * dim * log(2 * pi)
    return log_z - z.pow(2) / 2

def get_trnsform():
    transform = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
    ])
    return transform

class MyDataset(Dataset):
    def __init__(self, latents, attributes, transform=None):
        self.latents = latents
        self.attributes = attributes
        self.transform = transform

    def __getitem__(self, index):
        x = self.latents[index]
        y = self.attributes[index]


        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

    def __len__(self):
        return len(self.latents)

def visualize_uncertainty(gt_x, gt_y, xdata, mean, var):
    # Visualize the result
    dyfit = 2 * np.sqrt(var)
    # dyfit = np.asarray(var)
    plt.plot(gt_x, gt_y, 'ok')
    # plt.plot(gt_x, gt_y, '-', color='k')
    # plt.plot(xdata, mean, 'og')
    plt.plot(xdata, mean, '-', color='g')
    # print("xdata", xdata)
    # print("ydata", ydata)
    # print("mean", mean)
    # print("var", var)
    # print("dyfit", dyfit)
    # print("var shape", len(var))
    # print("dyfit shape", len(dyfit))
    # print("var type", type(var))
    # print("dyfit type", type(dyfit))
    plt.fill_between(xdata, mean - dyfit, mean + dyfit, color='g', alpha=0.2)
    # plt.xlim(xdata.min(), xdata.max())
    plt.savefig(f"var_no_sqrt.png")
    if (wandb != None):
        # wandb.Image(plt)
        wandb.log({"var": plt})

def sortData(x, y):
    x_sorted, y_sorted = zip(*sorted(zip(x, y)))
    return np.asarray(x_sorted), np.asarray(y_sorted)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Uncertainty trainer")

    parser.add_argument("--output_foloder", type=str, default="my_1d")

    parser.add_argument("--flow_modules", type=str, default='32-32-32-32-32')
    parser.add_argument("--cond_size", type=int, default=1)
    parser.add_argument("--inputDim", type=int, default=1)
    parser.add_argument("--sample_size", type=int, default=100)
    parser.add_argument("--model_path", type=str, default="")

    args = parser.parse_args()

    # wandb=====
    os.environ["WANDB_WATCH"] = "false"
    if (wandb != None):
        wandb.init(project="UncertaintyFlow", entity="andy-su", name=args.output_foloder)
        wandb.config.update(args)
    #===========

    torch.manual_seed(0)

    prior = cnf(args.inputDim, args.flow_modules, args.cond_size, 1)
    prior.load_state_dict(torch.load(args.model_path))
    prior.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_test, y_test = load_my_1d("./dataset")
    X_train, y_train = load_wiggle()
    gt_X, gt_y = sortData(X_train, y_train)
    X_test, y_test = sortData(X_test, y_test)
    trainset = MyDataset(torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device), transform=None)
    valset = MyDataset(torch.Tensor(X_test).to(device), torch.Tensor(y_test).to(device), transform=None)

    X_eval = np.linspace(-3, 3, 300).reshape(-1, 1)
    y_eval = np.linspace(0, 0, 300).reshape(-1, 1)

    evalset = MyDataset(torch.Tensor(X_eval).to(device), torch.Tensor(y_eval).to(device), transform=None)
    print("shape = ", X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_eval.shape, y_eval.shape)

    # my_dataset = MyDataset(latents=torch.Tensor(sg_latents).cuda(), attributes=torch.tensor(sg_attributes).float().cuda())
    # train_loader = data.DataLoader(my_dataset, shuffle=False, batch_size=args.batch)
    # train_loader = data.DataLoader(trainset, shuffle=False, batch_size=args.batch, drop_last = True)
    val_loader = data.DataLoader(valset, shuffle=False, batch_size=1, drop_last = True)

   
    loss_fn = nn.MSELoss()

    mse_list = []
    mean_list = []
    var_list = []
    x_list = []
    # y_list = []
    for i, x in tqdm(enumerate(evalset)):
        input_x = torch.normal(mean = 0.0, std = 1.0, size=(args.sample_size ,1)).unsqueeze(1).to(device)
        condition_y = x[0].expand(args.sample_size, args.inputDim).unsqueeze(1) 
        delta_p = torch.zeros(args.sample_size, args.inputDim, 1).to(x[0])
        # print("input_x = ", input_x)
        # print("condition_y= ", condition_y[0])

        approx21, delta_log_p2 = prior(input_x, condition_y, delta_p, reverse=True)
        # print("x[1] output want = ", x[1])
        # print("approx21 mean = ", torch.mean(approx21))
        # print("approx21  var = ", torch.var(approx21))
        # print("approx21 size = ", approx21.size())
        mseLoss = loss_fn(x[0].expand(args.sample_size, args.inputDim).unsqueeze(1), approx21)
        np_mse = mseLoss.item()
        np_x = float(x[0].detach().cpu().numpy()[0])
        # np_y = float(x[1].detach().cpu().numpy()[0])
        np_var = float(torch.var(approx21).detach().cpu().numpy())
        np_mean = float(torch.mean(approx21).detach().cpu().numpy())
        mse_list.append(np_mse)
        x_list.append(np_x)
        # y_list.append(np_y)
        var_list.append(np_var)
        mean_list.append(np_mean)
        # if (i > 50):
        #     break
    
    mseMean = sum(mse_list) / len(mse_list)
    print("mseMean ", mseMean)
    print("mse_list", mse_list)
    if (wandb != None):
        logMsg = {}
        logMsg["mseMean"] = mseMean
        wandb.log(logMsg)
        wandb.watch(prior,log = "all", log_graph=True)  
    # print("mseLoss = ", np.mean(mseLoss))
    # print(x_list)
    # print(y_list)
    # x_list = sum(x_list, [])
    # y_list = sum(y_list, [])
    # x_list = [0,1,2,3,4,5,6,7,8,9,10,11]
    # y_list = [0,1,2,3,4,5,6,7,8,9,10,11]
    # print(x_list)
    # print(y_list)
    visualize_uncertainty(gt_X.reshape(-1), gt_y.reshape(-1), x_list, mean_list, var_list)

