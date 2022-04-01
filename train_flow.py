
# import dnnlib
from torch import nn, optim
import torch
import numpy as np
from torch.utils import data
from module.flow import cnf
from math import log, pi
import os
from tqdm import tqdm

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Uncertainty trainer")

    parser.add_argument("--output_foloder", type=str, default="my_1d")
    parser.add_argument("--latent_path",default='data_numpy/latents.npy', type=str, help="path to the latents")
    parser.add_argument("--light_path",default='data_numpy/lighting.npy', type=str, help="path to the lighting parameters")
    parser.add_argument("--attributes_path",default='data_numpy/attributes.npy', type=str, help="path to the attribute parameters")
    parser.add_argument(
        "--batch", type=int, default=5, help="batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of epochs"
    )

    parser.add_argument("--flow_modules", type=str, default='32-32-32-32-32')
    parser.add_argument("--cond_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--inputDim", type=int, default=1)

    args = parser.parse_args()

    # wandb=====
    os.environ["WANDB_WATCH"] = "false"
    if (wandb != None):
        wandb.init(project="UncertaintyFlow", entity="andy-su", name=args.output_foloder)
        wandb.config.update(args)
        wandb.define_metric("loss", summary="min")
    #===========

    torch.manual_seed(0)

    prior = cnf(args.inputDim, args.flow_modules, args.cond_size, 1)

    # sg_latents = np.load(args.latent_path)
    # lighting = np.load(args.light_path)
    # attributes = np.load(args.attributes_path)
    # sg_attributes = np.concatenate([lighting,attributes], axis = 1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X_train, y_train, X_test, y_test = load_my_1d("./dataset")
    # X_train, y_train = load_wiggle()
    trainset = MyDataset(torch.Tensor(X_train).to(device), torch.Tensor(y_train).to(device), transform=None)
    # valset = MyDataset(torch.Tensor(X_test).to(device), torch.Tensor(y_test).to(device), transform=None)
    print("shape = ", X_train.shape, y_train.shape)

    # my_dataset = MyDataset(latents=torch.Tensor(sg_latents).cuda(), attributes=torch.tensor(sg_attributes).float().cuda())
    # train_loader = data.DataLoader(my_dataset, shuffle=False, batch_size=args.batch)
    train_loader = data.DataLoader(trainset, shuffle=False, batch_size=args.batch, drop_last = True)
    # val_loader = data.DataLoader(valset, shuffle=False, batch_size=args.batch, drop_last = True)

    optimizer = optim.Adam(prior.parameters(), lr=args.lr)

    with tqdm(range(args.epochs)) as pbar:
        for epoch in pbar:
            for i, x in enumerate(train_loader):
                input_x = x[1].unsqueeze(1)
                condition_y = x[0].unsqueeze(1)
                delta_p = torch.zeros(x[1].size()[0], args.inputDim, 1).to(x[1])

                approx21, delta_log_p2 = prior(input_x, condition_y, delta_p)

                approx2 = standard_normal_logprob(approx21).view(x[1].size()[0], -1).sum(1, keepdim=True)
              
                delta_log_p2 = delta_log_p2.view(x[1].size()[0], args.inputDim, 1).sum(1)
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

                if i % 1000 == 0:
                    torch.save(
                        prior.state_dict(), f'trained_model/{args.output_foloder}_{str(i).zfill(6)}_{str(epoch).zfill(2)}.pt'
                    )

