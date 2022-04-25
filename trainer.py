import os
import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils import data
from scipy.stats import norm
from torch.optim.lr_scheduler import CosineAnnealingLR
import math

from module.flow import cnf
from module.utils import standard_normal_logprob, position_encode, addUniform, sortData
from module.dun_datasets.loader import loadDataset, MyDataset
from module.visualize import visualize_uncertainty
from module.resnet import MyResNet
from condition_sampler.flow_sampler import FlowSampler

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


class UncertaintyTrainer:
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        torch.manual_seed(self.config["time_seed"])
        self.batch_size = self.config["batch"]
        self.cond_size = self.config["cond_size"]

        if (self.config["image_task"]):
            self.loadImageDataset()
        else:
            self.loadDataset()
        self.creatModel()
        self.defOptimizer()
        self.model_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config["epochs"])
        return

    def loadDataset(self) -> None:
        X_train, y_train = loadDataset(self.config["dataset"])
        if self.config["condition_scale"] != 1:
            X_train = X_train * self.config["condition_scale"] 

        if (self.config["add_uniform"]):
            self.X_mean = X_train.mean()
            self.y_mean = y_train.mean()
            self.X_var = X_train.var()
            self.y_var = y_train.var()
            self.uniform_count = int(self.config["batch"] * self.config["uniform_rate"])
            print("X mean :", self.X_mean, "y mean :", self.y_mean,"X var :", self.X_var,"y var :", self.y_var)

        if self.config["position_encode"]:
            X_train = position_encode(X_train, self.config["position_encode_m"])
            self.cond_size += (self.config["position_encode_m"] * 2)
        print("shape = ", X_train.shape, y_train.shape)
        self.X_train = X_train
        self.y_train = y_train
        trainset = MyDataset(torch.Tensor(X_train).to(self.device), torch.Tensor(y_train).to(self.device), transform=None)
        self.train_loader = data.DataLoader(trainset, shuffle=True, batch_size=self.batch_size, drop_last = True)
        return

    def loadImageDataset(self) -> None:
        from module.dun_datasets.image_loaders import get_image_loader
        _, train_loader, _, _, N_classes, _ = get_image_loader(self.config["dataset"], batch_size=self.batch_size, cuda=True, workers=self.config["workers"], distributed=False)

        # X_train, y_train = loadDataset(self.config["dataset"])
        # if self.config["condition_scale"] != 1:
        #     X_train = X_train * self.config["condition_scale"] 

        # if (self.config["add_uniform"]):
        #     self.X_mean = X_train.mean()
        #     self.y_mean = y_train.mean()
        #     self.X_var = X_train.var()
        #     self.y_var = y_train.var()
        #     self.uniform_count = int(self.config["batch"] * self.config["uniform_rate"])
        #     print("X mean :", self.X_mean, "y mean :", self.y_mean,"X var :", self.X_var,"y var :", self.y_var)

        # if self.config["position_encode"]:
        #     X_train = position_encode(X_train, self.config["position_encode_m"])
        #     self.cond_size += (self.config["position_encode_m"] * 2)
        # print("shape = ", X_train.shape, y_train.shape)
        # self.X_train = X_train
        # self.y_train = y_train
        # trainset = MyDataset(torch.Tensor(X_train).to(self.device), torch.Tensor(y_train).to(self.device), transform=None)
        self.train_loader = train_loader
        self.N_classes = N_classes
        return

    def creatModel(self) -> None:
        if self.config["linear_encode"]:
            self.prior = cnf(self.config["inputDim"], self.config["flow_modules"], self.cond_size, 1, self.config["linear_encode_m"])
        else:
            self.prior = cnf(self.config["inputDim"], self.config["flow_modules"], self.cond_size, 1)
        if self.config["image_task"]:
            self.encoder = MyResNet()
        return

    def defOptimizer(self) -> None:
        self.optimizer = optim.Adam(self.prior.parameters(), lr=self.config["lr"])
        if self.config["image_task"]:
            self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.config["lr"])
        return

    def fit(self) -> list:
        self.prior.train()
        if self.config["image_task"]:
            self.encoder.train()
        loss_list = []

        with tqdm(range(self.config["epochs"])) as pbar:
            for epoch in pbar:
                for i, x in enumerate(self.train_loader):
                    if (self.config["image_task"]):
                        input_y_one_hot = torch.nn.functional.one_hot(x[1], self.N_classes)
                        input_y_one_hot = input_y_one_hot.type(torch.cuda.FloatTensor)
                        input_y = input_y_one_hot.unsqueeze(1).to(self.device)
                        
                        condition_X_feature = self.encoder(x[0])
                        condition_X = condition_X_feature.unsqueeze(2).to(self.device)
                    else:
                        input_y = x[1].unsqueeze(1)
                        condition_X = x[0].unsqueeze(1)

                    # print("input_y : ", input_y.size())
                    # print("condition_X : ", condition_X.size())
                    if (self.config["add_uniform"]):
                        if (self.config["uniform_scheduler"]):
                            self.uniform_count = int(self.config["batch"] * self.config["uniform_rate"] * math.cos((math.pi / 2) * (epoch / self.config["epochs"])))
                        input_y, condition_X = addUniform(input_y, condition_X, self.uniform_count, self.X_mean, self.y_mean, self.X_var, self.y_var, self.config)

                    delta_p = torch.zeros(input_y.shape[0], input_y.shape[1], 1).to(input_y)
                    # print("delta_p: ", delta_p.size())
                    approx21, delta_log_p2 = self.prior(input_y, condition_X, delta_p)

                    approx2 = standard_normal_logprob(approx21).view(input_y.size()[0], -1).sum(1, keepdim=True)
                
                    delta_log_p2 = delta_log_p2.view(input_y.size()[0], input_y.shape[1], 1).sum(1)
                    log_p2 = (approx2 - delta_log_p2)

                    loss = -log_p2.mean()

                    self.optimizer.zero_grad()
                    if self.config["image_task"]:
                        self.encoder_optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.config["image_task"]:
                        self.encoder_optimizer.step()

                    pbar.set_description(
                        f'logP: {loss:.5f}')
                    loss_list.append(loss.item())

                    if (wandb != None):
                        logMsg = {}
                        logMsg["epoch"] = epoch
                        logMsg["loss"] = loss
                        wandb.log(logMsg)
                        wandb.watch(self.prior,log = "all", log_graph=True)
                
                # epoch
                if (self.config["lr_scheduler"] == "cos"):
                    self.model_scheduler.step()
                self.save(f'result/{self.config["output_folder"]}/flow_{str(epoch).zfill(2)}.pt')
                if self.config["image_task"]:
                    self.save_encoder(f'result/{self.config["output_folder"]}/encoder_{str(epoch).zfill(2)}.pt')
        return loss_list


    def save(self, path) -> None:
        torch.save(self.prior.state_dict(), path)

    def load(self, path) -> None:
        self.prior.load_state_dict(torch.load(path))
        # self.prior.load_state_dict(torch.load(path, map_location=self.gpu))

    def save_encoder(self, path) -> None:
        torch.save(self.encoder.state_dict(), path)

    def load_encoder(self, path) -> None:
        self.encoder.load_state_dict(torch.load(path))
        # self.prior.load_state_dict(torch.load(path, map_location=self.gpu))


    def loadValDataset(self):
        show_range = 5 * self.config["condition_scale"]
        torch.manual_seed(0)
        var_scale = self.config["var_scale"]
        gt_X, gt_y = loadDataset(self.config["dataset"])
        gt_X, gt_y = sortData(gt_X, gt_y)
        self.config["eval_data"]["count"] *= self.config["condition_scale"]
        X_eval = np.linspace(-show_range, show_range, self.config["eval_data"]["count"]).reshape(-1, 1)
        y_eval = np.linspace(0, 0, self.config["eval_data"]["count"]).reshape(-1, 1)
        if self.config["position_encode"]:
            X_eval = position_encode(X_eval, self.config["position_encode_m"])
        if self.config["condition_scale"] != 1:
            X_eval = X_eval * self.config["condition_scale"]
            gt_X = gt_X * self.config["condition_scale"]
        self.gt_X = gt_X
        self.gt_y = gt_y
        self.evalset = MyDataset(torch.Tensor(X_eval).to(self.device), torch.Tensor(y_eval).to(self.device), transform=None)

    def loadValImageDataset(self) -> None:
        from module.dun_datasets.image_loaders import get_image_loader
        _, _, val_loader, _, N_classes, _ = get_image_loader(self.config["dataset"], batch_size=self.batch_size, cuda=True, workers=self.config["workers"], distributed=False)

        self.val_loader = val_loader
        self.N_classes = N_classes
        return

    def sample(self) -> None:
        self.prior.eval()

        mean_list = []
        var_list = []
        x_list = []

        for i, x in tqdm(enumerate(self.evalset)):
            input_x = torch.normal(mean = 0.0, std = 1.0, size=(self.config["sample_count"] ,1)).unsqueeze(1).to(self.device)
            condition_y = x[0].expand(self.config["sample_count"], -1).unsqueeze(1) 
            delta_p = torch.zeros(self.config["sample_count"], self.config["inputDim"], 1).to(x[0])

            approx21, delta_log_p2 = self.prior(input_x, condition_y, delta_p, reverse=True)

            np_x = float(x[0].detach().cpu().numpy()[0])
            np_var = float(torch.var(approx21).detach().cpu().numpy())
            np_mean = float(torch.mean(approx21).detach().cpu().numpy())
            x_list.append(np_x)
            var_list.append(np_var)
            mean_list.append(np_mean)
        
        savePath = f'result/{self.config["output_folder"]}/var.png'
        visualize_uncertainty(savePath, self.gt_X.reshape(-1), self.gt_y.reshape(-1), x_list, mean_list, var_list)
        return 
    

    def sampleImage(self) -> None:
        self.prior.eval()
        self.encoder.eval()

        # mean_list = []
        # var_list = []
        # x_list = []

        # for i, x in tqdm(enumerate(self.evalset)):
        #     input_x = torch.normal(mean = 0.0, std = 1.0, size=(self.config["sample_count"] ,1)).unsqueeze(1).to(self.device)
        #     condition_y = x[0].expand(self.config["sample_count"], -1).unsqueeze(1) 
        #     delta_p = torch.zeros(self.config["sample_count"], self.config["inputDim"], 1).to(x[0])

        #     approx21, delta_log_p2 = self.prior(input_x, condition_y, delta_p, reverse=True)

        #     np_x = float(x[0].detach().cpu().numpy()[0])
        #     np_var = float(torch.var(approx21).detach().cpu().numpy())
        #     np_mean = float(torch.mean(approx21).detach().cpu().numpy())
        #     x_list.append(np_x)
        #     var_list.append(np_var)
        #     mean_list.append(np_mean)
        
        # savePath = f'result/{self.config["output_folder"]}/var.png'
        # visualize_uncertainty(savePath, self.gt_X.reshape(-1), self.gt_y.reshape(-1), x_list, mean_list, var_list)
        return 