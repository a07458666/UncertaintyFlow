from distutils.command.config import config
import os

import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.utils import data
from scipy.stats import norm
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import torch.nn.functional as F
import pandas as pd


from module.flow import cnf
from module.utils import standard_normal_logprob, position_encode, addUniform, sortData, accuracy,set_parameter_requires_grad
from module.dun_datasets.loader import loadDataset, MyDataset
from module.visualize import visualize_uncertainty
from module.resnet import MyResNet
from module.image.OOD_utils import rotate_load_dataset, load_corrupted_dataset, cross_load_dataset
from module.dun_datasets.image_loaders import get_image_loader
from module.image.test_methods import class_brier, class_err, class_ll, class_ECE
from module.condition_sampler.flow_sampler import FlowSampler
from module.noise_datasets.noise_datasets import cifar_dataloader
from module.losses.loss_coteaching import loss_coteaching
from module.losses.vicreg import vicreg_loss_func

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
        self.eps = 1e-40
        # self.loadImageDataset()
        self.loadNoiseDataset()
        self.creatModel()
        self.defOptimizer()
        self.model_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config["epochs"])
        self.encoder_scheduler = CosineAnnealingLR(self.encoder_optimizer, T_max=self.config["epochs"])
        return

    def setDataFrame(self, path):
        dtypes = np.dtype([
            ("method", str), ("dataset", str), ("model", str), 
            # ("stop", int), ("number", int), ("n_samples", int), ("warmup", int),
            ("ll", float), ("err", float), ("ece", float), ("brier", float), ("rotation", int), ("corruption", int),
            ("auc_roc", float), ("err_props", list), ("target_dataset", str),
            # ("batch_time", float), ("batch_size", int),
            # ("best_or_last", str), ("use_no_train_post", bool)
        ])
        data = np.empty(0, dtype=dtypes)
        self.df = pd.DataFrame(data)  
        self.df_path = path
        method = "flow"
        model = "res18"
        self.row_to_add_proto = {"dataset": self.config["dataset"], "method": method, "model": model}

    def loadImageDataset(self) -> None:
        from module.dun_datasets.image_loaders import get_image_loader
        _, train_loader, _, input_channels, N_classes, _ = get_image_loader(self.config["dataset"], batch_size=self.batch_size, cuda=True, workers=self.config["workers"], distributed=False)
        self.train_loader = train_loader
        self.N_classes = N_classes
        self.input_channels = input_channels
        return

    def loadNoiseDataset(self) -> None:
        dataloaders = cifar_dataloader(cifar_type=self.config['dataset'], root="./dataset", batch_size=self.batch_size, 
                            num_workers=self.config["workers"], noise_type=self.config['noise_type'], percent=self.config['percent'])
        if (self.config["ssl"]):
            self.train_loader = dataloaders.run(mode='train')
        else:
            self.train_loader = dataloaders.run(mode='train_single')
        self.val_loader = dataloaders.run(mode='test')

        self.N_classes = 10
        self.input_channels = 3
        self.num_test_images = len(self.val_loader.dataset)
        return

    def creatModel(self) -> None:
        self.prior = cnf(self.config["inputDim"], self.config["flow_modules"], self.cond_size, 1).to(self.device)
        self.encoder = MyResNet(in_channels = self.input_channels, out_features = self.cond_size).to(self.device)
        if (self.config["fix_encoder"]):
            set_parameter_requires_grad(self.encoder)
        return

    def defOptimizer(self) -> None:
        self.optimizer = optim.Adam(self.prior.parameters(), lr=self.config["lr"])
        # self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.config["encoder_lr"])
        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=self.config["encoder_lr"], momentum=0.9, weight_decay=5e-4)
        return


    def train(self, epoch) -> list:
        self.prior.train()
        self.encoder.train()
        acclist = []
        loss_list = []

        pbar = tqdm(self.train_loader)
        for i, x in enumerate(pbar):
            image, target = x[0].to(self.device), x[1].to(self.device)
            # print("x[0] : ", x[0].size())
            # print("x[1] : ", x[1].size())
            # print("self.N_classes : ", self.N_classes)
            input_y_one_hot = torch.nn.functional.one_hot(target, self.N_classes)
            input_y_one_hot = input_y_one_hot.type(torch.cuda.FloatTensor)
            input_y = input_y_one_hot.unsqueeze(1).to(self.device)
            if (self.config["blur"]):
                input_y += torch.normal(mean=0, std=0.2, size=input_y.size()).to(self.device)
            condition_X_feature = self.encoder(image)
            condition_X = condition_X_feature.unsqueeze(2).to(self.device)
            # weight = self.getWeightByEntropy(input_y, condition_X)
            delta_p = torch.zeros(input_y.shape[0], input_y.shape[1], 1).to(input_y)
            # print("input_y : ", input_y.size())
            # print("condition_X : ", condition_X.size())
            # print("delta_p: ", delta_p.size())
            approx21, delta_log_p2 = self.prior(input_y, condition_X, delta_p)
            

            approx2 = standard_normal_logprob(approx21).view(input_y.size()[0], -1).sum(1, keepdim=True)
        
            delta_log_p2 = delta_log_p2.view(input_y.size()[0], input_y.shape[1], 1).sum(1)
            log_p2 = (approx2 - delta_log_p2)

            # loss = -(log_p2 * weight).mean()
            loss = -log_p2.mean()

            self.optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            if (not self.config["fix_encoder"]):
                condition_X.retain_grad()
            loss.backward()
            self.optimizer.step()
            self.encoder_optimizer.step()

            pbar.set_description(
                f'epoch: {epoch}, logP: {loss:.5f}')
            loss_list.append(loss.item())

            if (wandb != None):
                logMsg = {}
                logMsg["epoch"] = epoch
                logMsg["loss"] = loss
                logMsg["feature_var"] = condition_X.var(dim=1).mean().detach().cpu().item()
                if (not self.config["fix_encoder"]):
                    logMsg["feature_grad"] = condition_X.grad.mean().item()
                wandb.log(logMsg)
                wandb.watch(self.prior,log = "all", log_graph=True)
        
        # epoch
        self.model_scheduler.step()
        self.encoder_scheduler.step()
        self.save(f'result/{self.config["output_folder"]}/flow_{str(epoch).zfill(2)}.pt')
        if self.config["image_task"]:
            self.save_encoder(f'result/{self.config["output_folder"]}/encoder_{str(epoch).zfill(2)}.pt')

        return acclist

    def trainSSL(self, epoch) -> list:
        self.prior.train()
        self.encoder.train()
        acclist = []
        loss_list = []

        pbar = tqdm(self.train_loader)
        for i, x in enumerate(pbar):
            image, img1, img2, target = x[0].to(self.device), x[1].to(self.device), x[2].to(self.device), x[3].to(self.device), 
            # print("x[0] : ", x[0].size())
            # print("x[1] : ", x[1].size())
            # print("self.N_classes : ", self.N_classes)
            input_y_one_hot = torch.nn.functional.one_hot(target, self.N_classes)
            input_y_one_hot = input_y_one_hot.type(torch.cuda.FloatTensor)
            input_y = input_y_one_hot.unsqueeze(1).to(self.device)
            if (self.config["blur"]):
                y_noise_std = self.config["y_noise_std"] * math.cos((math.pi / 2) * (epoch / self.config["epochs"]))
                input_y += torch.normal(mean=0, std=y_noise_std, size=input_y.size()).to(self.device)
            
            condition_X_feature = self.encoder(image)
            _, _, z1, z2 = self.encoder.forward_ssl(img1, img2)
            condition_X = condition_X_feature.unsqueeze(2).to(self.device)
            # weight = self.getWeightByEntropy(input_y, condition_X)
            delta_p = torch.zeros(input_y.shape[0], input_y.shape[1], 1).to(input_y)
            # print("input_y : ", input_y.size())
            # print("condition_X : ", condition_X.size())
            # print("delta_p: ", delta_p.size())
            approx21, delta_log_p2 = self.prior(input_y, condition_X, delta_p)

            approx2 = standard_normal_logprob(approx21).view(input_y.size()[0], -1).sum(1, keepdim=True)
        
            delta_log_p2 = delta_log_p2.view(input_y.size()[0], input_y.shape[1], 1).sum(1)
            log_p2 = (approx2 - delta_log_p2)

            loss_vic, loss_vic_sim, loss_vic_var, loss_vic_cov = vicreg_loss_func(z1, z2) # loss
            # loss = -(log_p2 * weight).mean()
            std_cond = torch.sqrt(condition_X.var(dim=1) + 1e-4)
            cond_reg_loss = self.config["lambda_reg"] *  torch.mean(F.relu(1 - std_cond))
            loss_ll = -log_p2.mean()
            loss = loss_ll + loss_vic + cond_reg_loss

            self.optimizer.zero_grad()
            self.encoder_optimizer.zero_grad()
            condition_X.retain_grad()
            loss.backward()
            self.optimizer.step()
            self.encoder_optimizer.step()

            pbar.set_description(
                f'epoch: {epoch}, logP: {loss:.5f}')
            loss_list.append(loss.item())

            if (wandb != None):
                logMsg = {}
                logMsg["epoch"] = epoch
                logMsg["loss"] = loss
                logMsg["loss_ll"] = loss
                logMsg["loss_vic"] = loss_vic
                logMsg["loss_vic_sim"] = loss_vic_sim
                logMsg["loss_vic_var"] = loss_vic_var
                logMsg["loss_vic_cov"] = loss_vic_cov
                logMsg["cond_reg_loss"] = cond_reg_loss
                logMsg["feature_var"] = condition_X.var(dim=1).mean().detach().cpu().item()
                logMsg["feature_grad"] = condition_X.grad.mean().item()
                wandb.log(logMsg)
                wandb.watch(self.prior,log = "all", log_graph=True)
        
        # epoch
        self.model_scheduler.step()
        self.encoder_scheduler.step()
        self.save(f'result/{self.config["output_folder"]}/flow_{str(epoch).zfill(2)}.pt')
        if self.config["image_task"]:
            self.save_encoder(f'result/{self.config["output_folder"]}/encoder_{str(epoch).zfill(2)}.pt')

        return acclist


    def train_co_teaching(self, epoch) -> list:
        self.prior.train()
        self.encoder.train()
        acclist = []
        loss_list = []

        pbar = tqdm(self.train_loader)
        for i, x in enumerate(pbar):
            image, target = x[0].to(self.device), x[1].to(self.device)
            # print("image : ", image.size())
            # print("target : ", target.size())
            image = torch.cat((image, image), 0)
            target = torch.cat((target, target), 0)
            # print("af image : ", image.size())
            # print("af target : ", target.size())
            # print("self.N_classes : ", self.N_classes)
            input_y_one_hot = torch.nn.functional.one_hot(target, self.N_classes)
            input_y_one_hot = input_y_one_hot.type(torch.cuda.FloatTensor)
            input_y = input_y_one_hot.unsqueeze(1).to(self.device)
            
            condition_X_feature = self.encoder(image)
            condition_X = condition_X_feature.unsqueeze(2).to(self.device)



            if (self.config["add_uniform"]):
                if (self.config["uniform_scheduler"]):
                    self.uniform_count = int(self.config["batch"] * self.config["uniform_rate"] * math.cos((math.pi / 2) * (epoch / self.config["epochs"])))
                input_y, condition_X = addUniform(input_y, condition_X, self.uniform_count, self.X_mean, self.y_mean, self.X_var, self.y_var, self.config)

            delta_p = torch.zeros(input_y.shape[0], input_y.shape[1], 1).to(input_y)
            # print("input_y : ", input_y.size())
            # print("condition_X : ", condition_X.size())
            # print("delta_p: ", delta_p.size())
            approx21, delta_log_p2 = self.prior(input_y, condition_X, delta_p)

            approx2 = standard_normal_logprob(approx21).view(input_y.size()[0], -1).sum(1, keepdim=True)
        
            delta_log_p2 = delta_log_p2.view(input_y.size()[0], input_y.shape[1], 1).sum(1)
            log_p2 = (approx2 - delta_log_p2)

            loss_co = self.fit_co_teaching(input_y, condition_X, target)
            loss = -log_p2.mean() + loss_co

            self.optimizer.zero_grad()
            if self.config["image_task"]:
                self.encoder_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.config["image_task"]:
                self.encoder_optimizer.step()

            pbar.set_description(
                f'epoch: {epoch}, logP: {loss:.5f}')
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

        return acclist

    def fit(self) -> list:
        self.prior.train()
        self.encoder.train()
        if self.config["image_task"]:
            self.encoder.train()
        loss_list = []

        with tqdm(range(self.config["epochs"])) as pbar:
            for epoch in pbar:
                for i, x in enumerate(self.train_loader):
                    if (self.config["image_task"]):
                        # print("x[0] : ", x[0].size())
                        # print("x[1] : ", x[1].size())
                        # print("self.N_classes : ", self.N_classes)
                        input_y_one_hot = torch.nn.functional.one_hot(x[1], self.N_classes)
                        input_y_one_hot = input_y_one_hot.type(torch.cuda.FloatTensor)
                        input_y = input_y_one_hot.unsqueeze(1).to(self.device)
                        
                        condition_X_feature = self.encoder(x[0])
                        condition_X = condition_X_feature.unsqueeze(2).to(self.device)
                    else:
                        input_y = x[1].unsqueeze(1)
                        condition_X = x[0].unsqueeze(1)


                    if (self.config["add_uniform"]):
                        if (self.config["uniform_scheduler"]):
                            self.uniform_count = int(self.config["batch"] * self.config["uniform_rate"] * math.cos((math.pi / 2) * (epoch / self.config["epochs"])))
                        input_y, condition_X = addUniform(input_y, condition_X, self.uniform_count, self.X_mean, self.y_mean, self.X_var, self.y_var, self.config)

                    delta_p = torch.zeros(input_y.shape[0], input_y.shape[1], 1).to(input_y)
                    # print("input_y : ", input_y.size())
                    # print("condition_X : ", condition_X.size())
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

    def getWeightByEntropy(self,input_y, condition_X, sample_n = 1):
        self.prior.eval()
        self.encoder.eval()
        weight = torch.tensor([[1.]] * condition_X.size()[0]).to(self.device)

        condition_loc = condition_X.repeat(sample_n, 1, 1)
        input_y_loc = input_y.repeat(sample_n, 1, 1)
        
        input_normal = torch.normal(mean = 0.0, std = 0, size=(input_y_loc.shape)).to(self.device)
        delta_p_forword = torch.zeros(input_normal.shape[0], input_normal.shape[1], 1).to(input_y_loc)
        # print("input_normal : ", input_normal.size())
        # print("condition_X : ", condition_X.size())
        # print("delta_p_forword: ", delta_p_forword.size())
        approx21_forword, _ = self.prior(input_normal, condition_loc, delta_p_forword, reverse=True)
        approx21_forword = torch.clamp(approx21_forword, min=0, max=1)
        approxSum = torch.sum(approx21_forword, 2).unsqueeze(1).expand(approx21_forword.size())
        approx21_forword /= approxSum
        probs = approx21_forword
        
        probs = probs.view(sample_n, -1, self.N_classes)
        probs = torch.mean(probs, dim=0, keepdim=False)
        log_probs = torch.log(probs.clamp(min=self.eps))

        entropy = self.entropy_from_logprobs(log_probs).detach().cpu().numpy()
        # print(entropy)
        sort_entropy_idxs = np.argsort(entropy, axis=0)
        # print(sort_entropy_idxs)
        # accepted_idx = sort_entropy_idxs[int(-self.batch_size/4):]
        # accepted_idx = sort_entropy_idxs[int(-self.batch_size/10):]
        accepted_idx = sort_entropy_idxs[int(-self.batch_size/10):]
        # print(accepted_idx)
        # print(entropy[accepted_idx])
        weight[accepted_idx] = 0
        # print(weight)
        self.prior.train()
        self.encoder.train()
        return weight

    def getWeight(self, input_y, condition_X, uniform_count, X_mean, X_var):
        true_data_weight = torch.tensor([[1.]] * condition_X.size()[0]).to(self.device)

        batch_size = input_y.size(0)
        var_scale = self.config["var_scale"]
        X_uniform = np.random.uniform(X_mean - (var_scale * X_var), X_mean + (var_scale * X_var), uniform_count * self.cond_size).reshape(-1, self.cond_size, 1)
        X_uniform = torch.Tensor(X_uniform).to(condition_X)
        
        # y_uniform = np.random.uniform(y_mean - (var_scale * y_var), y_mean + (var_scale * y_var), uniform_count * self.config["inputDim"]).reshape(-1, 1, self.config["inputDim"])
        y_uniform_num = torch.randint(0, self.N_classes, (uniform_count,))
        y_uniform_one_hot = torch.nn.functional.one_hot(y_uniform_num, self.N_classes)
        y_uniform_one_hot = y_uniform_one_hot.type(torch.cuda.FloatTensor)
        y_uniform = y_uniform_one_hot.unsqueeze(1).to(self.device)

        idxs=torch.randperm(batch_size)
        # print("X_uniform ", X_uniform.size())
        # print("condition_X ", condition_X.size())
        # print("y_uniform ", y_uniform.size())
        # print("input_y ", input_y.size())
        condition_X = torch.cat((X_uniform, condition_X), 0)[idxs, :]
        input_y = torch.cat((y_uniform, input_y), 0)[idxs, :]
        

        # compute noise weight
        noise_logp = self.sampler.logprob(torch.tensor(X_uniform).float().to(self.device))
        
        # noise_weight = torch.clamp(1 - torch.exp(noise_logp), 0, 1)  # (1-p)
        # noise_weight = 1 / (1 + torch.exp(noise_logp))               # 1 / (1+p)
        noise_weight = torch.pow(10, -torch.exp(noise_logp))         # 10 ** -p

        weight = torch.cat([true_data_weight, noise_weight])[idxs, :]
        return input_y, condition_X, weight

    def fit_ce(self, input_y, condition_X, y_target) -> list:
        loss_ce_fn = torch.nn.CrossEntropyLoss()
        input_normal = torch.normal(mean = 0.0, std = 1.0, size=(input_y.shape)).to(self.device)
        delta_p_forword = torch.zeros(input_normal.shape[0], input_normal.shape[1], 1).to(input_y)
        # print("input_normal : ", input_normal.size())
        # print("condition_X : ", condition_X.size())
        # print("delta_p_forword: ", delta_p_forword.size())
        approx21_forword, delta_log_p2_forword = self.prior(input_normal, condition_X, delta_p_forword, reverse=True)
        smax = torch.nn.Softmax(dim=1)
        y_pre = smax(approx21_forword.squeeze(1))
        # print("y_pre", y_pre.size())
        # print("y_target", y_target.size())
        # print("y_pre", y_pre[:1])
        # print("y_target", y_target)
        loss_ce = loss_ce_fn(y_pre, y_target)
        return loss_ce

    def fit_co_teaching(self, input_y, condition_X, target) -> list:
        loss_ce_fn = torch.nn.CrossEntropyLoss()
        input_normal = torch.normal(mean = 0.0, std = 1.0, size=(input_y.shape)).to(self.device)
        delta_p_forword = torch.zeros(input_normal.shape[0], input_normal.shape[1], 1).to(input_y)
        # print("input_normal : ", input_normal.size())
        # print("condition_X : ", condition_X.size())
        # print("delta_p_forword: ", delta_p_forword.size())
        approx21_forword, delta_log_p2_forword = self.prior(input_normal, condition_X, delta_p_forword, reverse=True)
        approx21_forword = torch.clamp(approx21_forword, min=0, max=1)
        approxSum = torch.sum(approx21_forword, 2).unsqueeze(1).expand(approx21_forword.size())
        approx21_forword /= approxSum
        y_pre = approx21_forword
        # print("y_pre", y_pre.size())
        # print("y_target", y_target.size())
        # print("y_pre", y_pre[:1])
        # print("y_target", y_target)
        size = int(y_pre.size()[0] / 2)
        forget_rate = self.config['percent']
        y_pre = y_pre.squeeze(1)
        y_target = target[:size]
        # print("y_pre 1 ", y_pre[:size].size())
        # print("y_pre 2 ", y_pre[size:].size())
        # print("input_y ", y_target.size())
        loss_1, loss_2 = loss_coteaching(y_pre[:size], y_pre[size:], y_target, forget_rate)
        return loss_1 + loss_2

    def fit_sampler(self) -> list:
        self.prior.train()
        self.encoder.eval()
        loss_list = []

        with tqdm(range(self.config["epochs"])) as pbar:
            for epoch in pbar:
                for i, x in enumerate(self.train_loader):

                    if (self.config["image_task"]):
                        input_y_one_hot = torch.nn.functional.one_hot(x[1], self.N_classes)
                        input_y_one_hot = input_y_one_hot.type(torch.cuda.FloatTensor)
                        input_y = input_y_one_hot.unsqueeze(1).to(self.device)
                        
                        condition_X = x[0].unsqueeze(2).to(self.device)
                    else:
                        condition_X = x[0].unsqueeze(1)
                        input_y = x[1].unsqueeze(1)

                    input_y, condition_X, weight = self.getWeight(input_y, condition_X, self.uniform_count, self.X_mean, self.X_var)
                    delta_p = torch.zeros(input_y.shape[0], input_y.shape[1], 1).to(input_y)
                    # print("input_y : ", input_y.size())
                    # print("condition_X : ", condition_X.size())
                    # print("delta_p: ", delta_p.size())
                    approx21, delta_log_p2 = self.prior(input_y, condition_X, delta_p)

                    approx2 = standard_normal_logprob(approx21).view(input_y.size()[0], -1).sum(1, keepdim=True)
                
                    delta_log_p2 = delta_log_p2.view(input_y.size()[0], input_y.shape[1], 1).sum(1)
                    log_p2 = (approx2 - delta_log_p2)

                    # print("log_p2 : ", log_p2.size())
                    # print("weight : ", weight.size())

                    loss_ce = self.fit_ce(input_y, condition_X, x[1].to(self.device))
                    loss = -(log_p2 * weight).mean() + loss_ce

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

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

    def load_sampler(self, path) -> None:
        self.sampler = FlowSampler((self.cond_size, 1), '128-128', 1)
        self.sampler.load(path)

    def sample(self) -> None:
        self.prior.eval()

        mean_list = []
        var_list = []
        x_list = []

        for i, x in tqdm(enumerate(self.val_loader)):
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
    
    def sampling(self, loader, sample_n = 1, mean = 0.0, std = 1.0) -> float:
        self.prior.eval()
        self.encoder.eval()
        approx21_vec = []
        target_vec = []
        prob_vec = []
        probs_all_vec = []
        condition_feature_vec = []

        for i_batch, x in tqdm(enumerate(loader)):
            # y_one_hot = torch.nn.functional.one_hot(x[1], self.N_classes).to(self.device)
            
            image, target = x[0].to(self.device), x[1].to(self.device)

            condition_feature = self.encoder(image)
            condition_feature_vec.append(condition_feature.data.cpu())
            condition = condition_feature.unsqueeze(2).to(self.device)
            condition = condition.repeat(sample_n, 1, 1)

            input_z = torch.normal(mean = mean, std = std, size=(sample_n * self.config["batch"] , self.N_classes)).unsqueeze(1).to(self.device)
            delta_p = torch.zeros(input_z.shape[0], input_z.shape[1], 1).to(input_z)

            approx21, _ = self.prior(input_z, condition, delta_p, reverse=True)

            probs = torch.clamp(approx21, min=0, max=1)
            probsSum = torch.sum(probs, 2).unsqueeze(1).expand(probs.size())
            probs /= probsSum

            probs = probs.detach().squeeze(1)            
            probs = probs.view(sample_n, -1, self.N_classes)
            probs_all = probs
            probs_mean = torch.mean(probs, dim=0, keepdim=False)

            probs_all_vec.append(probs_all.data.cpu())
            approx21_vec.append(approx21.data.cpu())
            prob_vec.append(probs_mean.data.cpu())
            target_vec.append(target.data.cpu())

        prob_vec = torch.cat(prob_vec, dim=0)
        target_vec = torch.cat(target_vec, dim=0)
        approx21_vec = torch.cat(approx21_vec, dim=0)
        probs_all_vec = torch.cat(probs_all_vec, dim=1)
        condition_feature_vec = torch.cat(condition_feature_vec, dim=0)

        return prob_vec.data.cpu(), target_vec.data.cpu(), approx21_vec.data.cpu() ,probs_all_vec.data.cpu(), condition_feature_vec.data.cpu()

    def sampleImageAcc(self, MC_sample = 1, mean = 0.0, std = 0) -> float:        
        probs, target, _, _ = self.sampling(self.val_loader, MC_sample, mean, std)
        acc = accuracy(probs, target, topk=(1,))[0].cpu().item()
        print("acc : ", acc)
        return acc

    def get_preds_targets(self, val_loader):
        self.prior.eval()
        self.encoder.eval()
        logprob_vec = []
        target_vec = []
        entropy_vec = []

        for i_batch, x in tqdm(enumerate(val_loader)):
            # y_one_hot = torch.nn.functional.one_hot(x[1], self.N_classes).to(self.device)
            y = x[1].to(self.device)            
            condition_feature = self.encoder.forward_flatten(x[0])
            condition = condition_feature.unsqueeze(2).to(self.device)

            input_z = torch.normal(mean = 0.0, std = 1.0, size=(self.config["sample_count"] , self.N_classes)).unsqueeze(1).to(self.device)
            delta_p = torch.zeros(input_z.shape[0], input_z.shape[1], 1).to(input_z)

            approx21, delta_log_p2 = self.prior(input_z, condition, delta_p, reverse=True)
            # print("approx21", approx21[:1])
            approx21 = torch.clamp(approx21, min=0, max=1)
            approxSum = torch.sum(approx21, 2).unsqueeze(1).expand(approx21.size())
            approx21 /= approxSum
            # print("approx21", approx21[:1])
            probs = approx21.detach().squeeze(1)
            log_probs = torch.log(probs.clamp(min=self.eps))

            entropy_vec.append(self.entropy_from_logprobs(log_probs))
            logprob_vec.append(log_probs)
            target_vec.append(y)
    
        logprob_vec = torch.cat(logprob_vec, dim=0)
        target_vec = torch.cat(target_vec, dim=0)
        entropy_vec = torch.cat(entropy_vec, dim=0)

        return logprob_vec.data.cpu(), target_vec.data.cpu(), entropy_vec.data.cpu() 

    def entropy_from_logprobs(self, log_probs):
        return - (log_probs.exp() * log_probs).sum(dim=1)

    def flow_test_stats(self, dset, data_dir, corruption=None, rotation=None, batch_size=256, cuda=True, workers=4, iterate=False, no_ece=False):
        assert not (corruption is not None and rotation is not None)
        if corruption is None and rotation is None:
            _, _, val_loader, _, _, _ = \
                get_image_loader(dset, batch_size, cuda=cuda, workers=workers, distributed=False, data_dir=data_dir)
        elif corruption is not None:
            val_loader = load_corrupted_dataset(dset, severity=corruption, data_dir=data_dir, batch_size=batch_size,
                                                cuda=cuda, workers=workers)
        elif rotation is not None:
            val_loader = rotate_load_dataset(dset, rotation, data_dir=data_dir,
                                            batch_size=batch_size, cuda=cuda, workers=workers)

        # logprob_vec, target_vec = get_preds_targets(model, val_loader, cuda, MC_samples, return_vector=iterate)
        logprob_vec, target_vec, _ = self.get_preds_targets(val_loader)
        # print("logprob_vec", logprob_vec)
        # print("target_vec", target_vec)
        if iterate:
            brier_vec = []
            err_vec = []
            ll_vec = []
            ece_vec = []

            for n_samples in range(1, logprob_vec.shape[1]+1):
                comb_logprobs = torch.logsumexp(logprob_vec[:, :n_samples, :], dim=1, keepdim=False) - np.log(n_samples)
                # brier_vec.append(class_brier(y=target_vec, log_probs=comb_logprobs, probs=None))
                err_vec.append(class_err(y=target_vec, model_out=comb_logprobs))
                ll_vec.append(class_ll(y=target_vec, log_probs=comb_logprobs, probs=None, eps=1e-40))
                #ece_vec.append(float('nan') if no_ece else class_ECE(y=target_vec, log_probs=comb_logprobs,
                #                                                    probs=None, nbins=10))
            return err_vec, ll_vec, brier_vec, ece_vec

        # brier = class_brier(y=target_vec, log_probs=logprob_vec, probs=None)
        err = class_err(y=target_vec, model_out=logprob_vec)
        ll = class_ll(y=target_vec, log_probs=logprob_vec, probs=None, eps=1e-40)
        # ece = class_ECE(y=target_vec, log_probs=logprob_vec, probs=None, nbins=10)
        return  err, ll#, brier, ece

    def rot_measurements(self, rotations, corruptions):
        err_list = []
        ll_list = []
        # all measurements of err, ll, ece, brier
        for rotation, corruption in [(0, 0)] + corruptions[self.config["dataset"]] + rotations[self.config["dataset"]]:
            row_to_add = self.row_to_add_proto.copy()
            row_to_add.update({"rotation": rotation, "corruption": corruption})
            rotation = None if rotation == 0 else rotation
            corruption = None if corruption == 0 else corruption
            # err, ll, brier = self.flow_test_stats(self.config["dataset"], data_dir="./dataset", corruption=corruption, rotation=rotation, batch_size=self.config["sample_count"], cuda=True, workers=self.config["workers"])
            err, ll = self.flow_test_stats(self.config["dataset"], data_dir="./dataset", corruption=corruption, rotation=rotation, batch_size=self.config["sample_count"], cuda=True, workers=self.config["workers"])

            # row_to_add.update({"err": err, "ll": ll.item(), "brier": brier.item(), "ece": ece})
            row_to_add.update({"err": err, "ll": ll.item()})
            self.df = self.df.append(row_to_add, ignore_index=True)
            self.df.to_csv(self.df_path)  
            print("rot : ", rotation)
            print("corruption : ", corruption)
            print("err : ", err)
            print("ll : ", ll)
            err_list.append(err)
            ll_list.append(ll.item())
            
            # print("brier : ", brier)
            #print("ece : ", ece)
        
        return err_list, ll_list#, brier, ece

    def flow_class_rej(self, source_dset, target_dset, data_dir, batch_size=256, cuda=True,
                       rejection_step=0.005, workers=4):

        source_loader, target_loader = cross_load_dataset(source_dset, target_dset, data_dir=data_dir,
                                                        batch_size=batch_size, cuda=cuda, workers=workers)

        _, _, source_entropy = self.get_preds_targets(source_loader)
        _, _, target_entropy = self.get_preds_targets(target_loader)

        logprob_vec, target_vec, _ = self.get_preds_targets(source_loader)
        pred = logprob_vec.max(dim=1, keepdim=False)[1]  # get the index of the max probability
        err_vec_in = pred.ne(target_vec.data).cpu().numpy()
        err_vec_out = np.ones(target_entropy.shape[0])

        full_err_vec = np.concatenate([err_vec_in, err_vec_out], axis=0)
        full_entropy_vec = np.concatenate([source_entropy, target_entropy], axis=0)
        sort_entropy_idxs = np.argsort(full_entropy_vec, axis=0)
        Npoints = sort_entropy_idxs.shape[0]

        err_props = []

        for rej_prop in np.arange(0, 1, rejection_step):
            N_reject = np.round(Npoints * rej_prop).astype(int)
            if N_reject > 0:
                accepted_idx = sort_entropy_idxs[:-N_reject]
            else:
                accepted_idx = sort_entropy_idxs

            err_props.append(full_err_vec[accepted_idx].sum() / accepted_idx.shape[0])

            assert err_props[-1].max() <= 1 and err_props[-1].min() >= 0

        return np.array(err_props)
    # rejection measurements
    def rejection_measurements(self, target_datasets):
        err_props_list = []
        for target_dataset in target_datasets[self.config["dataset"]]:
            row_to_add = self.row_to_add_proto.copy()
            row_to_add.update({"target_dataset": target_dataset})
            # if len(df.loc[(df[list(row_to_add)] == pd.Series(row_to_add)).all(axis=1)]) > 0:
            #     continue
            err_props = self.flow_class_rej(self.config["dataset"], target_dataset, data_dir="./dataset",
                                                batch_size=self.config["sample_count"],
                                                cuda=True, workers=self.config["workers"])
            row_to_add.update({"err_props": err_props})
            self.df = self.df.append(row_to_add, ignore_index=True)
            self.df.to_csv(self.df_path)  
            err_props_list.append(err_props)
        return err_props_list