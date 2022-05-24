import os
import math
from tqdm import tqdm
import numpy as np

# torch
import torch
from torch import optim
from torch.utils import data
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

# module
from module.flow import cnf
from module.utils import standard_normal_logprob, accuracy, set_parameter_requires_grad
from module.resnet import MyResNet
from module.noise_datasets.noise_datasets import cifar_dataloader

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
        self.loadNoiseDataset()
        self.creatModel()
        self.defOptimizer()
        self.model_scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config["epochs"])
        self.encoder_scheduler = CosineAnnealingLR(self.encoder_optimizer, T_max=self.config["epochs"])
        return

    def loadNoiseDataset(self) -> None:
        dataloaders = cifar_dataloader(cifar_type=self.config['dataset'], root="./dataset", batch_size=self.batch_size, 
                            num_workers=self.config["workers"], noise_type=self.config['noise_type'], percent=self.config['percent'])
        if (self.config.get("ssl", False)):
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
        if (self.config.get("fix_encoder", False)):
            set_parameter_requires_grad(self.encoder)
        return

    def defOptimizer(self) -> None:
        self.optimizer = optim.AdamW(self.prior.parameters(), lr=self.config["lr"])
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
            if (self.config.get("blur", False)):
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
            if (not self.config.get("fix_encoder", False)):
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
                if (not self.config.get("fix_encoder", False)):
                    logMsg["feature_grad"] = condition_X.grad.mean().item()
                wandb.log(logMsg)
                wandb.watch(self.prior,log = "all", log_graph=True)
        
        # epoch
        self.model_scheduler.step()
        self.encoder_scheduler.step()
        self.save(f'result/{self.config["output_folder"]}/flow_{str(epoch).zfill(2)}.pt')
        self.save_encoder(f'result/{self.config["output_folder"]}/encoder_{str(epoch).zfill(2)}.pt')

        return acclist

    def trainSSL(self, epoch) -> list:
        self.prior.train()
        self.encoder.train()
        acclist = []
        loss_list = []

        pbar = tqdm(self.train_loader)
        for i, x in enumerate(pbar):
            image, img1, img2, target = x[0].to(self.device), x[1].to(self.device), x[2].to(self.device), x[3].to(self.device)
            # print("x[0] : ", x[0].size())
            # print("target : ", target.size())
            # print("target : ", target)
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

    def save(self, path) -> None:
        torch.save(self.prior.state_dict(), path)

    def load(self, path) -> None:
        self.prior.load_state_dict(torch.load(path))

    def save_encoder(self, path) -> None:
        torch.save(self.encoder.state_dict(), path)

    def load_encoder(self, path) -> None:
        self.encoder.load_state_dict(torch.load(path))
    
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
        probs, target, _, _, _ = self.sampling(self.val_loader, MC_sample, mean, std)
        acc = accuracy(probs, target, topk=(1,))[0].cpu().item()
        print("acc : ", acc)
        return acc