import argparse
import torch
import os
from torch import nn, optim
from torch.utils import data

from module.resnet import MyResNet
from module.dun_datasets.image_loaders import get_image_loader
from tqdm import tqdm
from module.config import checkOutputDirectoryAndCreate, loadConfig, dumpConfig, showConfig
from module.noise_datasets.noise_datasets import cifar_dataloader
from module.losses.vicreg import vicreg_loss_func
from module.mixup import mixup_data, mixup_criterion
from module.auto_drop import auto_drop_byTarget

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

class TrainImageClassification():
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        self.lossType = self.config.get("lossType", "ce")
        self.output_folder = config["output_folder"]
        self.batch_size = config["batch"]
        self.dataset = config["dataset"]
        self.worker = config["workers"]
        self.cond_size = config["cond_size"]
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.loadNoiseDataset()
        self.model = MyResNet(in_channels = self.input_channels, feature_dim = self.cond_size, isPredictor = True, isLinear = True, num_classes = self.N_classes).to(self.device)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.loss_fn = nn.CrossEntropyLoss()
        

    def loadImageDataset(self, config) -> None:
        _, self.train_loader, self.val_loader, self.input_channels, self.N_classes, _ = get_image_loader(self.dataset, batch_size=self.batch_size, cuda=True, workers=self.worker, distributed=False)
        return 

    
    def loadNoiseDataset(self) -> None:
        dataloaders = cifar_dataloader(cifar_type=self.config['dataset'], root="./dataset", batch_size=self.batch_size, 
                            num_workers=self.config["workers"], noise_type=self.config['noise_type'], percent=self.config['percent'])
        if (self.lossType == "vic" or self.lossType == "vic_mixup"):
            self.train_loader = dataloaders.run(mode='train')
        else:
            self.train_loader = dataloaders.run(mode='train_single')
        self.val_loader = dataloaders.run(mode='test')

        self.N_classes = 10
        self.input_channels = 3
        self.num_test_images = len(self.val_loader.dataset)
        return

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top
        predictions for the specified values of k
        """
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def train(self, loader):
        self.model.train()
        loss = 0
        acc = 0
        acc_noise = 0
        acc_clean = 0
        pbar = tqdm(enumerate(loader))
        for batch_idx, (data, target, correct, target_real) in pbar:
            data, target = data.to(self.device), target.to(self.device)
            correct, target_real = correct.to(self.device), target_real.to(self.device)

            output = self.model(data)
            loss_batch = self.loss_fn(output, target)
            # acc_batch = self.accuracy(output, target_real)[0].cpu()
            
            loss += loss_batch.item()

            self.optimizer.zero_grad()
            loss_batch.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.model.eval()
                logits = self.model(data)
                acc_batch = self.accuracy(logits, target_real)[0].cpu()
                acc_noise_batch = self.accuracy(logits[correct == False], target_real[correct == False])[0].cpu()
                acc_clean_batch = self.accuracy(logits[correct == True], target_real[correct == True])[0].cpu()
                acc_noise += acc_noise_batch.item()
                acc_clean += acc_clean_batch.item()
                acc += acc_batch.item()
                self.model.train()
            pbar.set_description(f'epoch: {self.epoch}, acc: {float(acc / (batch_idx + 1)):.5f}, loss: {float(loss / (batch_idx + 1)):.5f}')
        return loss / (batch_idx + 1), acc / (batch_idx + 1), acc_noise / (batch_idx + 1), acc_clean / (batch_idx + 1)
    
    def trainAutoDrop(self, loader, logMsg):
        self.model.train()
        loss = 0
        acc = 0
        acc_noise = 0
        acc_clean = 0
        pbar = tqdm(enumerate(loader))
        for batch_idx, (data, target, correct, target_real) in pbar:
            data, target = data.to(self.device), target.to(self.device)
            correct, target_real = correct.to(self.device), target_real.to(self.device)

            output = self.model(data)
            
            drop_mask, drop_precision, drop_recall, drop_acc, drop_rate = auto_drop_byTarget(output, target, self.batch_size, correct, self.epoch)
            
            logMsg["drop_precision"] = drop_precision
            logMsg["drop_recall"] = drop_recall
            logMsg["drop_acc"] = drop_acc
            logMsg["drop_rate"] = drop_rate

            loss_ce = self.loss_fn(output, target)
            loss_batch = (loss_ce * (1-drop_mask.to(self.device)))

            loss += loss_batch.item()

            self.optimizer.zero_grad()
            loss_batch.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.model.eval()
                logits = self.model(data)
                acc_batch = self.accuracy(logits, target_real)[0].cpu()
                acc_noise_batch = self.accuracy(logits[correct == False], target_real[correct == False])[0].cpu()
                acc_clean_batch = self.accuracy(logits[correct == True], target_real[correct == True])[0].cpu()
                acc_noise += acc_noise_batch.item()
                acc_clean += acc_clean_batch.item()
                acc += acc_batch.item()
                self.model.train()
            pbar.set_description(f'epoch: {self.epoch}, acc: {float(acc / (batch_idx + 1)):.5f}, loss: {float(loss / (batch_idx + 1)):.5f}')
        return loss / (batch_idx + 1), acc / (batch_idx + 1), acc_noise / (batch_idx + 1), acc_clean / (batch_idx + 1)
    
    def trainMixup(self, loader):
        self.model.train()
        loss = 0
        loss_sup = 0
        acc = 0
        acc_noise = 0
        acc_clean = 0
        pbar = tqdm(enumerate(loader))
        for batch_idx, (data, target, correct, target_real) in pbar:
            data, target = data.to(self.device), target.to(self.device)
            correct, target_real = correct.to(self.device), target_real.to(self.device)
            
            mixed_x, targets_a, targets_b, lam, mixed_y = mixup_data(data, target)
            
            logits = self.model(mixed_x)

            loss_sup_batch = mixup_criterion(logits, targets_a, targets_b, lam)

            loss_batch = loss_sup_batch
            
            loss += loss_batch.item()
            loss_sup += loss_sup_batch.item()

            self.optimizer.zero_grad()
            loss_batch.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                self.model.eval()
                logits = self.model(data)
                acc_batch = self.accuracy(logits, target_real)[0].cpu()
                acc_noise_batch = self.accuracy(logits[correct == False], target_real[correct == False])[0].cpu()
                acc_clean_batch = self.accuracy(logits[correct == True], target_real[correct == True])[0].cpu()
                acc_noise += acc_noise_batch.item()
                acc_clean += acc_clean_batch.item()
                acc += acc_batch.item()
                self.model.train()
            pbar.set_description(f'epoch: {self.epoch}, acc: {float(acc / (batch_idx + 1)):.5f}, loss: {float(loss / (batch_idx + 1)):.5f}')
        return loss / (batch_idx + 1), loss_sup / (batch_idx + 1), acc / (batch_idx + 1), acc_noise / (batch_idx + 1), acc_clean / (batch_idx + 1)

    def trainSSL_Mixup(self, loader):
        self.model.train()
        loss = 0
        loss_vic = 0
        loss_sup = 0
        acc = 0
        acc_noise = 0
        acc_clean = 0
        pbar = tqdm(enumerate(loader))
        for batch_idx, (data, img1, img2, target, correct, target_real) in pbar:
            data, target = data.to(self.device), target.to(self.device)
            img1, img2 = img1.to(self.device), img2.to(self.device)
            correct, target_real = correct.to(self.device), target_real.to(self.device)

            # Self-learning
            _, _, z1, z2 = self.model.forward_ssl(img1, img2)
            loss_vic_batch, loss_vic_sim_batch, loss_vic_var_batch, loss_vic_cov_batch = vicreg_loss_func(z1, z2) # loss
            
            
            # Supervised-learning
            mixed_x, targets_a, targets_b, lam, _ = mixup_data(data, target)
            logits = self.model(mixed_x)
            loss_sup_batch = mixup_criterion(logits, targets_a, targets_b, lam)

            # loss
            loss_batch = loss_vic_batch + loss_sup_batch

            loss += loss_batch.item()
            loss_vic += loss_vic_batch.item()
            loss_sup += loss_sup_batch.item()

            self.optimizer.zero_grad()
            loss_batch.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                logits = self.model(data)
                acc_batch = self.accuracy(logits, target_real)[0].cpu()
                acc_noise_batch = self.accuracy(logits[correct == False], target_real[correct == False])[0].cpu()
                acc_clean_batch = self.accuracy(logits[correct == True], target_real[correct == True])[0].cpu()
                acc += acc_batch.item()
                acc_noise += acc_noise_batch.item()
                acc_clean += acc_clean_batch.item()
                self.model.train()

            pbar.set_description(f'epoch: {self.epoch}, acc: {float(acc / (batch_idx + 1)):.5f}, loss: {float(loss / (batch_idx + 1)):.5f}')
        return loss / (batch_idx + 1), loss_vic / (batch_idx + 1), loss_sup / (batch_idx + 1), acc / (batch_idx + 1), acc_noise / (batch_idx + 1), acc_clean / (batch_idx + 1)

    def trainSSL(self, loader):
        self.model.train()
        loss = 0
        loss_vic = 0
        loss_ce = 0
        acc = 0
        acc_noise = 0
        acc_clean = 0
        pbar = tqdm(enumerate(loader))
        for batch_idx, (data, img1, img2, target, correct, target_real) in pbar:
            data, target = data.to(self.device), target.to(self.device)
            img1, img2 = img1.to(self.device), img2.to(self.device)
            correct, target_real = correct.to(self.device), target_real.to(self.device)

            # Self-learning
            _, _, z1, z2 = self.model.forward_ssl(img1, img2)
            loss_vic_batch, loss_vic_sim_batch, loss_vic_var_batch, loss_vic_cov_batch = vicreg_loss_func(z1, z2) # loss
            
            
            # Supervised-learning
            # mixed_x, targets_a, targets_b, lam, _ = mixup_data(data, target)
            # logits = self.model(mixed_x)
            # loss_sup_batch = mixup_criterion(logits, targets_a, targets_b, lam)
            logits = self.model(data)
            loss_ce_batch = self.loss_fn(logits, target)

            # loss
            loss_batch = loss_vic_batch + loss_ce_batch

            loss += loss_batch.item()
            loss_vic += loss_vic_batch.item()
            loss_ce += loss_ce_batch.item()

            self.optimizer.zero_grad()
            loss_batch.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.model.eval()
                logits = self.model(data)
                acc_batch = self.accuracy(logits, target_real)[0].cpu()
                acc_noise_batch = self.accuracy(logits[correct == False], target_real[correct == False])[0].cpu()
                acc_clean_batch = self.accuracy(logits[correct == True], target_real[correct == True])[0].cpu()
                acc += acc_batch.item()
                acc_noise += acc_noise_batch
                acc_clean += acc_clean_batch
                self.model.train()
            pbar.set_description(f'epoch: {self.epoch}, acc: {float(acc / (batch_idx + 1)):.5f}, loss: {float(loss / (batch_idx + 1)):.5f}')
        return loss / (batch_idx + 1), loss_vic / (batch_idx + 1), loss_ce / (batch_idx + 1), acc / (batch_idx + 1), acc_noise / (batch_idx + 1), acc_clean / (batch_idx + 1)

    def val(self, loader):
        self.model.eval()
        loss = 0
        acc = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(loader))
            for batch_idx, (data, target) in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss_batch = self.loss_fn(output, target)
                acc_batch = self.accuracy(output, target)[0].cpu()

                loss += loss_batch.item()
                acc += acc_batch.item()

        return loss / (batch_idx + 1), acc / (batch_idx + 1)

    def fit(self):
        for self.epoch in range(self.epochs):
            logMsg = {}
            logMsg["epoch"] = self.epoch
            loss_val = 0
            acc_val = 0

            if (self.lossType == "vic"):
                logMsg["loss/train"], logMsg["loss/train_vic"], logMsg["loss/train_ce"], logMsg["acc/train"],logMsg["acc/train_noise"], logMsg["acc/train_clean"] = self.trainSSL(self.train_loader)
            elif (self.lossType == "mixup"):
                logMsg["loss/train"], logMsg["loss/train_sup"], logMsg["acc/train"],logMsg["acc/train_noise"], logMsg["acc/train_clean"] = self.trainMixup(self.train_loader)
            elif (self.lossType == "vic_mixup"):
                logMsg["loss/train"], logMsg["loss/train_vic"], logMsg["loss/train_sup"], logMsg["acc/train"],logMsg["acc/train_noise"], logMsg["acc/train_clean"] = self.trainSSL_Mixup(self.train_loader)
            elif (self.lossType == "auto_drop"):
                logMsg["loss/train"], logMsg["acc/train"],logMsg["acc/train_noise"], logMsg["acc/train_clean"] = self.trainAutoDrop(self.train_loader, logMsg)
            else:
                logMsg["loss/train"], logMsg["acc/train"], logMsg["acc/train_noise"], logMsg["acc/train_clean"] = self.train(self.train_loader)

            logMsg["loss/val"], logMsg["acc/val"] = self.val(self.val_loader)
            print(f'epoch: {self.epoch}, acc_val: {logMsg["acc/val"]:.5f}')

            self.scheduler.step()
            if (wandb != None):
                wandb.log(logMsg)
                wandb.watch(self.model,log = "all", log_graph=True)
            self.save(f'result/{self.output_folder}/encoder_{str(self.epoch).zfill(2)}.pt')


    def save(self, path):
        torch.save(self.model.state_dict(), path)
        return

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
        wandb.define_metric("acc", summary="max")

    if (args.gpu != ""):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "-1" else "cpu")

    # main
    trainer = TrainImageClassification(config, device)
    trainer.fit()