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

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")

class TrainImageClassification():
    def __init__(self, config, device) -> None:
        self.config = config
        self.device = device
        self.output_folder = config["output_folder"]
        self.batch_size = config["batch"]
        self.dataset = config["dataset"]
        self.worker = config["workers"]
        self.cond_size = config["cond_size"]
        self.epoch = config["epochs"]
        self.lr = config["lr"]
        # self.loadImageDataset(config)
        self.loadNoiseDataset()
        if (self.config["ssl"]):
            self.model = MyResNet(in_channels = self.input_channels, out_features = self.cond_size).to(self.device)
        else:
            self.model = MyResNet(in_channels = self.input_channels, out_features = self.N_classes).to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=5e-4)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch)
        self.loss_fn = nn.CrossEntropyLoss()
        

    def loadImageDataset(self, config) -> None:
        _, self.train_loader, self.val_loader, self.input_channels, self.N_classes, _ = get_image_loader(self.dataset, batch_size=self.batch_size, cuda=True, workers=self.worker, distributed=False)
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
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            
            output = self.model(data)
            loss_batch = self.loss_fn(output, target)
            acc_batch = self.accuracy(output, target)[0].cpu()
            
            loss += loss_batch.item()
            acc += acc_batch
            # print("batch_idx {} acc_batch {} acc {}".format(batch_idx, acc_batch, acc))

            self.optimizer.zero_grad()
            loss_batch.backward()
            self.optimizer.step()
        return loss / (batch_idx + 1), acc / (batch_idx + 1)

    def trainSSL(self, loader):
        self.model.train()
        loss = 0
        acc = 0
        for batch_idx, (data, img1, img2, target) in enumerate(loader):
            data, target = data.to(self.device), target.to(self.device)
            img1, img2 = img1.to(self.device), img2.to(self.device)
            # output = self.model(data)
            # loss_batch = self.loss_fn(output, target)
            # acc_batch = self.accuracy(output, target)[0].cpu()
            _, _, z1, z2 = self.model.forward_ssl(img1, img2)
            loss_vic, loss_vic_sim, loss_vic_var, loss_vic_cov = vicreg_loss_func(z1, z2) # loss

            loss += loss_vic.item()
            acc = 0
            # acc += acc_batch
            # print("batch_idx {} acc_batch {} acc {}".format(batch_idx, acc_batch, acc))

            self.optimizer.zero_grad()
            loss_vic.backward()
            self.optimizer.step()
        return loss / (batch_idx + 1), acc / (batch_idx + 1)

    def val(self, loader):
        self.model.eval()
        loss = 0
        acc = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loader):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss_batch = self.loss_fn(output, target)
                acc_batch = self.accuracy(output, target)[0].cpu()

                loss += loss_batch.item()
                acc += acc_batch

        return loss / (batch_idx + 1), acc / (batch_idx + 1)

    def fit(self):
        with tqdm(range(self.epoch)) as pbar:
            for epoch in pbar:
                loss_val = 0
                acc_val = 0
                if (self.config["ssl"]):
                    loss_train, acc_train = self.trainSSL(self.train_loader)
                else:
                    loss_train, acc_train = self.train(self.train_loader)
                    loss_val, acc_val = self.val(self.val_loader)
                self.scheduler.step()
                if (wandb != None):
                    logMsg = {}
                    logMsg["epoch"] = epoch
                    logMsg["loss_train"] = loss_train
                    logMsg["acc_train"] = acc_train
                    logMsg["loss_val"] = loss_val
                    logMsg["acc_val"] = acc_val
                    wandb.log(logMsg)
                    wandb.watch(self.model,log = "all", log_graph=True)
                self.save(f'result/{self.output_folder}/encoder_{str(epoch).zfill(2)}.pt')


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