
import torch
from torch import nn, optim
from torch.utils import data

import os
import numpy as np
import argparse
import nni
import matplotlib.pyplot as plt

from trainer import UncertaintyTrainer
from module.config import checkOutputDirectoryAndCreate, loadConfig, dumpConfig, showConfig
from module.resnet import MyResNet
from tqdm import tqdm
import json

try:
    import wandb
except ImportError:
    wandb = None
    logger.info("Install Weights & Biases for experiment logging via 'pip install wandb' (recommended)")


def save_results(config, last_ten, best_acc, best_epoch, jsonfile):
    result_dict = config
    result_dict['last10_acc_mean'] = last_ten.mean()
    result_dict['last10_acc_std'] = last_ten.std()
    result_dict['best_acc'] = best_acc
    result_dict['best_epoch'] = best_epoch
    with open(jsonfile, 'w') as out:
        json.dump(result_dict, out, sort_keys=False, indent=4)

def plot_results(epochs, test_acc, plotfile):
    plt.style.use('ggplot')
    plt.plot(np.arange(0, epochs), test_acc, label='scratch - acc')
    plt.xticks(np.arange(0, epochs + 1, max(1, epochs // 20))) # train epochs
    plt.xlabel('Epoch')
    plt.yticks(np.arange(0, 101, 10)) # Acc range: [0, 100]
    plt.ylabel('Acc divergence')
    plt.savefig(plotfile)

def get_log_name(path, config):
    # log_name =  config['dataset'] + '_' + config['algorithm'] + '_' + config['noise_type'] + '_' + \
    #             str(config['percent']) + '_seed' + str(config['seed']) + '.json'
    log_name =  config["output_folder"] + '_' + config['dataset'] + '_' + config['algorithm'] + '_' + config['noise_type'] + '_' + \
                 str(config['percent']) + '.json'
    if os.path.exists('./log') is False:
        os.mkdir('./log')
    log_name = os.path.join('./log', log_name)
    return log_name

def sample_condition(train_loader, encoder):
    x = []
    for data in tqdm(train_loader):
        condition_X_feature = encoder(data[0])
        x.append(condition_X_feature)

    x = torch.cat(x, dim=0)
    return x.detach().cpu().numpy()

def main(config, device):
    trainer = UncertaintyTrainer(config, device)
    acc_list, acc_all_list = [], []
    best_acc, best_epoch = 0.0, 0

    #load pre model
    if (config["pretrain_encoder"] != ""):
        encoder_path = "./result/cifar_noise_fix_encoder_sym05_lr1e2/encoder_90.pt"
        trainer.load_encoder(encoder_path)
    # model_path = "./result/cifar_noise_fix_encoder_sym05_lr1e2/flow_90.pt"
    # trainer.load(model_path)
    

    for epoch in range(config["epochs"]):
        if (config["ssl"]):
            trainer.trainSSL(epoch)
        else:
            trainer.train(epoch)
        # evaluate 
        test_acc = trainer.sampleImageAcc()
        nni.report_intermediate_result(test_acc)
        if best_acc < test_acc:
            best_acc, best_epoch = test_acc, epoch

        print('Epoch [%d/%d] Test Accuracy on the %s test images: %.4f %%' % (
                epoch + 1, config['epochs'], trainer.num_test_images, test_acc))
        if (wandb != None):
                logMsg = {}
                logMsg["epoch"] = epoch
                logMsg["test_acc"] = test_acc
                wandb.log(logMsg)

        if epoch >= config['epochs'] - 10:
            acc_list.extend([test_acc])
        acc_all_list.extend([test_acc])
    trainer.save(f'result/{config["output_folder"]}/flow_last.pt')
    trainer.save_encoder(f'result/{config["output_folder"]}/encoder_last.pt')
    acc_np = np.array(acc_list)
    nni.report_final_result(acc_np.mean())
    jsonfile = get_log_name(args.config, config)
    np.save(jsonfile.replace('.json', '.npy'), np.array(acc_all_list))
    save_results(config=config, last_ten=acc_np, best_acc=best_acc, best_epoch=best_epoch, jsonfile=jsonfile)
    plot_results(epochs=config['epochs'], test_acc=acc_all_list, plotfile=jsonfile.replace('.json', '.png'))

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
    print("device : ", device)
    # main
    main(config, device)
    