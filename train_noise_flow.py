import os
import json
import numpy as np
import argparse
import nni
import matplotlib.pyplot as plt


import torch
from torch import nn, optim
from torch.utils import data

from module.config import checkOutputDirectoryAndCreate, loadConfig, dumpConfig, showConfig
from trainer import UncertaintyTrainer

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
    log_name =  config.get("output_folder") + '_' + config.get('dataset') + '_' + config.get('algorithm') + '_' + config.get('noise_type') + '_' + \
                 str(config.get('percent')) + '.json'
    if os.path.exists('./log') is False:
        os.mkdir('./log')
    log_name = os.path.join('./log', log_name)
    return log_name

def main(config, device):
    trainer = UncertaintyTrainer(config, device)

    acc_list, acc_all_list = [], []
    best_acc, best_epoch = 0.0, 0

    #load pre model
    if (config.get("pretrain_flow", "") != ""):
        flow_path = config.get("pretrain_flow")
        trainer.load(flow_path)
    if (config.get("pretrain_encoder", "") != ""):
        encoder_path = config.get("pretrain_encoder")
        trainer.load_encoder(encoder_path)

    for epoch in range(config.get("epochs", 0)):
        if (config.get("ssl", False)):
            trainer.trainSSL(epoch)
        else:
            trainer.train(epoch)
        # evaluate 
        test_acc = trainer.sampleImageAcc()
        test_acc02 = trainer.sampleImageAcc(10, 0, 0.2)
        test_acc10 = trainer.sampleImageAcc(10, 0, 1)
        nni.report_intermediate_result(test_acc)
        if best_acc < test_acc:
            best_acc, best_epoch = test_acc, epoch

        print('Epoch [%d/%d] Test Accuracy on the %s test images: %.4f %%' % (
                epoch + 1, config.get('epochs', 0), trainer.num_test_images, test_acc))
        if (wandb != None):
                logMsg = {}
                logMsg["epoch"] = epoch
                logMsg["acc/val"] = test_acc
                logMsg["acc/val_0.2"] = test_acc02
                logMsg["acc/val_1.0"] = test_acc10
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

    if (args.gpu != ""):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu != "-1" else "cpu")
    print("device : ", device)

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
        wandb.define_metric("test_acc", summary="max")
        wandb.define_metric("loss_vic_cov", summary="min")
        wandb.define_metric("loss_vic_sim", summary="min")
        wandb.define_metric("loss_vic_var", summary="min")
        wandb.define_metric("drop_acc", summary="max")
        wandb.define_metric("drop_acc1.0", summary="max")
        wandb.define_metric("drop_acc0.2", summary="max")
        wandb.define_metric("drop_precision", summary="max")
        wandb.define_metric("drop_recall", summary="max")
        
    # main
    main(config, device)
    