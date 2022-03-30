import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.flow import cnf


class CNFModel(pl.LightningModule):
    def __init__(self, learning_rate, batch_size):
        super().__init__()
        self.save_hyperparameters()
        flow_modules = '8-8-8-8-8'
        cond_size = 17
        num_blocks = 1
        inputDim = 1
        self.flow = cnf(inputDim, flow_modules, cond_size, num_blocks)

    def forward(self, x, context=None, logpx=None, integration_times=None):
        y = self.flow(x, context, logpx, integration_times)
        return y
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        approx21, delta_log_p2 = self.forward(x[0].squeeze(1), x[1], torch.zeros(self.hparams.batch_size, x[0].shape[2], 1).to(x[0]))

        approx2 = standard_normal_logprob(approx21).view(self.hparams.batch_size, -1).sum(1, keepdim=True)

        delta_log_p2 = delta_log_p2.view(self.hparams.batch_size, x[0].shape[2], 1).sum(1)

        log_p2 = (approx2 - delta_log_p2)

        loss = -log_p2.mean()

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        approx21, delta_log_p2 = self.forward(x[0].squeeze(1), x[1], torch.zeros(self.hparams.batch_size, x[0].shape[2], 1).to(x[0]))
        approx2 = standard_normal_logprob(approx21).view(self.hparams.batch_size, -1).sum(1, keepdim=True)

        delta_log_p2 = delta_log_p2.view(self.hparams.batch_size, x[0].shape[2], 1).sum(1)

        log_p2 = (approx2 - delta_log_p2)

        val_loss = -log_p2.mean()
        self.log("val_loss", val_loss)


    @staticmethod
    def standard_normal_logprob(z):
        dim = z.size(-1)
        log_z = -0.5 * dim * log(2 * pi)
        return log_z - z.pow(2) / 2