import matplotlib.pyplot as plt
import numpy as np

# Visualize the result
def visualize_uncertainty(savePath, gt_x, gt_y, xdata, mean, var):
    dyfit = 2 * np.sqrt(var)
    plt.plot(gt_x, gt_y, 'ok', ms=1)
    plt.plot(xdata, mean, '-', color='g')
    plt.fill_between(xdata, mean - dyfit, mean + dyfit, color='g', alpha=0.2)
    # plt.xlim(xdata.min(), xdata.max())
    plt.savefig(savePath)
    # if (wandb != None):
    #     wandb.log({"var": plt})