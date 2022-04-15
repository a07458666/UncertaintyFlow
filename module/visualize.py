import matplotlib.pyplot as plt
import numpy as np

# Visualize the result
def visualize_uncertainty(savePath, gt_x, gt_y, xdata, mean, var):
    dyfit = 2 * np.sqrt(var)
    plt.plot(gt_x, gt_y, 'ok', ms=1)
    plt.plot(xdata, mean, '-', color='g')
    plt.plot(xdata, var, '-', color='r')
    plt.fill_between(xdata, mean - dyfit, mean + dyfit, color='g', alpha=0.2)
    plt.savefig(savePath)