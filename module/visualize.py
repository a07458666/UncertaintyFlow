import matplotlib.pyplot as plt
import numpy as np

show_range = 10
ylim = 3
# Visualize the result
def visualize_uncertainty(savePath, gt_x, gt_y, xdata, mean, var):
    plt.figure(dpi=200)
    var = np.sqrt(var)
    plt.plot(gt_x, gt_y, 'ok', ms=1)
    plt.plot(xdata, mean, '-', color='g')
    plt.plot(xdata, var, '-', color='r')
    plt.ylim([-ylim, ylim])
    plt.xlim([-show_range, show_range])
    mean = np.array(mean)
    var = np.array(var)
    plt.fill_between(xdata, mean - var, mean + var, color='g', alpha=0.1)
    plt.tight_layout()
    plt.savefig(savePath, format='png', bbox_inches='tight')