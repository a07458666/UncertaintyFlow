from os import path
from numpy.random import uniform, randn
from sklearn.model_selection import train_test_split
import numpy as np
import torch.utils.data as data

class Datafeed(data.Dataset):

    def __init__(self, x_train, y_train=None, transform=None):
        self.data = x_train
        self.targets = y_train
        self.transform = transform

    def __getitem__(self, index):
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.targets is not None:
            return img, self.targets[index]
        else:
            return img

    def __len__(self):
        return len(self.data)

def load_my_1d(base_dir):
    if not path.exists(base_dir + '/my_1d_data/'):
        mkdir(base_dir + '/my_1d_data/')

        def gen_my_1d(hetero=False):

            np.random.seed(0)
            Npoints = 1002
            x0 = uniform(-1, 0, size=int(Npoints / 3))
            x1 = uniform(1.7, 2.5, size=int(Npoints / 3))
            x2 = uniform(4, 5, size=int(Npoints / 3))
            x = np.concatenate([x0, x1, x2])

            def function(x):
                return x - 0.1 * x ** 2 + np.cos(np.pi * x / 2)

            y = function(x)

            homo_noise_std = 0.25
            homo_noise = randn(*x.shape) * homo_noise_std
            y_homo = y + homo_noise

            hetero_noise_std = np.abs(0.1 * np.abs(x) ** 1.5)
            hetero_noise = randn(*x.shape) * hetero_noise_std
            y_hetero = y + hetero_noise

            X = x[:, np.newaxis]
            y_joint = np.stack([y_homo, y_hetero], axis=1)

            X_train, X_test, y_joint_train, y_joint_test = train_test_split(X, y_joint, test_size=0.5, random_state=42)
            y_hetero_train, y_hetero_test = y_joint_train[:, 1, np.newaxis], y_joint_test[:, 1, np.newaxis]
            y_homo_train, y_homo_test = y_joint_train[:, 0, np.newaxis], y_joint_test[:, 0, np.newaxis]

            x_means, x_stds = X_train.mean(axis=0), X_train.std(axis=0)
            y_hetero_means, y_hetero_stds = y_hetero_train.mean(axis=0), y_hetero_train.std(axis=0)
            y_homo_means, y_homo_stds = y_homo_test.mean(axis=0), y_homo_test.std(axis=0)

            X_train = ((X_train - x_means) / x_stds).astype(np.float32)
            X_test = ((X_test - x_means) / x_stds).astype(np.float32)

            y_hetero_train = ((y_hetero_train - y_hetero_means) / y_hetero_stds).astype(np.float32)
            y_hetero_test = ((y_hetero_test - y_hetero_means) / y_hetero_stds).astype(np.float32)

            y_homo_train = ((y_homo_train - y_homo_means) / y_homo_stds).astype(np.float32)
            y_homo_test = ((y_homo_test - y_homo_means) / y_homo_stds).astype(np.float32)

            if hetero:
                return X_train, y_hetero_train, X_test, y_hetero_test
            else:
                return X_train, y_homo_train, X_test, y_homo_test

        X_train, y_homo_train, X_test, y_homo_test = gen_my_1d()
        xy = np.concatenate([X_train, y_homo_train, X_test, y_homo_test], axis=1)
        np.save(base_dir + '/my_1d_data/my_1d_data.npy', xy)
        return X_train, y_homo_train, X_test, y_homo_test

    xy = np.load(base_dir + '/my_1d_data/my_1d_data.npy')
    X_train = xy[:, 0, None].astype(np.float32)
    y_homo_train = xy[:, 1, None].astype(np.float32)
    X_test = xy[:, 2, None].astype(np.float32)
    y_homo_test = xy[:, 3, None].astype(np.float32)

    return X_train, y_homo_train, X_test, y_homo_test

def load_wiggle():

    np.random.seed(0)
    Npoints = 300
    x = randn(Npoints) * 2.5 + 5  # uniform(0, 10, size=Npoints)

    def function(x):
        return np.sin(np.pi * x) + 0.2 * np.cos(np.pi * x * 4) - 0.3 * x

    y = function(x)

    homo_noise_std = 0.25
    homo_noise = randn(*x.shape) * homo_noise_std
    y = y + homo_noise

    x = x[:, None]
    y = y[:, None]

    x_means, x_stds = x.mean(axis=0), x.std(axis=0)
    y_means, y_stds = y.mean(axis=0), y.std(axis=0)

    X = ((x - x_means) / x_stds).astype(np.float32)
    Y = ((y - y_means) / y_stds).astype(np.float32)

    return X, Y