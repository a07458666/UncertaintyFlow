import torch
import matplotlib.pyplot as plt

from module.dun_datasets.additional_gap_loader import *
from condition_sampler.flow_sampler import FlowSampler


# load data
x, y = load_agw_1d()

x_sampler = FlowSampler((1, 1), '64-128-256-128-64', 1)
loss = x_sampler.fit(x, batch=128, lr=5e-3, epoch=50)

# draw loss
print('draw loss')
plt.plot([i for i in range(len(loss))], loss)
plt.savefig('loss.png')
plt.clf()

# sample test
print('print sample')
sampled_x = x_sampler.sample(x.shape[0]*3).squeeze().cpu().numpy()

plt.hist(x.squeeze(), bins=32, color='r')
plt.hist(sampled_x, bins=32*2, color='g')
plt.savefig('sample_result.png')

# print log liklihood
x = torch.Tensor([[[-2.]], [[-1.5]], [[-1.]], [[-0.5]], [[0.]], [[0.5]], [[1.]], [[1.5]], [[2.]]]).cuda()
print(torch.exp(x_sampler.logprob(x)))
