############

#   @File name: nn_main.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-07-01 12:31:00

#   @Last modified by:  Xi He
#   @Last Modified time:    2018-10-22 10:57:11

#   @Description:
#   @Example:

############

from problems.nn_pb import nnProblem
import optimizers

import numpy
import torch
from utils.synthetic_data import SyntheticData
from utils.net_zoo import Net

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

seed = int(config['DEFAULT']['RANDOM_SEED'])

numpy.random.seed(seed)
torch.manual_seed(seed)

data = SyntheticData(2)
net = Net([data.n_features, 5, data.n_classes])

problem = nnProblem(data = SyntheticData(1), net = net)

print(problem.getParameters())
print(problem.getSize())
problem.setInitialPoint()

# print (problem.getName())
# c, f = problem.getAttrs()

# x = problem.initial_point
# v = np.ones(x.shape)

# fx = problem.getFn(x)

# optim = optimizers.GD(problem, lr=0.001, momentum=0.99, linesearch=None)
# print(optim.getName())

# for _ in range(10):
#     print(x)
#     x = optim.step(x)
# print(x)