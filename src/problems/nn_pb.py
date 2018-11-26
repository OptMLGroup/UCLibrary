############

#   @File name: nn_pb.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-08-17 23:57:33

#   @Last modified by:  Xi He
#   @Last Modified time:    2018-10-22 10:56:45

#   @Description:
#   @Example:

############

from .problem import Problem
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, grad

import numpy
from numpy import linalg as LA
from scipy.stats import ortho_group

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

class nnProblem(Problem):
    def __init__(self, data, net):
        super(nnProblem, self).__init__(func_name='NN')
        self.data = data
        self.net = net

    def getParameters(self):
        self.params = self.getFlatParameters()
        return self.params

    def getFlatParameters(self):
        """pytorch/pytorch/blob/master/torch/optim/lbfgs.py
        """
        views = []
        for p in self.net.parameters():
            view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0).numpy()

    def getFlatGrad(self):
        """pytorch/pytorch/blob/master/torch/optim/lbfgs.py
        """
        views = []
        for p in self.net.parameters():
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0).numpy()

    def getSize(self):
        return len(self.params)

    # def getParameters(self):
    #     self.params = self.net.parameters()
    #     print(self.getFlatGrad())
    #     return self.params

    def condenseParameters(self):
        return

    def getInitialPoint(self):
        return self.initial_point

    def getInitialStatus(self, grad_norm_type = 2):
        initial_fn = self.getFn(self.initial_point)
        initial_grad_norm = LA.norm(self.getGrad(self.initial_point), grad_norm_type)
        return initial_fn, initial_grad_norm

    def setInitialPoint(self):
        for p in self.net.parameters():
            print(p)
            # p.data.normal_(0.0, 1./sum(p.shape))

        # m.weight.data.normal_(1.0, 0.02)
        # m.bias.data.fill_(0)
        # if x0 is None:
        #     print("Default initial point is used...")
        #     self.initial_point = deepcopy(self.model.x0)
        # else:
        #     assert x0.shape == self.initial_point.shape, "Initial point dimension mismatch!"
        #     self.initial_point = x0

    def getModel(self):
        return self.model

    def getName(self):
        # return self.model.name
        return self.name

    def getFn(self, x):
        # self.fn = self.model.obj(x)
        # return self.fn
        fx = self.net(x)
        loss_type = str(config['NN']['LOSS_TYPE'])
        if loss_type == 'CrossEntropyLoss':
            loss = torch.nn.CrossEntropyLoss()(fx, y)
        elif loss_type == 'MSELoss':
            loss = torch.nn.MSELoss()(fx, y)
        else:
            raise ValueError('invalid loss type: %s.'%loss_type)
        return

    def getGrad(self, x):
        # self.gn = self.model.grad(x)
        # return self.gn
        params = self.getParameters()
        loss = self.getFn(x)
        g = torch.autograd.grad(loss, params, creat_graph=True)
        return g

    def getHessian(self, x):
        self.Hn = self.model.hess_dense(x)
        return self.Hn

    def getHv(self, x, v):
        assert len(v) == len(x), "Hessian vector product dimension unmatched!!"
        self.Hv = self.model.hprod(x, 0, v)
        return self.Hv

    # def getInitialPoint(self):
    #     return None

    # def getModel(self):
    #     return None

    # def getAttrs(self):
    #     print('#classes: {}, #features: {}'.format(self.n_class, self.n_feat))
    #     return self.n_class, self.n_feat

    # def getName(self):
    #     return self.name

    # def getFn(self, x):
    #     return None

    # def getGrad(self, x):
    #     return None

    # def getHessian(self, x):
    #     return None

    # def getHv(self, x, v):
    #     return None

