############

#   @File name: cutest_pb.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-08-17 23:57:33

#   @Last modified by:  Xi He
#   @Last Modified time:    2018-11-14 20:43:28

#   @Description:
#   @Example:

############

from problem import Problem
from cutest.model.cutestmodel import CUTEstModel
from numpy import linalg as LA
from copy import deepcopy

class CutestProblem(Problem):
    def __init__(self, func_name='ROSENBR'):
        if isinstance(func_name, int):
            self.size = func_name
        else:
            self.model = CUTEstModel(func_name)
            self.name = self.model.name
            self.initial_point = deepcopy(self.model.x0)
            self.size = len(self.model.x0)

    def getSize(self):
        return self.size

    def getInitialPoint(self):
        return self.initial_point

    def getInitialStatus(self, grad_norm_type = 2):
            initial_fn = self.getFn(self.initial_point)
            initial_grad_norm = LA.norm(self.getGrad(self.initial_point), grad_norm_type)
            return initial_fn, initial_grad_norm

    def setInitialPoint(self, x0=None):
        if x0 is None:
            print("Default initial point is used...")
            self.initial_point = deepcopy(self.model.x0)
        else:
            assert x0.shape == self.initial_point.shape, "Initial point dimension mismatch!"
            self.initial_point = x0

    def getModel(self):
        return self.model

    def getName(self):
        return self.model.name

    def getFn(self, x):
        self.fn = self.model.obj(x)
        return self.fn

    def getGrad(self, x):
        self.gn = self.model.grad(x)
        return self.gn

    def getHessian(self, x):
        self.Hn = self.model.hess_dense(x)
        return self.Hn

    def getHv(self, x, v):
        assert len(v) == len(x), "Hessian vector product dimension unmatched!!"
        self.Hv = self.model.hprod(x, 0, v)
        return self.Hv