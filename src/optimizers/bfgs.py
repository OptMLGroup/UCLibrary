# @Author: Xi He <Heerye>
# @Date:   2018-08-18T07:50:43-04:00
# @Email:  xih314@lehigh.edu; heeryerate@gmail.com
# @Filename: bfgs.py
# @Last modified by:   Heerye
# @Last modified time: 2018-08-18T07:54:41-04:00

from .optimizer import Optimizer, required
import numpy as np
from numpy import linalg as LA

class BFGS(Optimizer):
    def __init__(self, problem, lr=required, mode = 'vanilla', linesearch='strong-wolfe'):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid step-size: {}".format(lr))
        params = dict(lr=lr, linesearch=linesearch)
        super(BFGS, self).__init__(problem, params)

    def getName(self):
        base = ['BFGS']
        for feature in ['mode', 'linesearch']:
            if self.params[feature]:
                base.append(feature)
        return '-'.join(base)

    def step(self, x):
        lr = self.params['lr']
        linesearch = self.params['linesearch']

        B = np.eyes(x.size)
        I = np.eyes(x.size)

        curvature_threholds = 1e-3

        grad = self.problem.getGrad(x)
        p = LA.solve(B, -grad)

        x_new = x + lr * p

        s = (x_new-x).reshape(1, x.size)
        grad_new = self.problem.getGrad(x_new)
        y = (grad_new-grad).reshape(1, x.size)

        rho = s.dot(y.T)
        tmp = s.T.dot(y)







        if linesearch is not None:
            lr = self.lineSearch(x, p, grad, linesearch)

        x = x - lr * p

        return x
