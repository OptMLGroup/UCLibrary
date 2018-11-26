############

#   @File name: newton.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-08-18 00:29:39

# @Last modified by:   Heerye
# @Last modified time: 2018-08-18T07:55:00-04:00

#   @Description:
#   @Example:

############

from .optimizer import Optimizer, required
from utils.matrix_wrapper import leftRightMost, matrixDamping, leftMost
import numpy as np

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

class Newton(Optimizer):
    def __init__(self, problem, lr=required, damping='LM', mode='exact', linesearch='strong_wolfe'):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid step-size: {}".format(lr))
        if damping not in ['LM', 'constant', 'min_eig', True, False]:
            raise ValueError("Invalid damping: {}".format(damping))
        if mode not in ['exact', 'cg', 'krylov']:
            raise ValueError("Invalid mode: {}".format(mode))
        params = dict(lr=lr, damping=damping, mode=mode, linesearch=linesearch)
        super(Newton, self).__init__(problem, params)

        self.subroutine.setMode('newton')

    def getName(self):
        base = ['Newton']
        features =  ['damping', 'mode', 'linesearch']
        return self.getOptimName(base, features)

    def step(self, save=False, trace=False):

        min_eig_tol = float(config['NEWTON']['MINIMAL_EIGENVALUE_TOL'])

        lr = self.params['lr']
        damping = self.params['damping']
        mode = self.params['mode']
        linesearch = self.params['linesearch']

        grad = self.problem.getGrad(self.x)
        self.counter.incrementGradCount()

        if mode == 'exact':
            hess = self.problem.getHessian(self.x)

            if damping == 'min_eig':
                leftmost = leftMost(hess)
                if leftmost <= min_eig_tol:
                    lmd = min_eig_tol * 2 - leftmost
                else:
                    lmd = 0.

            p = self.subroutine.exactSolver(matrixDamping(hess, lmd), -grad)

        if mode == 'cg':
            p = self.subroutine.cgSolver(self.x, -grad)

        self.counter.incrementSubrountineCount()

        if linesearch is not None:
            lr = self.linesearcher.lineSearch(self.x, p, grad, linesearch, lr =1.)

        self.stepLog(save, trace)

        self.x = self.x + lr * p
        self.counter.incrementStepCount()

    def summary(self):
        return
