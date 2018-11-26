############

#   @File name: cubic.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-09-23 21:50:20

#   @Last modified by:  Xi He
#   @Last Modified time:    2018-11-15 13:58:56

#   @Description:
#   @Example:

############

from .optimizer import Optimizer, required
from utils.matrix_wrapper import leftRightMost, matrixDamping, rightMost
import numpy as np
from numpy import linalg as LA

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

class Cubic(Optimizer):
    def __init__(self, problem, lr=required, mode='exact', linesearch='strong_wolfe', adaptive=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid step-size: {}".format(lr))
        if not isinstance(adaptive, bool):
            raise ValueError("Invalid adaptive: {}".format(adaptive))
        if mode not in ['exact', 'cg', 'krylov', 'inexact', 'cauchy', 'adaNT']:
            raise ValueError("Invalid mode: {}".format(mode))

        if adaptive:
            params = dict(lr=lr, mode=mode, adaptive=adaptive)
        else:
            params = dict(lr=lr, mode=mode, linesearch=linesearch, adaptive=adaptive)

        super(Cubic, self).__init__(problem, params)

        self.subroutine.setMode('cubic')

    def getName(self):

        base = ['Cubic']
        adaptive = self.params['adaptive']
        if adaptive:
            features = ['mode', 'adaptive']
        else:
            features = ['mode', 'linesearch', 'adaptive']

        return self.getOptimName(base, features)

    def step(self, save=False, trace=False):

        lr = self.params['lr']
        adaptive = self.params['adaptive']

        grad = self.problem.getGrad(self.x)
        self.counter.incrementGradCount()

        eta_1 = float(config['CUBIC']['WEAK_ACCEPTANCE_RATIO'])
        eta_2 = float(config['CUBIC']['STRONG_ACCEPTANCE_RATIO'])
        gamma_1 = float(config['CUBIC']['REGULARIZATION_EXPAND'])
        gamma_2 = float(config['CUBIC']['REGULARIZATION_DECAY'])
        sigma_min_tol = float(config['CUBIC']['MINIMAL_REGULARIZATION'])
        sigma_max_tol = float(config['CUBIC']['MAXIMAL_REGULARIZATION'])

        if adaptive:

            if 'sigma' not in self.buffer_state or 'f_old' not in self.buffer_state:
                self.buffer_state['sigma'] = 1.
                self.buffer_state['f_old'] = self.problem.getFn(self.x)

            sigma = self.buffer_state['sigma']
            f_old = self.buffer_state['f_old']

            p, _ = self.subRoutineStep(grad, sigma)

            f = self.problem.getFn(self.x + p)
            tmp = self.subModelReduction(p, grad, sigma)

            rho = (f_old - f)/tmp

            self.stepLog(save, trace)

            if rho <= eta_1:
                sigma = min(gamma_1 * sigma, sigma_max_tol)
                f = f_old
            elif rho >= eta_2:
                sigma = max(gamma_2 * sigma, sigma_min_tol)
                self.x += p
            else:
                 self.x += p

            self.buffer_state['sigma'] = sigma
            self.buffer_state['f_old'] = f

        else:
            max_sigma = float(config['CUBIC']['MAXIMAL_REGULARIZATION'])

            linesearch = self.params['linesearch']

            p, _ = self.subRoutineStep(grad, max_sigma)

            if linesearch is not None:
                lr = self.linesearcher.lineSearch(self.x, p, grad, linesearch)

            self.stepLog(save, trace)

            self.x = self.x + lr * p

        self.counter.incrementStepCount()

    def subRoutineStep(self, grad, sigma):
        mode = self.params['mode']
        if mode == 'exact':
            hess = self.problem.getHessian(self.x)
            self.counter.incrementHessCount()

            p, lmd = self.subroutine.exactSolver(hess, -grad, sigma)

        if mode == 'cauchy':
            p, lmd = self.subroutine.cauchySolver(self.x, -grad, sigma)

        if mode == 'krylov':
            p, lmd = self.subroutine.lanczosSolver(self.x, -grad, sigma)

        if mode == 'adaNT':
            hess = self.problem.getHessian(self.x)
            self.counter.incrementHessCount()
            p, lmd = self.subroutine.adaNT_solver(hess, -grad, sigma)

        return p, lmd

    def subModelReduction(self, p, grad, sigma):
        tmp = grad.dot(p) + 0.5 * p.dot(self.problem.getHv(self.x, p)) + 1/3. *sigma * LA.norm(p) ** 3
        # assert tmp <=0, 'cubic submodel does not reduce.!!'

        return -tmp

    def summary(self):
        return
