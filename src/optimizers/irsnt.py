############

#   @File name: irsnt.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-09-23 21:50:20

#   @Last modified by:  Xi He
#   @Last Modified time:    2019-02-03 16:42:28

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

class Irsnt(Optimizer):
    def __init__(self, problem):
        super(Irsnt, self).__init__(problem)

    def getName(self):
        base = ['Irsnt']
        return self.getOptimName(base)

    def step(self, save=False, trace=False):

        grad = self.problem.getGrad(self.x)
        self.counter.incrementGradCount()

        eta_1 = float(config['IRSNT']['WEAK_ACCEPTANCE_RATIO'])
        eta_2 = float(config['IRSNT']['STRONG_ACCEPTANCE_RATIO'])
        gamma_1 = float(config['IRSNT']['SIGMA_EXPAND'])
        gamma_2 = float(config['IRSNT']['SIGMA_DECAY'])
        sigma_min = float(config['IRSNT']['MINIMAL_SIGMA'])
        sigma_max = float(config['IRSNT']['MAXIMAL_SIGMA'])

        if 'sigma_L' not in self.buffer_state or 'sigma_U' not in self.buffer_state or 'f_old' not in self.buffer_state:
            self.buffer_state['sigma_L'] = 1e-6
            self.buffer_state['sigma_U'] = 1.
            self.buffer_state['f_old'] = self.problem.getFn(self.x)

        sigma_L = self.buffer_state['sigma_L']
        sigma_U = self.buffer_state['sigma_U']
        f_old = self.buffer_state['f_old']

        p, lmd = self.subRoutineStep(grad, sigma_L, sigma_U)
        normp = LA.norm(p)

        # print('subproblem:')
        # print(LA.norm(self.problem.getHv(self.x, p) + lmd * p + grad), sigma_L, lmd/normp, sigma_U, self.x)

        f = self.problem.getFn(self.x + p)
        self.counter.incrementFnCount()
        tmp = self.subModelReduction(p, grad, lmd)

        rho = (f_old - f)/tmp

        # print('xxx', f_old, f, tmp, rho, p)

        self.stepLog(save, trace)

        if rho >= eta_1:
            self.x += p
            if normp <= max(LA.norm(grad), lmd):
                sigma_L = 0.
                sigma_U = min(gamma_2 * lmd / normp, sigma_min)
            else:
                sigma_L = min(sigma_min, gamma_2 * lmd / normp)
                sigma_U = max(sigma_L, lmd / normp)
        else:
             sigma_L = max(sigma_min, gamma_1 * lmd / normp)
             sigma_U = max(gamma_1 * lmd / normp, np.random.uniform(sigma_L, sigma_max))
             f = f_old

        self.buffer_state['sigma_L'] = sigma_L
        self.buffer_state['sigma_U'] = sigma_U
        self.buffer_state['f_old'] = f

        self.counter.incrementStepCount()

    def subRoutineStep(self, grad, sigma_L, sigma_U):
        p, lmd = self.subroutine.adaNTSolver(self.x, -grad, sigma_L, sigma_U)
        return p, lmd

    def subModelReduction(self, p, grad, lmd):
        tmp = grad.dot(p) + 0.5 * p.dot(self.problem.getHv(self.x, p)) + 0.5 * lmd * LA.norm(p) ** 2
        self.counter.incrementHessVCount()
        assert tmp <= 0, 'irsnt submodel does not reduce.!!'

        return -tmp

    def summary(self):
        return
