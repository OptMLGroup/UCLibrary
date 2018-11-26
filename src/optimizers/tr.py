############

#   @File name: tr.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-08-18 00:29:39

# @Last modified by:   Heerye
# @Last modified time: 2018-08-18T07:55:00-04:00

#   @Description:
#   @Example:

############

from .optimizer import Optimizer, required
from utils.matrix_wrapper import leftRightMost, matrixDamping, rightMost, leftMost
import numpy as np
from pprint import pprint
import numpy.linalg as LA

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

class TrustRegion(Optimizer):
    def __init__(self, problem, mode='exact'):
        if mode not in ['exact', 'cg', 'gltr', 'inexact', 'cauchy', 'dogleg', 'adaNT']:
            raise ValueError("Invalid mode: {}".format(mode))
        params = dict(mode=mode)
        super(TrustRegion, self).__init__(problem, params)

        self.subroutine.setMode('tr')

    def getName(self):
        base = ['TR']
        features = ['mode']
        return self.getOptimName(base, features)

    def step(self, save=False, trace=False):

        mode = self.params['mode']

        grad = self.problem.getGrad(self.x)
        self.counter.incrementGradCount()

        eta_1 = float(config['TRUSTREGION']['WEAK_ACCEPTANCE_RATIO'])
        eta_2 = float(config['TRUSTREGION']['STRONG_ACCEPTANCE_RATIO'])
        gamma_1 = float(config['TRUSTREGION']['RADIUS_EXPAND'])
        gamma_2 = float(config['TRUSTREGION']['RADIUS_DECAY'])
        radius_min_tol = float(config['TRUSTREGION']['MINIMAL_RADIUS'])
        radius_max_tol = float(config['TRUSTREGION']['MAXIMAL_RADIUS'])

        if 'radius' not in self.buffer_state or 'f_old' not in self.buffer_state:
            self.buffer_state['radius'] = 1.
            self.buffer_state['f_old'] = self.problem.getFn(self.x)

        radius = self.buffer_state['radius']
        f_old = self.buffer_state['f_old']

        p, lmd = self.subRoutineStep(grad, radius)

        f = self.problem.getFn(self.x + p)
        tmp = self.subModelReduction(p, grad, lmd, radius)


        rho = (f_old - f)/tmp

        self.stepLog(save, trace)

        # print(rho, f_old - f, tmp, radius)

        if rho <= eta_1:
            radius = max(gamma_2 * radius, radius_min_tol)
            f = f_old
        elif rho >= eta_2:
            self.x += p
            radius = min(gamma_1 * radius, radius_max_tol)
        else:
             self.x += p

        self.buffer_state['radius'] = radius
        self.buffer_state['f_old'] = f

        self.counter.incrementStepCount()

    def subRoutineStep(self, grad, radius):
        mode = self.params['mode']
        if mode == 'exact':
            hess = self.problem.getHessian(self.x)
            self.counter.incrementHessCount()

            p, lmd = self.subroutine.exactSolver(hess, -grad, radius)

            # left = leftMost(hess)
            # print("lmd", lmd)
            # print("left", left)
            # print("equation norm", LA.norm((hess+lmd*np.eye(len(grad))).dot(p) + grad))
            # print("complmentary", lmd*(LA.norm(p) - radius))
            # print("norm(p), radius", LA.norm(p), radius)

        if mode == 'cauchy':
            p, lmd = self.subroutine.cauchySolver(self.x, -grad, radius)

        if mode == 'gltr':
            p, lmd = self.subroutine.gltrSolver(self.x, grad, radius)

        if mode == 'adaNT':
            hess = self.problem.getHessian(self.x)
            self.counter.incrementHessCount()
            p, lmd = self.subroutine.adaNT_solver(hess, -grad, radius)

        return p, lmd

    def subModelReduction(self, p, grad, lmd, radius):
        tmp = -grad.dot(p) - 0.5 * p.dot(self.problem.getHv(self.x, p))

        if tmp <= 0:
            print(self.problem.getHessian(self.x), grad)
            print(tmp, (grad + self.problem.getHv(self.x, p) + lmd*p).dot(p) + 0.5 * radius**2*lmd)
            hess = self.problem.getHessian(self.x)
            left = leftMost(hess)
            print("lmd", lmd)
            print("left", left)
            print("equation norm", LA.norm((hess+lmd*np.eye(len(grad))).dot(p) + grad))
            print("complmentary", lmd*(LA.norm(p) - radius))
            print("norm(p), radius", LA.norm(p), radius)
        assert tmp >= 0, 'trust region submodel does not reduce.!!'

        return tmp

    def summary(self):
        return
