############

#   @File name: linesearch.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-09-24 09:09:38

#   @Last modified by:  Xi He
#   @Last Modified time:    2018-10-21 19:36:35

#   @Description:
#   @Example:

############
from .counter import Counter

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

import logging

class LineSearch(object):
    def __init__(self, problem, counter):
        self.problem = problem
        self.counter = counter

    def lineSearch(self, x, p, grad, mode, lr=None):
        if mode == 'strong_wolfe':
            return self.lineSearch__StrongWolfe(x, p, grad, lr)
        elif mode == 'amijo':
            return self.lineSearch__Amijo(x, p, grad, lr)
        else:
            raise NotImplementedError('Line search mode not implemented! {}'.format(mode))

    def lineSearch__Amijo(self, x, p, grad, lr):
        if not lr:
            lr = float(config['LINESEARCH']['AMIJO_INITIAL'])
        c = float(config['LINESEARCH']['AMIJO_MULTIPLER_FUNC'])
        rho = float(config['LINESEARCH']['AMIJO_DECAY'])
        max_iters = int(config['LINESEARCH']['AMIJO_MAXITER'])

        f0 = self.problem.getFn(x)
        self.counter.incrementFnCount()

        i = 1
        while True:
            fx = self.problem.getFn(x + lr * p)
            self.counter.incrementFnCount()

            if fx <= f0 + c * lr * grad.dot(p.T) or i  > max_iters:
                return lr
            else:
                lr *= rho
            i += 1


    def lineSearch__StrongWolfe(self, x, p, grad, lr):
        if not lr:
            lr = float(config['LINESEARCH']['STRONG_WOLFE_INITIAL'])
        c1 = float(config['LINESEARCH']['STRONG_WOLFE_MULTIPLER_FUNC'])
        c2 = float(config['LINESEARCH']['STRONG_WOLFE_MULTIPLER_GRAD'])
        beta = float(config['LINESEARCH']['STRONG_WOLFE_DECAY'])
        lr_eps = float(config['LINESEARCH']['STRONG_WOLFE_TOL'])

        alphap = 0.
        alphax = 1e-12

        f0 = self.problem.getFn(x)
        self.counter.incrementFnCount()

        g0 = grad
        fp = f0
        gp = g0

        i =1
        while True:

            fx = self.problem.getFn(x + alphax * p)
            self.counter.incrementFnCount()

            if fx > f0 + c1 * alphax * g0.dot(p.T) or (fx >= fp and i > 1):
                return self.zoom(x, p, alphap, alphax, f0, g0)

            gx = self.problem.getGrad(x + alphax * p)
            self.counter.incrementGradCount()

            if abs(gx.dot(p.T)) <= -c2*g0.dot(p.T):
                return alphax

            if gx.dot(p.T) >=0:
                return self.zoom(x, p, alphax, alphap, f0, g0)

            i = i + 1
            alphap = alphax
            fp = fx
            gp = fx

            alphax += (lr - alphax) * beta
            if abs(lr - alphax) < lr_eps:
                return alphax

    def zoom(self, x, p, alphal, alphah, f0, g0):
        c1 = float(config['LINESEARCH']['STRONG_WOLFE_MULTIPLER_FUNC'])
        c2 = float(config['LINESEARCH']['STRONG_WOLFE_MULTIPLER_GRAD'])
        lr_eps = float(config['LINESEARCH']['STRONG_WOLFE_TOL'])

        while True:
            alphax = (alphal + alphah) * 0.5

            fx = self.problem.getFn(x + alphax * p)
            self.counter.incrementFnCount()

            fl = self.problem.getFn(x + alphal * p)
            self.counter.incrementFnCount()

            if fx > f0 + c1 * alphax * g0.dot(p.T) or fx >= fl:
                alphah = alphax
            else:
                gx = self.problem.getGrad(x + alphax * p)
                self.counter.incrementGradCount()

                if abs(gx.dot(p.T)) <= -c2*g0.dot(p.T):
                    return alphax
                if gx.dot(p.T) * (alphah -alphal) >= 0:
                    alphah = alphal
                alphal = alphax

            if abs(alphal -  alphah) < lr_eps:
                return (alphal + alphah) * 0.5
