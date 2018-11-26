############

#   @File name: gd.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-08-18 00:29:39

# @Last modified by:   Heerye
# @Last modified time: 2018-08-18T07:55:00-04:00

#   @Description:
#   @Example:

############

from .optimizer import Optimizer, required
import numpy as np

class GD(Optimizer):
    def __init__(self, problem, lr=required, momentum=0, nesterov=False, dynamic=False, restart=False, linesearch='strong_wolfe'):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid step-size: {}".format(lr))
        if momentum < 0.0 and momentum not in [True, False]:
            raise ValueError("Invalid momentum: {}".format(momentum))
        if not isinstance(nesterov, bool):
            raise ValueError("Invalid nesterov: {}".format(nesterov))
        if not isinstance(dynamic, bool):
            raise ValueError("Invalid dynamic: {}".format(dynamic))
        if not isinstance(restart, bool):
            raise ValueError("Invalid restart: {}".format(restart))
        params = dict(lr=lr, momentum=momentum, nesterov=nesterov, dynamic=dynamic, restart=restart, linesearch=linesearch)
        super(GD, self).__init__(problem, params)

    def getName(self):
        base = ['GD']
        features = ['momentum', 'nesterov', 'dynamic', 'restart', 'linesearch']
        return self.getOptimName(base, features)

    def step(self, save=False, trace=False):
        lr = self.params['lr']
        momentum = self.params['momentum']
        nesterov = self.params['nesterov']
        dynamic = self.params['dynamic']
        restart = self.params['restart']
        linesearch = self.params['linesearch']

        grad = self.problem.getGrad(self.x)
        self.counter.incrementGradCount()

        # TODO: Here double check momentum algorithm, make sure it works
        # if momentum != 0:
        #     if 'momentum_buffer' not in self.state:
        #         self.state['momentum_buffer'] = np.zeros(x.shape)
        #         buf = self.state['momentum_buffer']
        #         buf = buf * momentum + grad
        #     else:
        #         buf = self.state['momentum_buffer']
        #         buf = buf * momentum + grad

        #     if nesterov:
        #         p = grad + momentum * buf
        #     else:
        #         p = buf
        p = -grad

        if linesearch is not None:
            lr = self.linesearcher.lineSearch(self.x, p, grad, linesearch)

        self.stepLog(save, trace)

        self.x = self.x + lr * p
        self.counter.incrementStepCount()

    def summary(self):
        return
