############

#   @File name: optimizer.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-08-18 00:28:52

# @Last modified by:   Heerye
# @Last modified time: 2018-08-18T07:54:52-04:00

#   @Description:
#   @Example:

############

from collections import defaultdict
import numpy as np
from numpy import linalg as LA
from utils.counter import Counter
from utils.linesearch import LineSearch
from utils.subroutine import SubRoutine
from copy import deepcopy

required = object()

class Optimizer(object):
    def __init__(self, problem, params):
        self.problem  = problem
        self.params = params
        self.size = self.problem.getSize()
        self.x = self.problem.getInitialPoint()
        self.x0_status = self.problem.getInitialStatus()

        self.counter = Counter()
        self.linesearcher = LineSearch(problem, self.counter)
        self.subroutine = SubRoutine(problem, self.counter)

        self.state = []
        self.trace = []
        self.buffer_state = defaultdict(dict)

    def step(self):
        raise NotImplementedError

    def stepLog(self, save, trace):
        if save:
            step_state = [self.counter.step_counter, self.problem.getFn(self.x), LA.norm(self.problem.getGrad(self.x))]
            self.state.append(step_state)
        if trace:
            trace_x = deepcopy(self.x).reshape(1, 2)
            self.trace.append(trace_x)

    def getOptimName(self, base, features):
        for feature in features:
            if self.params[feature]:
                if isinstance(self.params[feature], bool):
                    if self.params[feature] is True:
                        base.append(feature)
                else:
                        base.append(str(self.params[feature]))
                        # base.append(feature)
        return '-'.join(base)

    def terminationCondition(self, mode = 'first_order', tol = 1e-6):
        if mode == 'zeroth_order':
            return LA.norm(self.problem.getFn(self.x)) <= tol * max(1, self.x0_status[0])
        elif mode == 'first_order':
            return LA.norm(self.problem.getGrad(self.x)) <= tol * max(1, self.x0_status[1])
        elif mode == 'second_order':
            raise NotImplementedError('convergence condition not implemented: {}'.format(mode))
        else:
            raise NotImplementedError('convergence condition not implemented: {}'.format(mode))


