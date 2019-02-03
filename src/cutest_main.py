############

#   @File name: cutest_main.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-07-01 12:31:00

#   @Last modified by:  Xi He
#   @Last Modified time:    2019-02-03 12:15:16

#   @Description:
#   @Example:

############

from problems.cutest_pb import CutestProblem
import numpy as np
import optimizers
from plots.demo import Demo

import matplotlib
import matplotlib.pyplot as plt

import logging
import logging.config
import os
import json

def run(problem, optim, demo):
    # try:
        print(optim.getName())
        for _ in range(300):
            optim.step(save=True, trace=demo.doDrawTrace())
            if optim.terminationCondition(mode='first_order', tol =1e-6):
                break

        demo.addContourTrace(optim)
        optim.counter.printCounter(optim.counter)

        demo.drawPerformancePlot(optim, log_plot=True)
    # except:
        # print('RUN failed: %s'%optim.getName())

def main(func_name='ROSENBR'):

    if os.path.isfile('cutest_logging.log'):
        os.remove('cutest_logging.log')

    with open('logging_configuration.json', 'r') as logging_config:
        config_dict = json.load(logging_config)

    logging.config.dictConfig(config_dict)
    logging.info('Started')

    # draw_trace = None # alternatives: '2d', '3d'
    draw_trace = '2d' # alternatives: '2d', '3d'

    problem = CutestProblem(func_name)

    if problem.getSize() > 2 and draw_trace: draw_trace = None

    demo = Demo(problem, draw_trace = draw_trace)

    for linesearch in ['amijo', 'strong_wolfe']:
        problem.setInitialPoint()
        optim = optimizers.GD(problem, lr=False, momentum=0.99, linesearch=linesearch)
        run(problem, optim, demo)

    for mode in ['exact', 'cg']:
        problem.setInitialPoint()
        optim = optimizers.Newton(problem, lr=False, damping='min_eig', mode=mode, linesearch='strong_wolfe')
        run(problem, optim, demo)

    for mode in ['exact', 'krylov', 'cauchy', 'adaNT']:
        problem.setInitialPoint()
        optim = optimizers.Cubic(problem, lr=False, mode=mode, adaptive=True)
        run(problem, optim, demo)

    for mode in ['exact', 'cauchy', 'adaNT']:
        problem.setInitialPoint()
        optim = optimizers.TrustRegion(problem, mode=mode)
        run(problem, optim, demo)

    demo.savePlot()
    # demo.showPlot()

    logging.info('Finished')

if __name__ == '__main__':
    func_name = 'ROSENBR'  # dim = 2
    # func_name = 'AKIVA'  # dim = 2
    # func_name = 'ARGTRIGLS'  # dim = 10
    # func_name = '3PK'  # dim = 30
    # func_name = 'BQPGASIM' # dim = 50

    main(func_name)

    # with open('uc_small.txt', 'r') as f:
    #     for line in f:
    #         name, size = line.split('@@')
    #         print(name, size)
    #         main(name)