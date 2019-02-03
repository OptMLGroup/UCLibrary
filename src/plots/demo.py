############

#   @File name: demo.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-10-05 14:26:04

#   @Last modified by:  Xi He
#   @Last Modified time:    2019-02-03 17:35:12

#   @Description:
#   @Example:

############

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import axes3d

import numpy as np

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

class Demo(object):
    def __init__(self, problem, draw_trace = None):
        self.problem = problem
        self.draw_trace = draw_trace
        assert draw_trace in ['2d', '3d', None], 'draw_trace type error!!'
        self.fig_performance, (self.ax1, self.ax2) = plt.subplots(1, 2, sharex=True, figsize=(12, 8))

        if self.draw_trace:
            self.fig_trace, self.ax = plt.subplots(figsize=(12, 8))
        self.drawContourBase()

    def doDrawTrace(self):
        return True if self.draw_trace else False

    def drawContourBase(self):

        M = int(config['DEMO']['SAMPLE_POINTS'])
        x1 = float(config['DEMO']['CONTOUR_XRANGE_1'])
        x2 = float(config['DEMO']['CONTOUR_XRANGE_2'])
        y1 = float(config['DEMO']['CONTOUR_YRANGE_1'])
        y2 = float(config['DEMO']['CONTOUR_YRANGE_2'])


        if not self.draw_trace:
            return

        assert self.problem.getSize() == 2, 'contour demo not available for dim >= 2. !!'

        matplotlib.rcParams['xtick.direction'] = 'out'
        matplotlib.rcParams['ytick.direction'] = 'out'

        x = np.linspace(x1, x2, M)
        y = np.linspace(y1, y2, M)
        X, Y = np.meshgrid(x, y)

        dx, dy = X.shape

        Z = np.zeros_like(X)
        for i in range(dx):
            for j in range(dy):
                Z[i][j] = self.problem.getFn(np.array([X[i][j], Y[i][j]]))

        if self.draw_trace == '2d':
            # b = np.min(Z)
            min_val = float(config['DEMO']['FUNC_RANGE_LOWER'])
            max_val = float(config['DEMO']['FUNC_RANGE_HIGHER'])
            level_range = int(config['DEMO']['LEVELS_OF_COLORS'])
            levels = np.linspace(min_val, max_val, level_range)
            contour = self.ax.contour(X, Y, Z, levels, colors='k')
            self.ax.clabel(contour, colors = 'k', fmt = '%2.1f', fontsize=10)
            contour_filled = self.ax.contourf(X, Y, Z, levels, cmap=cm.Pastel1)
            self.fig_trace.colorbar(contour_filled)
            self.ax.set_title(self.problem.getName())
            self.ax.set_xlabel('x1')
            self.ax.set_ylabel('x2')
        elif self.draw_trace == '3d':
            pass

    def addContourTrace(self, optim):

        if not self.draw_trace:
            return

        assert self.problem.getSize() == 2, 'contour demo not available for dim >= 2. !!'
        if self.draw_trace == '2d':
            tmp = np.concatenate(optim.trace,axis=0).T
            self.ax.plot(tmp[0], tmp[1], linewidth=3., label=optim.getName()+','+str(len(optim.trace))+' iters')
            self.ax.scatter(tmp[0], tmp[1])
            plt.legend(loc='best')
        elif self.draw_trace == '3d':
            pass

    def drawPerformancePlot(self, optim, log_plot=False):

        res = np.array(optim.state).T

        if not log_plot:
            self.ax2.plot(res[0].astype(int), res[2], label=optim.getName()+','+str(len(res[0]))+' iters')
        else:
            self.ax2.semilogy(res[0].astype(int), res[2], label=optim.getName()+','+str(len(res[0]))+' iters')
            # self.ax2.semilogy(res[0].astype(int), res[2])
        self.ax2.set_ylabel(r'$||\nabla f(x)||$')
        # self.ax2.legend(loc='best')
        self.ax2.set_xlabel(r'iters')

        self.ax1.plot(res[0].astype(int), res[1], label=optim.getName()+','+str(len(res[0]))+' iters')
        self.ax1.legend(loc='best')
        self.ax1.set_ylabel(r'$f(x)$')
        self.ax1.set_xlabel(r'iters')

        plt.legend(loc='best')
        self.fig_performance.suptitle(self.problem.getName())

    def showPlot(self):
        plt.subplots_adjust(top=0.92, bottom=0.1, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
        plt.show()
        self.closePlot()

    def savePlot(self):
        save_format = config['DEMO']['SAVE_FORMAT']
        self.fig_performance.savefig(self.problem.getName()+'-performance.'+save_format, bbox_inches='tight', format=save_format, dpi=1000)
        if self.draw_trace:
            self.fig_trace.savefig(self.problem.getName()+'-trace.'+save_format, bbox_inches='tight', format=save_format, dpi=1000)
        self.closePlot()

    def closePlot(self):
        plt.close()
