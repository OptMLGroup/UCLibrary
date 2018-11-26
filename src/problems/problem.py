############

#   @File name: problem.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-08-17 23:57:33

#   @Last modified by:  Xi He
#   @Last Modified time:    2018-09-18 17:13:29

#   @Description:
#   @Example:

############

class Problem(object):
    def __init__(self, func_name=None):
        self.name = func_name
        self.fn = None
        self.gn = None
        self.Hn = None
        self.Hv = None

    def getInitialPoint(self):
        return None

    def getModel(self):
        return None

    def getName(self):
        return None

    def getFn(self, x):
        return None

    def getGrad(self, x):
        return None

    def getHessian(self, x):
        return None

    def getHv(self, x, v):
        return None