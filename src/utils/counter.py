############

#   @File name: counter.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-09-24 08:41:21

#   @Last modified by:  Xi He
#   @Last Modified time:    2018-10-05 17:19:14

#   @Description:
#   @Example:

############

class Counter(object):
    def __init__(self):
        self.fn_counter = 0
        self.grad_counter = 0
        self.hess_counter = 0
        self.hessv_counter = 0
        self.step_counter = 0
        self.subrountine_counter = 0
        self.tri_decomp_counter = 0
        self.eigenvalue_counter = 0
        self.eigenvector_counter = 0
        self.cholesky_decomp_counter = 0
        self.cholesky_linear_solver_counter = 0

    def incrementFnCount(self):
        self.fn_counter += 1

    def incrementGradCount(self):
        self.grad_counter += 1

    def incrementHessCount(self):
        self.hess_counter += 1

    def incrementHessVCount(self):
        self.hessv_counter += 1

    def incrementStepCount(self):
        self.step_counter += 1

    def incrementSubrountineCount(self):
        self.subrountine_counter += 1

    def incrementTriDecompCount(self):
        self.tri_decomp_counter += 1

    def incrementEigenValueCount(self):
        self.eigenvalue_counter += 1

    def incrementEigenVectorCount(self):
        self.eigenvector_counter += 1

    def incrementCholeskyDecompCount(self):
        self.cholesky_decomp_counter += 1

    def incrementCholeskyLinearSolverCount(self):
        self.cholesky_linear_solver_counter += 1

    def resetCounter(self):
        self.fn_counter = 0
        self.grad_counter = 0
        self.hess_counter = 0
        self.hessv_counter = 0
        self.step_counter = 0
        self.subrountine_counter = 0
        self.tri_decomp_counter = 0
        self.eigenvalue_counter = 0
        self.eigenvector_counter = 0
        self.cholesky_decomp_counter = 0
        self.cholesky_linear_solver_counter = 0

    @staticmethod
    def printCounter(cls):
        for k, v in vars(cls).items():
            if v:
                print(k, v)


