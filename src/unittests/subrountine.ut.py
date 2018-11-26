############

#   @File name: subrountine.ut.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-10-05 12:01:13

#   @Last modified by:  Xi He
#   @Last Modified time:    2018-11-20 14:22:35

#   @Description:
#   @Example:

############
import unittest
from unittest import skip

import sys
import os
sys.path.append(os.getcwd())

from problems.cutest_pb import CutestProblem
import numpy as np

from utils.counter import Counter
from utils.subroutine import SubRoutine

from utils.matrix_wrapper import leftRightMost, leftMost
import numpy.linalg as LA
from scipy import linalg as SLA

class SubRoutineTest(unittest.TestCase):

    def setUp(self):
        name = self.shortDescription()
        if name == 'tridiagnoal':
            self.hessian = np.array([[10,2,0,0],[2,10,4,0],[0,4,7,5],[0,0,5,15]],dtype=float)
            self.grad = np.array([2., 2, 3, 4])

            self.hessian = np.array([[226.69618758, 199.1321379 ],[199.1321379 , 200.]])
            self.grad = np.array([-16.15239365, -13.21407293])

            self.hessian = np.array([[1330.,  480.],[480.,  200.]])
            self.grad = np.array([-215.6,  -88.])

        if name == 'general':
            self.hessian = np.array([[10., 2, 3, 4],[2, 10, 4, 2],[3, 4, 7, 5],[4, 2, 5,15]],dtype=float) # easy case
            self.grad = np.array([2., 2, 3, 4])

            self.hessian = np.array([[-10., 0, 0, 0],[0, 10, 0, 0],[0, 0, 7, 0],[0, 0, 0, 2]],dtype=float) # hard case
            self.grad = np.array([0., 2, 3, 4])

            from scipy.stats import ortho_group
            n = 10
            U = ortho_group.rvs(n)
            self.hessian = U.dot(np.diag(np.array([-5] + list(np.random.uniform(-4.5, 4.5, n-1))))).dot(U.T)
            self.grad = U.dot(np.array([0.]+list(np.random.uniform(-1, 1, n-1))))
            # self.hessian = np.diag(np.array([-5] + list(np.random.uniform(-4, 5, n-1))))
            # self.grad = np.array([0.]+list(np.random.uniform(-1, 1, n-1)))

            # print(self.hessian, self.grad)

            # self.hessian = np.diag(np.array(range(100))-5.)
            # self.grad = np.array([0.]+list(range(99)))

            # self.hessian = np.array([[1., 0, 4],[0, 2, 0],[4, 0, 3]],dtype=float) # easy case
            # self.grad = np.array([5., 0, 4])

            self.hessian = np.array([[1., 0, 4],[0, 2, 0],[4, 0, 3]],dtype=float) # hard case
            self.grad = np.array([0., 2, 0])

            # self.hessian = np.array([[-10., 0, 0, 0],[0, 1, 1, 0],[0, 1, 1, 0],[0, 0, 0, 1]], dtype=float) # hard case
            # self.grad = np.array([0., 0, 0, 4])

            # self.hessian = np.array([[226.69618758, 199.1321379 ],[199.1321379 , 200.]])
            # self.grad = np.array([-16.15239365, -13.21407293])

            self.hessian = np.array([[1330.,  480.],[480.,  200.]])
            self.grad = np.array([-215.6,  -88.])

        self.lmd_min = leftMost(self.hessian)[0]
        self.size = self.grad.size

        self.problem = CutestProblem(self.size)
        self.counter = Counter()
        self.subroutine = SubRoutine(self.problem, self.counter)
        self.eps = 1e-7

        self.M = 0.5
        self.setUpDependencies()

    def setUpDependencies(self):
        if self.lmd_min > 0:
            self.diag = self.hessian.diagonal(0)
            self.off_diag = self.hessian.diagonal(-1)

            self.l, self.d = self.subroutine.triDecomp(self.diag, self.off_diag)

            self.D = np.diag(self.d)
            self.U = np.diag(self.l, 1) + np.eye(self.size)
            self.Lc = LA.cholesky(self.hessian)

    def testTridiagnoalDecomposition(self):
        '''tridiagnoal'''
        if self.lmd_min > 0:
            self.assertTrue(np.isclose(LA.norm(np.matmul(self.Lc, self.Lc.T)- self.hessian), 0.))
            self.assertTrue(np.isclose(LA.norm(np.matmul(self.U, np.matmul(self.D, self.U.T))-self.hessian), 0.))


    def testTridiagnoalSolver(self):
        '''tridiagnoal'''
        if self.lmd_min > 0:
            w = self.subroutine.upDiagonalSolver(self.l, self.grad)
            s1 = self.subroutine.offDiagonalSolver(self.l, np.multiply(w, 1./self.d))

            w = SLA.solve_triangular(self.Lc, self.grad, lower=True)
            s2 = SLA.solve_triangular(self.Lc.T, w, lower=False)

            self.assertTrue(np.isclose(LA.norm(s1-s2), 0.))

            s = s1

            M = np.matmul(LA.inv(np.matmul(self.U, self.D**0.5)), self.Lc)
            self.assertTrue(np.isclose(LA.norm(np.matmul(M, M.T) - np.eye(self.size)), 0.))

            y1 = SLA.solve_triangular(self.Lc, -s, lower=True)

            y2 = self.subroutine.upDiagonalSolver(self.l, -s)

            self.assertTrue(np.isclose(LA.norm(self.Lc.dot(y1)+s) + LA.norm(self.U.dot(y2)+s), 0.))
            self.assertTrue(np.isclose(LA.norm(y1) - np.sqrt((1./self.d * y2 * y2).sum()), 0.))

    def testCubicSolver(self):
        '''general'''
        self.subroutine.setMode('cubic')
        s, lmd = self.subroutine.exact_subproblem_solver(self.hessian, -self.grad, self.M)

        self.assertGreaterEqual(self.lmd_min+lmd, -self.eps)
        if abs(self.lmd_min + lmd) >= self.eps:
            self.assertGreaterEqual(self.eps, LA.norm((self.hessian+lmd*np.eye(self.size)).dot(s) + self.grad))
        self.assertGreaterEqual(self.eps, abs(lmd - LA.norm(s) * self.M))

    @unittest.skip("skipping adaNT solver")
    def testAdaNTCubicSolver(self):
        '''general'''
        self.subroutine.setMode('cubic')
        s, lmd = self.subroutine.adaNT_solver(self.hessian, -self.grad, self.M)

        self.assertGreaterEqual(self.lmd_min+lmd, -self.eps)
        if abs(self.lmd_min + lmd) >= self.eps:
            self.assertGreaterEqual(self.eps, LA.norm((self.hessian+lmd*np.eye(self.size)).dot(s) + self.grad))
        self.assertGreaterEqual(self.eps, abs(lmd - LA.norm(s) * self.M))

    @unittest.skip("skipping adaNT solver")
    def testAdaNTTrustregionSolver(self):
        '''general'''
        self.subroutine.setMode('tr')
        s, lmd = self.subroutine.adaNT_solver(self.hessian, -self.grad, self.M)

        self.assertGreaterEqual(self.lmd_min+lmd, -self.eps)
        if abs(lmd) >= self.eps:
            self.assertGreaterEqual(self.eps, LA.norm((self.hessian+lmd*np.eye(self.size)).dot(s) + self.grad))
        self.assertGreaterEqual(self.eps, lmd*(LA.norm(s) - self.M))

    def testTrustregionSolver(self):
        '''general'''
        self.subroutine.setMode('tr')
        s, lmd = self.subroutine.exact_subproblem_solver(self.hessian, -self.grad, self.M)

        # print("lmd", lmd, " lmd_min", self.lmd_min)
        # print("norm, equation", LA.norm((self.hessian+lmd*np.eye(self.size)).dot(s) + self.grad))
        # print("complementary condition", lmd*(LA.norm(s) - self.M))
        # print("radius", LA.norm(s), "  M", self.M)

        self.assertGreaterEqual(self.lmd_min+lmd, -self.eps)
        # if abs(lmd) >= self.eps:
        self.assertGreaterEqual(self.eps, LA.norm((self.hessian+lmd*np.eye(self.size)).dot(s) + self.grad))
        self.assertGreaterEqual(self.eps, abs(lmd*(LA.norm(s) - self.M)))

    def testTrustregionTridiagnoalSolver(self):
        '''tridiagnoal'''
        self.subroutine.setMode('tr')
        s, lmd = self.subroutine.exact_tridiagonal_solver(self.hessian, -self.grad, self.M)

        # print("lmd", lmd, " lmd_min", self.lmd_min)
        # print("norm, equation", LA.norm((self.hessian+lmd*np.eye(self.size)).dot(s) + self.grad))
        # print(LA.norm(LA.inv(self.hessian+lmd*np.eye(self.size)).dot(self.grad)))
        # print("complementary condition", lmd*(LA.norm(s) - self.M))
        # print("radius", LA.norm(s), "  M", self.M)

        self.assertGreaterEqual(self.lmd_min+lmd, -self.eps)
        # if abs(lmd) >= self.eps:
        self.assertGreaterEqual(self.eps, LA.norm((self.hessian+lmd*np.eye(self.size)).dot(s) + self.grad))
        self.assertGreaterEqual(self.eps, lmd*(LA.norm(s) - self.M))

    def testCubicTridiagnoalSolver(self):
        '''tridiagnoal'''
        self.subroutine.setMode('cubic')
        s, lmd = self.subroutine.exact_tridiagonal_solver(self.hessian, -self.grad, self.M)

        # print("lmd", lmd, " lmd_min", self.lmd_min)
        # print("norm, equation", LA.norm((self.hessian+lmd*np.eye(self.size)).dot(s) + self.grad))
        # print("complementary condition", lmd - LA.norm(s) * self.M)

        self.assertGreaterEqual(self.lmd_min+lmd, -self.eps)
        self.assertGreaterEqual(self.eps, LA.norm((self.hessian+lmd*np.eye(self.size)).dot(s) + self.grad))
        self.assertGreaterEqual(self.eps, abs(lmd - LA.norm(s) * self.M))

if __name__ == '__main__':
    unittest.main()
