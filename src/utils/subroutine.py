 ############

#   @File name: subroutine.py
#   @Author:    Xi He
#   @Email: xih314@lehigh.edu, heeryerate@gmail.com

#   @Create date:   2018-09-24 09:09:53

#   @Last modified by:  Xi He
#   @Last Modified time:    2019-02-03 17:28:41

#   @Description:
#   @Example:

############

from .counter import Counter
from numpy import linalg as LA
from scipy import linalg as SLA
from utils.matrix_wrapper import rightMost, leftMost, leftRightMost, matrixDamping, isPDMatrix
import numpy as np

import configparser
config = configparser.ConfigParser()
config.read('config.ini')

class SubRoutine(object):

        def __init__(self, problem, counter):
            self.problem = problem
            self.counter = counter

        def setMode(self, mode):
            self.mode = mode

        def cgSolver(self, x, b):
            max_iters = 100
            tol = 1e-12

            cg_x = np.zeros(b.shape)
            r = b - self.problem.getHv(x, cg_x)
            self.counter.incrementHessVCount()

            p = r
            rs_old = r.dot(r)

            i = 1
            while True:
                Ap = self.problem.getHv(x, p)
                self.counter.incrementHessVCount()

                pAp = p.dot(Ap)

                if pAp <= 0:
                    return b

                alpha = rs_old / pAp

                cg_x = cg_x+ alpha * p
                r = r - alpha * Ap

                rs_new = r.dot(r)

                p = r + (rs_new/rs_old) * p
                rs_old = rs_new

                if i > max_iters or rs_old < tol:
                    break

                i += 1

            return cg_x

        def cauchySolver(self, x, b, tau):
            Ab = self.problem.getHv(x, b)
            self.counter.incrementHessVCount()

            bAb = b.dot(Ab)
            b_norm = LA.norm(b)

            if self.mode == 'cubic':
                a0 = tau * b_norm**3
                a1 = bAb
                a2 = -b_norm**2
                _, alpha = self.findQuadRoot(a0, a1, a2)
            elif self.mode == 'tr':
                if bAb <= 0:
                    tmp = 1.
                else:
                    tmp = min(b_norm ** 3 / (bAb*tau), 1.)
                alpha = tmp * tau / b_norm
            return alpha * b, None

        def dogLegSolver(self, x, b, tau):
            return

        def steihaugSolver(self, x, b, tau):
            return

        def exactSolver(self, A, b, tau=None):

            if self.mode == 'newton':
                self.counter.incrementSubrountineCount()
                return LA.solve(A, b)

            else:
                s, lmd_c = self.exact_subproblem_solver(A, b, tau)
                return s, lmd_c

        def gltrSolver(self, x, b, tau):

            krylov_tol = float(config['GLOBAL']['KRYLOV_TOL'])
            b_norm = LA.norm(b)
            s = np.zeros_like(b)
            dim = b.size

            if b_norm == 0: # escape along the direction of the leftmost eigenvector as far as tr_radius permits
                print ('zero gradient encountered')
                A = self.problem.getHessian(x)
                self.counter.incrementSubrountineCount()

                s, lmd_k = self.exact_subproblem_solver(A, -b, tau)

                return s, lmd_k

            g = b
            p = -g

            gamma = b_norm
            T = np.zeros((1, 1))
            alphas = []
            betas = []
            interior_flag = True
            k = 0

            while True:
                Ap = self.problem.getHv(x, p)
                self.counter.incrementHessVCount()

                pAp = np.dot(p, Ap)
                alpha = np.dot(g, g) / pAp

                alphas.append(alpha)

                ###Lanczos Step 1: Build up subspace
                # a) Create g_lanczos = gamma*e_1

                e_1 = np.zeros(k + 1)
                e_1[0] = 1.0
                g_lanczos = gamma * e_1
                # b) Create T for Lanczos Model
                T_new = np.zeros((k + 1, k + 1))
                if k == 0:
                    T[k, k] = 1. / alpha
                    T_new[0:k,0:k]=T
                else:
                    T_new[0:k,0:k]=T
                    T_new[k, k] = 1. / alphas[k] + beta/ alphas[k - 1]
                    T_new[k - 1, k] = beta**0.5 / abs(alphas[k - 1])
                    T_new[k, k - 1] = beta**0.5 / abs(alphas[k - 1])
                    T = T_new

                if (interior_flag == True and alpha < 0) or LA.norm(s + alpha * p) >= tau:
                    interior_flag = False

                if interior_flag == True:
                    s += alpha * p
                else:
                    ###Lanczos Step 2: solve problem in subspace
                    h, lmd_k = self.exact_tridiagonal_solver(T, g_lanczos, tau)

                g_next = g + alpha * Ap

                # test for convergence
                e_k = np.zeros(k + 1)
                e_k[k] = 1.0

                if interior_flag == True and LA.norm(g_next) < min(LA.norm(b)**0.5 * LA.norm(b),krylov_tol) :
                    break
                if interior_flag == False and LA.norm(g_next) * abs(np.dot(h, e_k)) < min(LA.norm(b)**0.5 * LA.norm(b),krylov_tol):
                    break

                if k==dim-1:
                    # hess = self.problem.getHessian(x)
                    # print(T, hess)
                    # print(leftMost(T), leftMost(hess))
                    print ('Krylov dimensionality reach full space! Breaking out..')
                    break

                beta= np.dot(g_next, g_next) / np.dot(g, g)
                betas.append(beta)
                p = -g_next + beta* p
                g = g_next
                k = k + 1

            if interior_flag == False:
                # print('k', k)
                #### Recover Q by building up the lanczos space, TBD: keep storable Qs in memory
                n = np.size(b)
                Q1 = np.zeros((n, k + 1))

                g = b
                p = -g
                for j in range(0, k + 1):
                    gn = LA.norm(g)
                    if j == 0:
                        sigma = 1
                    else:
                        sigma = -np.sign(alphas[j - 1]) * sigma
                    Q1[:, j] = sigma * g / gn

                    if j != k:
                        Ap = self.problem.getHv(x, p)
                        self.counter.incrementHessVCount()

                        g = g + alphas[j] * Ap
                        p = -g + betas[j] * p

                # compute final step in R^n
                s = np.dot(Q1, np.transpose(h))

                # print(Q1.dot(self.problem.getHessian(x)).dot(Q1), T, 'xxxxxx')

            return s, 0

        def lanczosSolver(self, x, b, tau):

            krylov_tol = 1e-6
            keep_Q = True

            y = b
            b_norm = LA.norm(b)
            gamma_k_plus = b_norm
            delta = []
            gamma = []

            dim = b.size
            if keep_Q:
                q_list = []

            k = 0
            T = np.zeros((1, 1))


            while True:
                if gamma_k_plus == 0:
                    A = self.problem.getHessian(x)
                    self.counter.incrementHessCount()
                    s, lmd_k = self.exact_subproblem_solver(A, b, tau)

                #a ) create g
                e1 = np.zeros(k+1)
                e1[0] = 1.0
                g_lanczos = b_norm * e1
                #b) generate H
                gamma_k = gamma_k_plus
                gamma.append(gamma_k)

                if not k == 0:
                    q_old = q
                q = y/gamma_k

                if keep_Q:
                    q_list.append(q)

                Aq = self.problem.getHv(x, q)
                self.counter.incrementHessVCount()

                delta_k = q.dot(Aq)
                delta.append(delta_k)

                T_new = np.zeros((k+1, k+1))

                if k == 0:
                    T[k, k] = delta_k
                    y = Aq - delta_k * q
                else:
                    T_new[0:k, 0:k] = T
                    T_new[k, k] = delta_k
                    T_new[k, k-1] = gamma_k
                    T_new[k, k-1] = gamma_k
                    T = T_new
                    y = Aq - delta_k*q - gamma_k*q_old

                gamma_k_plus = LA.norm(y)

                if k == dim - 1 or gamma_k_plus == 0:
                    u, lmd_k = self.exact_tridiagonal_solver(T, g_lanczos, tau)
                    e_k = np.zeros(k+1)
                    e_k[k] = 1.0
                    if LA.norm(y) * abs (u.dot(e_k)) < min(krylov_tol, LA.norm(u)/max(1, tau)) * b_norm:
                        break

                if k == dim - 1:
                    print('Krylov dimensionality reach full space!')
                    break

                k = k + 1

            # print(k)
            Q = np.zeros((k+1, dim))
            y = b

            for j in range(0, k+1):
                if keep_Q:
                    Q[j, :] = q_list[j]
                else:
                    if not  j == 0:
                        q_re_old = q_re
                    q_re = y / gamma[j]
                    Q[:, j] = q_re
                    Aq = self.problem.getHv(x, q)
                    self.counter.incrementHessVCount()

                    if  j == 0:
                        y = Aq - delta[j] * q_re
                    elif j != k:
                        y = Aq - delta[j] * q_re - gamma[j] * q_re_old

            s = np.dot(u, Q)
            del Q

            return s, lmd_k

        def exact_tridiagonal_solver(self, T, b, tau):
            '''could be further optimized'''
            lmd_min = SLA.eigh_tridiagonal(T.diagonal(0), T.diagonal(1), eigvals_only = True, select='i', select_range=(0, 0))[0]
            self.counter.incrementEigenValueCount()

            s = b
            max_iters = 200
            eps_exact = 1e-10
            do_order_two_update = False

            gu = max([T[i, i] + np.sum(np.abs(T[i, :])) - np.abs(T[i, i]) for i in range(len(T))])
            lmd_lower = max(0, -lmd_min)
            lmd_upper = max(gu, lmd_lower+eps_exact)

            lmd_c = np.random.uniform(lmd_lower, lmd_upper)

            if self.mode == 'tr':
                is_pd = isPDMatrix(T)
                if is_pd:
                    diag = T.diagonal(0)
                    off_diag = T.diagonal(-1)

                    l_, d_ = self.triDecomp(diag, off_diag)

                    w = self.upDiagonalSolver(l_, b)
                    s = self.offDiagonalSolver(l_, np.multiply(w, 1./d_))
                    sn = LA.norm(s)
                    if sn <= tau:
                        return s, 0

            for iter_ in range(max_iters):

                lmd_plus_in_N = False
                lmd_in_N = False

                B = T + lmd_c * np.eye(self.problem.getSize())

                if lmd_c + lmd_min <= 0:
                    lmd_in_N = True
                else:
                    diag = B.diagonal(0)
                    off_diag = B.diagonal(-1)

                    l_, d_ = self.triDecomp(diag, off_diag)

                    w = self.upDiagonalSolver(l_, b)
                    s = self.offDiagonalSolver(l_, np.multiply(w, 1./d_))
                    sn = LA.norm(s)

                    phi_lmd = self.getPhiLmd(sn, lmd_c, tau)
                    if  abs(phi_lmd) < eps_exact*max(1, tau):
                        break

                    y = self.upDiagonalSolver(l_, -s)
                    yn = np.sqrt((1./d_ * y * y).sum())

                    if phi_lmd < 0 and lmd_c > 0:
                        c_hi = self.orderOneUpdate(sn, yn, lmd_c, tau)
                        lmd_plus = lmd_c + c_hi
                        lmd_c = lmd_plus

                        if abs(c_hi) <= eps_exact*max(1, lmd_upper):
                            return s, lmd_c

                    elif phi_lmd >= 0 and lmd_c > 0:
                        lmd_upper = min(lmd_upper, lmd_c)
                        c_hi = self.orderOneUpdate(sn, yn, lmd_c, tau)
                        if do_order_two_update:
                            s_plus = self.offDiagonalSolver(l_, y)
                            s_plus_n = LA.norm(s_plus)
                            c_hi = max(c_hi, self.orderTwoUpdate(sn, yn, s_plus_n, lmd_c, tau))
                        lmd_plus  = lmd_c + c_hi
                    else:
                        lmd_plus = lmd_c

                    if lmd_plus + lmd_min > 0:
                        lmd_c = lmd_plus
                    else:
                        lmd_plus_in_N = True

                    if lmd_plus <= 0 or lmd_plus_in_N:
                            s, lmd_c, lmd_lower = self.updateInN(T, s, lmd_lower, lmd_upper, lmd_plus, tau)

                if lmd_in_N:
                    s, lmd_c, lmd_lower = self.updateInN(T, s, lmd_lower, lmd_upper, lmd_c, tau)

                sn = LA.norm(s)
                if np.isclose(sn, 0.) or np.isclose(lmd_c, 0.):
                    break

                phi_lmd = self.getPhiLmd(sn, lmd_c, tau)
                if abs(phi_lmd) < eps_exact*max(1, tau):
                    break

            return s, lmd_c

        def exact_subproblem_solver(self, A, b, tau):

            s = b
            eps_exact = float(config['GLOBAL']['TOL'])
            max_iters = int(config['GLOBAL']['MAX_SUBROUTINE_ITERS'])
            do_order_two_update = False # TODO: check why second order update not working

            gl = min([A[i, i] - np.sum(np.abs(A[i, :])) + np.abs(A[i, i]) for i in range(len(A))])
            gu = max([A[i, i] + np.sum(np.abs(A[i, :])) - np.abs(A[i, i]) for i in range(len(A))])

            A_ii_min = min(np.diagonal(A))

            lmd_lower = max(0, -A_ii_min, gl)
            lmd_upper = max(gu, lmd_lower + eps_exact)


            if self.mode == 'tr':
                is_pd = isPDMatrix(A)
                if is_pd:
                    L = LA.cholesky(A) # lower triangular factor
                    self.counter.incrementCholeskyDecompCount()

                    w = SLA.solve_triangular(L, b, lower=True)
                    self.counter.incrementCholeskyLinearSolverCount()

                    s = SLA.solve_triangular(L.T, w, lower=False)
                    self.counter.incrementCholeskyLinearSolverCount()

                    sn = LA.norm(s)

                    if sn <= tau:
                        return s, 0

            lmd_c = np.random.uniform(lmd_lower, lmd_upper)
            # print(lmd_c, lmd_lower, lmd_upper)

            for iter_ in range(max_iters):
                # print(iter_)
            # while True:

                lmd_in_N = False
                lmd_plus_in_N = False

                B = A + lmd_c * np.eye(self.problem.getSize())

                try: # if this succeeds, then lmd_c is in L or G
                    L = LA.cholesky(B) # lower triangular factor
                    self.counter.incrementCholeskyDecompCount()

                    w = SLA.solve_triangular(L, b, lower=True)
                    self.counter.incrementCholeskyLinearSolverCount()

                    s = SLA.solve_triangular(L.T, w, lower=False)
                    self.counter.incrementCholeskyLinearSolverCount()

                    sn = LA.norm(s)

                    phi_lmd = self.getPhiLmd(sn, lmd_c, tau)

                    if abs(phi_lmd) < eps_exact*max(1, tau):
                        break

                    y = SLA.solve_triangular(L, -s, lower=True)
                    self.counter.incrementCholeskyLinearSolverCount()
                    yn = LA.norm(y)

                    if phi_lmd < 0:# and lmd_c > 0: # lmd_c in L
                        # print('L', iter, lmd_lower, lmd_c, lmd_upper)
                        c_hi = self.orderOneUpdate(sn, yn, lmd_c, tau)
                        lmd_plus  = lmd_c + c_hi
                        lmd_c = lmd_plus

                        # if abs(c_hi) < eps_exact*max(1, lmd_upper):
                            # return s, lmd_c

                    elif phi_lmd > 0:# and lmd_c > 0: # lmd_c in G
                        # print('G', iter, lmd_lower, lmd_c, lmd_upper)
                        # lmd_upper = min(lmd_upper, lmd_c)
                        lmd_upper = lmd_c

                        c_hi = self.orderOneUpdate(sn, yn, lmd_c, tau)
                        if do_order_two_update:
                            s_plus = SLA.solve_triangular(L.T, y, lower=False)
                            self.counter.incrementCholeskyLinearSolverCount()
                            s_plus_n = LA.norm(s_plus)
                            c_hi = max(c_hi, self.orderTwoUpdate(sn, yn, s_plus_n, lmd_c, tau))
                        lmd_plus  = lmd_c + c_hi
                        # print(lmd_plus, lmd_c, c_hi)

                        if lmd_plus > 0:
                            try:
                                B_plus  = A + lmd_plus * np.eye(self.problem.getSize())

                                L = LA.cholesky(B_plus)
                                self.counter.incrementCholeskyDecompCount()
                                lmd_c = lmd_plus

                            except LA.LinAlgError:
                                lmd_plus_in_N = True

                        if lmd_plus <= 0 or lmd_plus_in_N:
                            s, lmd_c, lmd_lower = self.updateInN(A, s, lmd_lower, lmd_upper, lmd_plus, tau)

                except LA.LinAlgError: # lmd_c in N
                    lmd_in_N = True

                if lmd_in_N:
                    s, lmd_c, lmd_lower = self.updateInN(A, s, lmd_lower, lmd_upper, lmd_c, tau)

                sn = LA.norm(s)
                if np.isclose(sn, 0.) or np.isclose(lmd_c, 0.):
                    break

                phi_lmd = self.getPhiLmd(sn, lmd_c, tau)
                if abs(phi_lmd) < eps_exact*max(1, tau):
                    break

            return s, lmd_c

        def getPhiLmd(self, sn, lmd_c, tau):
                if self.mode == 'cubic':
                    phi_lmd = 1./sn - tau/lmd_c
                elif self.mode == 'tr':
                    phi_lmd = 1./sn - 1./tau
                else:
                    raise NotImplementedError('exact subsolver mode {} not implemented. '.format(self.mode))

                return phi_lmd

        def orderOneUpdate(self, sn, yn, lmd_c, tau):
            """
            find the largest root of pi_1(delta, -1) = tau(lmd_c + delta, -1)
            see p38, On solving trust-region and other regularized subproblems in optimization
            """
            if self.mode == 'cubic':
                tmp = yn**2/sn**3
                _, c_hi = self.findQuadRoot(tmp, 1./sn+tmp*lmd_c, 1./sn*lmd_c-tau)
            elif self.mode == 'tr':
                tmp = sn**3/yn**2
                c_hi = (1./tau - 1./sn) * tmp
            else:
                raise NotImplementedError('order one update mode {} not implemented!'.format(self.mode))

            return c_hi

        def orderTwoUpdate(self, sn, yn, s_plus_n, lmd_c, tau):
            """
            find the largest root of pi_2(delta, 2) = tau(lmd_c + delta, 2)
            see p38, On solving trust-region and other regularized subproblems in optimization
            """
            if self.mode == 'cubic':
                _, c_hi = self.findQuadRoot(3*s_plus_n**2*tau**2-1, -2*yn**2*tau**2-2*lmd_c, sn**2*tau**2 - lmd_c**2)
            elif self.mode == 'tr':
                _, c_hi = self.findQuadRoot(3*s_plus_n**2, -2*yn**2, sn**2 - tau**2)
            else:
                raise NotImplementedError('order one update mode {} not implemented!'.format(self.mode))

            return c_hi

        def updateInN(self, A, s, lmd_lower, lmd_upper, lmd, tau):
            theta_l = 0.01
            theta_u = 0.9
            eps_tol = 1e-10
            lmd_lower = max(lmd_lower, lmd)

            lmd_l = max((lmd_lower*lmd_upper)**0.5, lmd_lower + theta_l*(lmd_upper - lmd_lower))
            lmd_u = max((lmd_lower*lmd_upper)**0.5, lmd_lower + theta_u*(lmd_upper - lmd_lower))

            # lmd_l = lmd_lower + theta_l * (lmd_upper - lmd_lower)
            # lmd_u = lmd_lower + theta_u * (lmd_upper - lmd_lower)

            # print(lmd_l, lmd_u, lmd_u - lmd_l)
            lmd = np.random.uniform(lmd_l, lmd_u)
            # lmd = (lmd_l * lmd_u) ** 0.5

            # if np.isclose(lmd_lower, lmd_upper):
            if abs(lmd_upper - lmd_lower) < eps_tol * max(1, lmd_upper):
                lmd = lmd_lower

                ew, ev = SLA.eigh(A, eigvals=(0, 0))
                self.counter.incrementEigenValueCount()
                self.counter.incrementEigenVectorCount()

                d = ev[:, 0]

                # g = (A + lmd * np.eye(2)).dot(s)

                # print(g, d.dot(g), d)

                # if ew >= 0:
                    # return s, lmd, lmd_lower

                if self.mode == 'cubic':
                    tao_lower, tao_upper = self.findQuadRoot(1, 2*s.dot(d), s.dot(s) - lmd**2 / tau**2)
                elif self.mode == 'tr':
                    # print('xxxx:', s, d, s.dot(s), tau**2, lmd, lmd_lower)
                    tao_lower, tao_upper = self.findQuadRoot(1, 2*s.dot(d), s.dot(s) - tau ** 2)
                else:
                    raise NotImplementedError('order one update mode {} not implemented!'.format(self.mode))

                s += tao_upper * d

            return s, lmd, lmd_lower

        @staticmethod
        def upDiagonalSolver(off_diag, b):
            bc = np.array(b)
            x = bc
            for i in xrange(len(b)-2, -1, -1):
                x[i] = b[i] - x[i+1] * off_diag[i]

            return x

        @staticmethod
        def offDiagonalSolver(off_diag, b):
            bc = np.array(b)
            x = bc
            for i in xrange(1, len(b)):
                x[i] = b[i] - x[i-1] * off_diag[i-1]
            return x

        @staticmethod
        def triDiagonalSolver(diag, off_diag, b):
            off_diagc, diagc, bc = map(np.array, (off_diag, diag, b)) # copy arrays
            for i in xrange(1, len(b)):
                xmult = off_diagc[i-1]/diagc[i-1]
                diagc[i] = diagc[i] - xmult*off_diagc[i-1]
                bc[i] = bc[i] - xmult*bc[i-1]

            xc = diagc
            xc[-1] = bc[-1]/diagc[-1]

            for il in xrange(len(b)-2, -1, -1):
                xc[il] = (bc[il]-off_diagc[il]*xc[il+1])/diagc[il]

            return xc

        @staticmethod
        def triDecomp(diag, off_diag):
            off_diagc, diagc = map(np.array, (off_diag, diag))

            l, d = off_diagc, diagc
            for i in range(len(d)-2, -1, -1):
                l[i] = off_diagc[i] / diagc[i+1]
                d[i] = diagc[i] - l[i] ** 2 * d[i+1]

            return l, d

        @staticmethod
        def findQuadRoot(a,b,c):
            discriminant = b*b-4*a*c
            assert discriminant >=0, "root failed! {}".format(discriminant)
            sqrt_discriminant = discriminant ** 0.5
            t_lower = (-b - sqrt_discriminant) / (2 * a)
            t_upper = (-b + sqrt_discriminant) / (2 * a)
            return t_lower, t_upper

        @staticmethod
        def getRandVecOnBall(dims):
            x = np.random.standard_normal(dims)
            return x / LA.norm(x)

        def adaNTSolver(self, x, b, sigma_L, sigma_U, lmd_U = float('inf'), lmd_L = 0):
            eps = float(config['IRSNT']['HARDCASE_TOL'])
            lip_hessian = float(config['IRSNT']['LIP_HESSIAN'])
            max_iters = int(config['GLOBAL']['MAX_SUBROUTINE_ITERS'])
            lmd_U = lip_hessian + (sigma_U * LA.norm(b)) ** 0.5

            # print('xxxxx')

            for iter_ in range(max_iters):
                # print(sigma_L, sigma_U, lmd_L, lmd_U)
                lmd_c = np.random.uniform(lmd_L, min(lmd_L + 0.01*(lmd_U+lmd_L), lmd_U))
                d, psd_matrix = self.cgPhaseOne(x, b, sigma_L, sigma_U, lmd_c)

                # print(lmd_L, lmd_c, lmd_U)

                if not psd_matrix:
                    lmd_L = max(lmd_L, lmd_c)
                    continue

                normd = LA.norm(d)
                # print(iter_, lmd_L, lmd_U, sigma_L, lmd_c/normd, sigma_U)

                if sigma_L * normd < lmd_c < sigma_U * normd:
                    return d, lmd_c
                elif lmd_c < sigma_L * normd:
                    d, lmd_c = self.binarySearch(x, b, sigma_L, sigma_U, lmd_L, lmd_U)
                    return d, lmd_c
                else:
                    lmd_U = min(lmd_c, lmd_U)


                if abs(lmd_U - lmd_L) <= eps:
                    lmd_c = (lmd_U + lmd_L) * 0.5
                    d_plus = self.cgPhaseTwo(x, b, sigma_L, sigma_U, lmd_c, np.zeros(b.shape))
                    # print(d_plus, LA.norm(d_plus)* sigma_U, lmd_c)
                    if LA.norm(d_plus) * sigma_U > lmd_c:
                        return d_plus, lmd_c
                    else:
                        d = self.cgPhaseTwo(x, b, sigma_L, sigma_U, lmd_c, np.ones(b.shape))
                        # print(d, d-d_plus)
                        _, gamma = self.findQuadRoot(LA.norm(d - d_plus) ** 2, 2 * (d - d_plus).dot(d_plus), LA.norm(d_plus) ** 2 - lmd_c ** 2 / ((sigma_U+sigma_L)*0.5) ** 2)
                        # A = self.problem.getHessian(x)
                        # a = leftMost(A)
                        # print('xxxxxxx', A, a, lmd_c, (d - d_plus).dot(d_plus), self.problem.getHv(x, d-d_plus) - lmd_c*(d-d_plus))
                        return d_plus - np.sign(b.dot(d-d_plus)) * gamma * (d - d_plus), lmd_c

        def cgPhaseOne(self, x, b, sigma_L, sigma_U, lmd_c):
            cg_max_iters = int(config['IRSNT']['CG_MAX_ITERS'])
            max_iters = min(cg_max_iters, b.size * 2)
            eps = float(config['IRSNT']['MATRIX_PSD_TOL'])
            tol = float(config['IRSNT']['CG_STOP_TOL'])

            normb = LA.norm(b)

            cg_x = np.ones(b.shape)
            r = b - (self.problem.getHv(x, cg_x) + lmd_c * cg_x)
            self.counter.incrementHessVCount()

            p = r
            rs_old = r.dot(r)

            for i in range(max_iters):
                Ap = self.problem.getHv(x, p) + lmd_c * p
                self.counter.incrementHessVCount()
                pAp = p.dot(Ap)

                if pAp <= -eps or (abs(pAp) <= eps and LA.norm(p) >= eps):
                    psd_matrix = False
                    return 0, psd_matrix

                alpha = rs_old / pAp

                cg_x += alpha * p
                r -= alpha * Ap

                rs_new = r.dot(r)

                if rs_new ** 0.5 < tol * max(normb, 1.):
                    break

                p = r + (rs_new/rs_old) * p
                rs_old = rs_new

            # print('cgPhaseOne', i, rs_old, lmd_c)
            # if i == max_iters - 1:
                # print('Warning: cg run fails in ' + str(max_iters) + ' steps.')

            psd_matrix = True

            return cg_x, psd_matrix

        def cgPhaseTwo(self, x, b, sigma_L, sigma_U, lmd_c, cg_x):
            # print('start phase two...')
            cg_max_iters = int(config['IRSNT']['CG_MAX_ITERS'])
            max_iters = min(cg_max_iters, b.size * 2)
            tol = float(config['IRSNT']['CG_STOP_TOL'])

            tmp = b - (self.problem.getHv(x, cg_x) + lmd_c * cg_x)
            self.counter.incrementHessVCount()
            r = self.problem.getHv(x, tmp) + lmd_c * tmp
            self.counter.incrementHessVCount()

            Ab =self.problem.getHv(x, b) + lmd_c * b
            self.counter.incrementHessVCount()
            normAb = LA.norm(Ab)

            p = r
            rs_old = r.dot(r)

            for i in range(max_iters):
                Ap = self.problem.getHv(x, p) + lmd_c * p
                self.counter.incrementHessVCount()
                AAp = self.problem.getHv(x, Ap) + lmd_c * Ap
                self.counter.incrementHessVCount()

                pAAp = Ap.dot(Ap)

                alpha = rs_old / pAAp

                cg_x += alpha * p
                r -= alpha * AAp

                rs_new = r.dot(r)

                # print(i, rs_new, normAb)
                if rs_new ** 0.5 < tol * max(normAb, 1.):
                    break

                p = r + (rs_new/rs_old) * p
                rs_old = rs_new

            # print('phase II solution:', rs_old)
            # print('cgPhaseTwo', i, rs_new)
            return cg_x

        def binarySearch(self, x, b, sigma_L, sigma_U, lmd_L, lmd_U):
            eps = float(config['IRSNT']['BINARY_SEARCH_TOL'])
            # print('start binary search ...')
            while True:
                lmd_c = np.random.uniform(lmd_L, min(lmd_L+0.5*(lmd_L + lmd_U), lmd_U))
                d, _ = self.cgPhaseOne(x, b, sigma_L, sigma_U, lmd_c)
                normd = LA.norm(d)
                # print(sigma_L, lmd_c/normd, sigma_U)
                if sigma_L * normd - eps < lmd_c < sigma_U *normd + eps:
                    return d, lmd_c
                elif lmd_c > sigma_U * normd + eps:
                    lmd_U = lmd_c
                else:
                    lmd_L = lmd_c

        # def adaNT_solver(self, A, b, tau):

        #     eps_exact = 1e-8
        #     b += min(LA.norm(b), 1e-5) * self.getRandVecOnBall(len(A))
        #     # b += 1e-5 * self.getRandVecOnBall(len(A))
        #     s = b
        #     max_iters = 200
        #     do_order_two_update = False # TODO: check why second order update not working

        #     gl = min([A[i, i] - np.sum(np.abs(A[i, :])) + np.abs(A[i, i]) for i in range(len(A))])
        #     gu = max([A[i, i] + np.sum(np.abs(A[i, :])) - np.abs(A[i, i]) for i in range(len(A))])

        #     A_ii_min = min(np.diagonal(A))

        #     lmd_lower = max(0, -A_ii_min, gl)
        #     lmd_upper = max(gu, lmd_lower + eps_exact)

        #     lmd_c = np.random.uniform(lmd_lower, lmd_upper)

        #     if self.mode == 'tr':
        #         is_pd = isPDMatrix(A)
        #         if is_pd:
        #             L = LA.cholesky(A) # lower triangular factor
        #             self.counter.incrementCholeskyDecompCount()

        #             w = SLA.solve_triangular(L, b, lower=True)
        #             self.counter.incrementCholeskyLinearSolverCount()

        #             s = SLA.solve_triangular(L.T, w, lower=False)
        #             self.counter.incrementCholeskyLinearSolverCount()

        #             sn = LA.norm(s)

        #             if sn <= tau:
        #                 return s, 0

        #     for iter_ in range(max_iters):

        #         # print(iter, lmd_lower, lmd_c,  lmd_upper)

        #         lmd_plus_in_N = False
        #         lmd_in_N = False

        #         B = A + lmd_c * np.eye(self.problem.getSize())

        #         try: # if this succeeds, then lmd_c is in L or G
        #             L = LA.cholesky(B) # lower triangular factor
        #             self.counter.incrementCholeskyDecompCount()

        #             w = SLA.solve_triangular(L, b, lower=True)
        #             self.counter.incrementCholeskyLinearSolverCount()

        #             s = SLA.solve_triangular(L.T, w, lower=False)
        #             self.counter.incrementCholeskyLinearSolverCount()

        #             sn = LA.norm(s)

        #             phi_lmd = self.getPhiLmd(sn, lmd_c, tau)

        #             if abs(phi_lmd) < eps_exact*max(1, tau):
        #                 break

        #             y = SLA.solve_triangular(L, -s, lower=True)
        #             self.counter.incrementCholeskyLinearSolverCount()
        #             yn = LA.norm(y)

        #             if phi_lmd < 0 and lmd_c > 0: # lmd_c in L
        #                 # print('L', iter, lmd_lower, lmd_c, lmd_upper)
        #                 c_hi = self.orderOneUpdate(sn, yn, lmd_c, tau)
        #                 lmd_plus  = lmd_c + c_hi
        #                 lmd_c = lmd_plus

        #                 if abs(c_hi) < eps_exact*max(1, lmd_upper):
        #                     return s, lmd_c

        #             elif phi_lmd > 0 and lmd_c > 0: # lmd_c in G
        #                 # print('G', iter, lmd_lower, lmd_c, lmd_upper)
        #                 lmd_upper = min(lmd_upper, lmd_c)

        #                 c_hi = self.orderOneUpdate(sn, yn, lmd_c, tau)
        #                 if do_order_two_update:
        #                     s_plus = SLA.solve_triangular(L.T, y, lower=False)
        #                     self.counter.incrementCholeskyLinearSolverCount()
        #                     s_plus_n = LA.norm(s_plus)
        #                     c_hi = max(c_hi, self.orderTwoUpdate(sn, yn, s_plus_n, lmd_c, tau))
        #                 lmd_plus  = lmd_c + c_hi

        #                 if lmd_plus > 0:
        #                     try:
        #                         B_plus  = A + lmd_plus * np.eye(self.problem.getSize())

        #                         L = LA.cholesky(B_plus)
        #                         self.counter.incrementCholeskyDecompCount()
        #                         lmd_c = lmd_plus

        #                     except LA.LinAlgError:
        #                         lmd_plus_in_N = True

        #                 if lmd_plus <= 0 or lmd_plus_in_N:
        #                     if self.mode == 'tr':
        #                         if is_pd and lmd_lower == 0 and phi_lmd >= 0:
        #                                 lmd_c = 0
        #                                 break
        #                         else:
        #                             s, lmd_c, lmd_lower = self.adaNTupdateInN(A, s, lmd_lower, lmd_upper, lmd_plus, tau)
        #                     if self.mode == 'cubic':
        #                         s, lmd_c, lmd_lower = self.adaNTupdateInN(A, s, lmd_lower, lmd_upper, lmd_plus, tau)

        #         except LA.LinAlgError: # lmd_c in N
        #             lmd_in_N = True

        #         if lmd_in_N:
        #             if self.mode == 'tr':
        #                 if lmd_lower == 0 and is_pd and phi_lmd >= 0:
        #                     lmd_c = 0
        #                     break
        #                 else:
        #                     s, lmd_c, lmd_lower = self.adaNTupdateInN(A, s, lmd_lower, lmd_upper, lmd_c, tau)

        #             if self.mode == 'cubic':
        #                 s, lmd_c, lmd_lower = self.adaNTupdateInN(A, s, lmd_lower, lmd_upper, lmd_c, tau)

        #         sn = LA.norm(s)
        #         if np.isclose(sn, 0.) or np.isclose(lmd_c, 0.):
        #             break

        #         phi_lmd = self.getPhiLmd(sn, lmd_c, tau)
        #         if abs(phi_lmd) < eps_exact*max(1, tau):
        #             break

        #         # print(iter_, lmd_lower, lmd_c, lmd_upper, lmd_upper - lmd_lower)
        #     return s, lmd_c

        # def adaNTupdateInN(self, A, s, lmd_lower, lmd_upper, lmd, tau):
        #     theta_l = 0.01
        #     theta_u = 0.9
        #     eps_tol = 1e-6
        #     lmd_lower = max(lmd_lower, lmd)
        #     lmd_l = max((lmd_lower*lmd_upper)**0.5, lmd_lower + theta_l*(lmd_upper - lmd_lower))
        #     lmd_u = max((lmd_lower*lmd_upper)**0.5, lmd_lower + theta_u*(lmd_upper - lmd_lower))

        #     # print(lmd_l, lmd_u, lmd_u - lmd_l)
        #     lmd = np.random.uniform(lmd_l, lmd_u)

        #     # if np.isclose(lmd_lower, lmd_upper):
        #     # if abs(lmd_upper - lmd_lower) < eps_tol * max(1, lmd_upper):
        #     #     lmd = lmd_lower

        #     #     ew, ev = SLA.eigh(A, eigvals=(0, 0))
        #     #     self.counter.incrementEigenValueCount()
        #     #     self.counter.incrementEigenVectorCount()

        #     #     d = ev[:, 0]

        #     #     if ew >= 0:
        #     #         return s, lmd, lmd_lower

        #     #     if self.mode == 'cubic':
        #     #         tao_lower, tao_upper = self.findQuadRoot(1, 2*s.dot(d), s.dot(s) - lmd**2 / tau**2)
        #     #     elif self.mode == 'tr':
        #     #         # print('xxxx:', s, d, s.dot(s), tau**2, lmd, lmd_lower)
        #     #         tao_lower, tao_upper = self.findQuadRoot(1, 2*s.dot(d), s.dot(s) - tau ** 2)
        #     #     else:
        #     #         raise NotImplementedError('order one update mode {} not implemented!'.format(self.mode))

        #     #     s += tao_upper * d

        #     return s, lmd, lmd_lower




































