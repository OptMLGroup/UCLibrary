A `PYTHON`library for optimizing `CUTEST Unconstrained Problem` and/or `Empirical Risk Minimization`(including`Deep Neural Networks`).

[TOC]

---

Author  |   `Xi He`
--- | :---
Email   |`heeryerate@gmail.com`|

## Introduction
The library contains a collection of first/second-order optimization algorithms for standard unconstrained optimization problem set `CUTEST` and/or `Empirical Risk Minimization`.

## Dependencies

+ numpy
+ pytorch
+ cutest (see [cutest installation](http://xihey.com/2017/12/12/Installing-CUTEst-on-MacOSX-with-Matlab-Python/))

## Tour of the UCLibrary
Here we exemplify a instance as to solve a problem in Cutest problem set.

### Single Run
```python
import ...
config.read('config.ini')
problem = CutestProblem('ROSENBR')
demo = Demo(problem)

problem.setInitialPoint()
optim = optimizers.Cubic(problem, lr=False, mode='exact', adaptive=True)

for _ in range(config.max_iters):
    optim.step()
    if optim.terminationCondition(mode='first_order', tol=config.tol):
        break

demo.addContourTrace(optim)
demo.drawPerformancePlot(optim)
demo.showPlot()
------------------
Default initial point by Cutest...
Cubic-exact-adaptive
('cholesky_decomp_counter', 132)
('cholesky_linear_solver_counter', 292)
('hess_counter', 30)
('step_counter', 30)
('grad_counter', 30)
```

### Multiple Run
```
import ...

def run(problem, optim, demo):
    for _ in range(config.max_iters):
        optim.step()
        if optim.terminationCondition(mode='first_order', tol=config.tol):
            break
    demo.addContourTrace(optim)
    demo.drawPerformancePlot(optim)

config.read('config.ini')
problem = CutestProblem('ROSENBR')
problem.setInitialPoint()

demo = Demo(problem)

optim = optimizers.Cubic(problem, lr=False, mode='exact', adaptive=True)
run(problem, optim, demo)

optim = optimizers.TrustRegion(problem, mode='cauchy')
run(problem, optim, demo)

demo.showPlot()
```
![f](https://bytebucket.org/xih314/mreleven/raw/3826298c24e2eec517aafbdc6d0b381735ff0b6f/src/demos/ROSENBR-performance.png)
![f](https://bytebucket.org/xih314/mreleven/raw/3826298c24e2eec517aafbdc6d0b381735ff0b6f/src/demos/ROSENBR-trace.png)

## List of Main Modules
### Gradient Based Solver, *optimizer.GD*
+ Vanilla GD
+ Nesterov's acceleration
+ Heavy Ball acceleration
+ Dynamic momentum
+ Optimal momentum
+ Static momentum
+ Restart scheme

### Practical Curvature based, *optimizer.Newton*
+ Constant damping
+ Levenberg–Marquardt damping
+ Truncated hessian

### Trust Region, *optimizer.TrustRegion*
+ Vanilla TR

### Cubic Regularization, *optimizer.Cubic*
+ Vanilla CR
+ Adaptive CR
+ CRm (CR with momentum)

### Line Search, *LineSearch*
+ Backtracking (Amijo linesearch)
+ Strong-wolfe

### Subsolvers, *SubRoutine*
+ Exact solver for positive definite matrix
+ Cauchy point
+ Dog-leg
+ CG for positive positive definite matrix
+ Steinghaug-Toint CG
+ Generalized Lanczos trust region
+ exact tridiagonal matrix subsolver
+ exact regularized subproblem solver
+ AdaNT