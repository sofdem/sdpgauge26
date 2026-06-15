#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sophie Demassey (sofdem@gmail.com), Antonio Sasaki, Valentina Sessa

Mosek SDP model: min_{x PSD, tr(x) <= t} <c, x> : <a_k, x> = b_k for k in [m] test
"""

import sys
import numpy as np
import mosek.fusion as mk

import time

import inout as si
from scipy.linalg import eigh


def sdp_mosek(model: mk.Model, snmcabt: tuple, mode: str):
    """
    populate the Mosek model with the SDP instance (s,n,m,c,a,b,t)
    return the matrix variable x
    """
    s, n, m, c, a, b, t, l, u = si.check_sdpinstance(snmcabt)

    x = model.variable("x", mk.Domain.inPSDCone(n))
    if l > -si.MAXINT:
        model.constraint("lb", x,  mk.Domain.greaterThan(l))
    if u < si.MAXINT:
        model.constraint("ub", x,  mk.Domain.lessThan(u))

    if t >= 0:
        if s.startswith("PCA"):
            model.constraint("trub", mk.Expr.sum(x.diag()), mk.Domain.equalsTo(1.0))
        else:
            model.constraint("trub", mk.Expr.sum(x.diag()), mk.Domain.lessThan(t))

    cm = mk.Matrix.dense(c)
    model.objective(mk.ObjectiveSense.Minimize, mk.Expr.dot(cm, x))

    for k in range(m):
        ak = mk.Matrix.dense(a[k])
        model.constraint(f"c{k}", mk.Expr.dot(ak, x), mk.Domain.equalsTo(b[k]))

    if s.startswith("PCA"):
        assert l < 0 < t
        addnorm1cst(model, x, l, n, t)

    if s.startswith("QP"):
        model.constraint("schur", x[n-1, n-1], mk.Domain.equalsTo(1.0))
        if mode != "SDP_NOMC":
            addmccormick(model, x, n-1)

    return x


def addnorm1cst(model: mk.Model, x, lb: float, n: int, k: float):
    """
    decorate the Mosek SDP model with the additional constraint sum_{ij} |x{ij}| = k
    modeled as: -xp{ij} <= x{ij} <= xp{ij}, xp >= 0, sum_{ij} xp{ij} <= k
    """
    if lb >= 0.0:
        print(f"add 1-norm constraint with lb = {lb} >= 0")
        model.constraint(f"usum", mk.Expr.sum(x), mk.Domain.lessThan(k))
    else:
        print(f"add 1-norm constraint with lb = {lb} < 0")
        xp = model.variable("xp", [n, n], mk.Domain.greaterThan(0.0))
        model.constraint(f"abs", mk.Expr.sub(xp.diag(), x.diag()), mk.Domain.equalsTo(0))
        for j in range(n):
            for i in range(j):
                model.constraint(f"abs{i},{j}", mk.Expr.vstack(xp.index([j, i]), x.index([j, i])), mk.Domain.inQCone())
                model.constraint(f"abs{j},{i}", mk.Expr.vstack(xp.index([i, j]), x.index([i, j])), mk.Domain.inQCone())
       # @todo best way to model in Mosek ?

        model.constraint(f"usum", mk.Expr.sum(xp), mk.Domain.lessThan(k))


def addmccormick(model: mk.Model, x, n: int):
    """
    decorate  Mosek SDP model with the Mc Cormick's inequalities in the case of BoxQP[0,1]:
    y{i} + y{j} - 1 <= z{ij} <= min (y{i}, y{j}) with x = [[z, y],[y^T, 1]] Schur's complement matrix
    """
    print(f"add McCormick constraints for BoxQP with y in [0,1]")
    for j in range(n):
        for i in range(j+1):
            model.constraint(f"mcl{i},{j}", x[j, n] + x[i, n] - x[i, j], mk.Domain.lessThan(1))
            model.constraint(f"mcu{i},{j}", x[i, n] - x[i, j], mk.Domain.greaterThan(0))
            if i != j:
                model.constraint(f"mcu{j},{i}", x[j, n] - x[i, j], mk.Domain.greaterThan(0))
       # @todo best way to model in Mosek ?


def solve_mosek(snmcabt: tuple, eigtol=1e-7, maxcpu=-1, verbose: bool = False, mode: str = "SDP"):
    """
    create and solve a Mosek model with the SDP instance (s,n,m,c,a,b,t)
    return solving information as a dict
    """
    n = snmcabt[1]
    trace = {}

    print(f"=============== solve sdp")
    with mk.Model("sdp") as model:
        cpu00 = time.time()
        x = sdp_mosek(model, snmcabt, mode)
        model.setSolverParam("intpntCoTolRelGap", eigtol)
        model.setSolverParam("optimizerMaxTime", maxcpu)
        cpub = time.time() - cpu00
        if verbose:
            model.setLogHandler(sys.stdout)

        model.solve()
        if model.getProblemStatus() != mk.ProblemStatus.PrimalAndDualFeasible:
            # @todo maxcpu does not works when IP is in the eigenvalue computation phase
            print(f"Optimization ended with status {model.getProblemStatus()}")
            trace[0] = {"pobj": model.getSolverDoubleInfo("intpntPrimalObj"),
                        "dobj": model.getSolverDoubleInfo("intpntDualObj"),
                        "lb": model.getSolverDoubleInfo("intpntPrimalObj"),
                        "ipstat": model.getSolverDoubleInfo("intpntOptStatus"),
                        "cpu": model.getSolverDoubleInfo("optimizerTime"),
                        "cpub": cpub,
                        "niter": model.getSolverIntInfo("intpntIter"),
                        "status": -1,
                        "gc": 1}
            print(trace[0])
            return trace

        solution = np.reshape(x.level(), (n, n))
        trace[0] = {"lb": model.primalObjValue(), "cpu": model.getSolverDoubleInfo("optimizerTime"),
                    "cpub": cpub, "eigmin": eigh(solution)[0][0], "niter": model.getSolverIntInfo("intpntIter"),
                    "status": 1, "gc": 1}
        print(trace[0])
        # trace[0]["sol"] = solution

    return trace
