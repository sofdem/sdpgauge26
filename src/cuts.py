#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sophie Demassey (sofdem@gmail.com), Antonio Sasaki, Valentina Sessa

Spectral-Gauge Cutting-Plane for SDP: min_{x PSD, tr(x)<=t} <c, x> : <a_k, x> = b_k for k in [m]
with gurobipy:
initialize a SOC or LP relaxation for SDP (i.e. relax condition x PSD),
then run iteratively the spectral-gauge separation problem for a given a symmetric gauge
if the relaxed solution is not PSD then generate the optimal cut and add it to the relaxation
"""
import time
import random
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import inout as si


def soclincut(n: int, xrel: np.array, xvar: dict, model, soctol, nmaxcuts: int):
    """
    generate a limited number of linear OA cuts for the SOC relaxation at the current iterate y
    among all (i,j) such that i < j and yii, yjj, yij are not too close to 0:

    given f(x) = xij^2 - xii.xjj and point y such that f(y) > 0 (then y not SOC), the OA cut is:
    g(x) = f(y) + df(y)(x-y) = (2.yij.xij - yii.xjj - yjj.xii) - f(y) <= 0 for every x SOC
    add to the model the 'ncuts' with the highest violations f(y)

    :param n: the dimension
    :param xrel: the current relaxed symmetric solution matrix represented as an UPPER triangular matrix
    :param xvar: the gurobi variable matrix
    :param model: the gurobi model
    :param soctol: tolerance for violating the SOC constraint f(x) <= 0
    :param nmaxcuts: maximum number of cuts to add, or 0 if all cuts must be added
    :return: the number of generated cuts => number of added cuts = min(nmaxcuts, ngencuts)
    """

    coeftol = soctol/100
    cuts = []
    for j in range(n):
        if xrel[j, j] < coeftol:
            continue
        for i in range(j):
            if abs(xrel[i, j]) > coeftol and xrel[i, i] > coeftol:
                delta = xrel[i, j] * xrel[i, j] - xrel[i, i] * xrel[j, j]
                if delta > soctol:
                    cuts.append((delta, i, j, xrel[i, j], xrel[j, j], xrel[i, i]))
    if not cuts:
        return 0
    ordcuts = sorted(cuts, key=lambda x: x[0], reverse=True)
    print(ordcuts)
    print(f"number of SOC violations = {len(ordcuts)} max={ordcuts[0][0]} min={ordcuts[-1][0]}, "
          f"max number of cuts = {nmaxcuts}")
    if nmaxcuts == 0:
        nmaxcuts = len(ordcuts)
    cnt = 0
    for c in ordcuts:
            i = c[1]
            j = c[2]
            model.addConstr(2 * c[3] * xvar[i, j] - c[4] * xvar[i, i] - c[5] * xvar[j, j] <= c[0])
            cnt += 1
            if cnt > nmaxcuts:
                return cnt
    return len(ordcuts)


def gaugecuts(xrel: np.array, xvar: dict, eigtol, ranks: list, norm: int) -> (float, float, float, int, list):
    """
    solve the spectral-gauge separation problem for L2 (Frobenius cuts) if norm=2 or Lmax (nuclear cuts) if norm=0
    for each rank restriction listed in 'ranks' (rank=0 means no restriction) at the current iterate
    get a list of solutions S (optimal separator) and values g (optimal violation)
    (the resulting list may be smaller thant 'ranks' as redundant cuts are ignored)

    :param xrel: the current relaxed symmetric solution matrix represented as an UPPER triangular matrix
    :param xvar: the gurobi variable matrix
    :param eigtol: eigenvalue tolerance
    :param ranks: the list of considered ranks (possibly starting with k=0 (=all), then in decreasing order)
    :param norm: 2 if Frobenius cuts, 0 if nuclear cuts
    :return: the minimum eigenvalue of xrel,
             the violation and the density of the first generated cut (full rank),
             the number of selected negative eigenvectors in the last generated cut (lowest rank),
             and the list of gauge cuts <S,X> as gurobi linear expressions
    """
    n = xrel.shape[0]

    # xrel = (xrel + xrel.T)/2 do not use this: xrel is upper triangle

    eigvals, eigvecs = np.linalg.eigh(xrel, UPLO='U')  # sp.eigh(xrel, lower=False)  # upper triangle of xrel

    lmin = eigvals[0]
    if lmin > -eigtol:
        return lmin, 0, 0, 0, []

    lnegidx = np.nonzero(eigvals < -eigtol)[0]

    cuts = []
    depth = 1
    density = -1

    for k in ranks:
        if k == 'S':
            lnegidx = np.nonzero(eigvals < lmin * 9 / 10)[0]
        elif k > 0:
            if len(cuts) > 1 and len(lnegidx) < k:  # cut has already been generated
                continue
            lnegidx = lnegidx[:k]

        # print(eigvals[lnegidx])
        s, violation = solvegaugeseparation(eigvecs, eigvals, lnegidx, norm)
        cuts.append(2 * gp.quicksum(s[i, j] * xvar[i, j] for j in range(n) for i in range(j))
                    + gp.quicksum(s[i, i] * xvar[i, i] for i in range(n)))

        if density == -1:
            depth = violation
            density = np.count_nonzero(abs(s) > eigtol) / (n * n)

    return lmin, depth, density, len(lnegidx), cuts


def solvegaugeseparation(eigvecs, eigvals, lnegidx: np.array, norm: int):
    """
    'lnegidx' is a sublist of indexes (typically the k first) in arrays eigvecs/eigvals
    form matrix Q as the sublist of eigenvectors and vector w as the sublist of (all negative) eigenvalues
    define X = Q.diag(w).Q^T
    the optimal gauge separator S and the corresponding violation g are defined by:
    if norm = 2 (Frobenius cut, gauge=L2): g = -|X|_F = -|w|_2 and S = X / -g
    if norm = 0 (projector cut, gauge=Lmax): g = -|w|_1 and S = Q.Q^T
    :param eigvecs: eigenvectors (in decreasing order of eigenvalue)
    :param eigvals: eigenvalues (in decreasing order)
    :param lnegidx: list of the indexes of eigenvalues to consider
    :param norm: either 0 or 2
    :return: S (optimal gauge separator), g (optimal violation)
    """
    assert norm == 2 or norm == 0
    assert eigvals[lnegidx[-1]] < 0
    q = eigvecs[:, lnegidx]
    s = None
    depth = None
    if norm == 2:
        w = - eigvals[lnegidx]
        xneg = (q * w) @ q.T         # X^- = Q diag(w) Q^T
        fnorm = np.linalg.norm(w)    # ||X^-||_F
        s = xneg / fnorm
        depth = -fnorm
    elif norm == 0:
        s = q @ q.T
        depth = np.sum(eigvals[lnegidx])
    return s, depth


def sdprelax_gurobi(model: gp.Model, snmcabt: tuple, soc: bool = True):
    """
    LP/SOC relaxations of SDP in Gurobi
    min_{x in symmetric(n)} sum_{ij} c{ij}x{ij}: sum_{ij} a{kij}x{ij} = b_k,  sum_{i} x{ii} <= t,  x{ii} >= 0
    + LP: x{ii} + x{jj} + 2x{ij} >= 0, x{ii} + x{jj} - 2x{ij} >= 0  (first one always holds if x >= 0)
    + SOC: x{ij} * x{ij} <= x{ii} * x{jj}
    """

    s, n, m, c, a, b, t, l, u = si.check_sdpinstance(snmcabt)

    lb = l if l > -si.MAXINT else -float('inf')
    ub = u if u < si.MAXINT else float('inf')
    x = {(i, j): model.addVar(lb=lb, ub=ub, name=f"x({i},{j})") for j in range(n) for i in range(j+1)}

    if s.startswith("PCA"):
        model.addConstr(gp.quicksum(x[i, i] for i in range(n)) == 1.0, "trub")
    elif t >= 0:
        model.addConstr(gp.quicksum(x[i, i] for i in range(n)) <= t, "trub")

    for j in range(n):
        x[j, j].lb = 0.0
        for i in range(j):
            if soc:
                model.addQConstr(x[i, j] * x[i, j] <= x[i, i] * x[j, j], name=f"soc{i},{j}")
            else:
                model.addConstr(x[i, i] + x[j, j] - 2 * x[i, j] >= 0, name=f"lp-{i},{j}")
                if lb < 0:
                    model.addConstr(x[i, i] + x[j, j] + 2 * x[i, j] >= 0, name=f"lp+{i},{j}")

    obj = gp.quicksum(2 * c[i, j] * x[i, j] for j in range(n) for i in range(j)) \
        + gp.quicksum(c[j, j] * x[j, j] for j in range(n))
    model.setObjective(obj, GRB.MINIMIZE)

    model.addConstrs(((b[k] == gp.quicksum(a[k, i, j] * x[i, j] for j in range(n) for i in range(j))
                       + gp.quicksum(a[k, j, j] * x[j, j] for j in range(n))) for k in range(m)), name="lin")

    if s.startswith("PCA"):
        assert lb < 0 < ub and t > 0
        addnorm1cst(model, x, lb, ub, n, t)

    if s.startswith("QP"):
        assert lb == 0.0 and ub == 1.0
        addmccormick(model, x, n-1)
        model.addConstr(x[n-1, n-1] == 1, name="schur")

    return x


def addnorm1cst(model: gp.Model, x: dict, lb: float, ub: float, n: int, k: float):
    """
    decorate the Gurobi relaxation with the additional constraint sum_{ij} |x{ij}| <= k
    modeled as: -xp{ij} <= x{ij} <= xp{ij}, xp >= 0, sum_{ij} xp{ij} <= k
    """
    if lb >= 0.0:
        print(f"add 1-norm constraint with lb = {lb} >= 0")
        model.addConstr(2 * gp.quicksum(x[i, j] for j in range(n) for i in range(j))
                        + gp.quicksum(x[j, j] for j in range(n)) <= k, name=f"unitsum")
    else:
        assert ub > 0.0
        print(f"add 1-norm constraint with lb = {lb} < 0")
        xp = {(i, j): model.addVar(lb=0.0, ub=ub, name=f"x({i},{j})") for j in range(n) for i in range(j)}
        for j in range(n):
            for i in range(j):
                model.addConstr(x[i, j] <= xp[i, j], name=f"abs+{i},{j}")
                model.addConstr(x[i, j] >= -xp[i, j], name=f"abs-{i},{j}")

        model.addConstr(2 * gp.quicksum(xp[i, j] for j in range(n) for i in range(j))
                        + gp.quicksum(x[j, j] for j in range(n)) <= k, name=f"unitsum")


def addmccormick(model: gp.Model, x: dict, n: int):
    """
    decorate the Gurobi relaxation with the Mc Cormick's inequalities in the case of BoxQP[0,1]:
    y{i} + y{j} - 1 <= z{ij} <= min (y{i}, y{j}) with x = [[z, y],[y^T, 1]] Schur's complement matrix
    """
    print(f"add McCormick constraints for BoxQP with y in [0,1]")
    model.addConstrs(((x[i, n] + x[j, n] - 1 <= x[i, j]) for j in range(n) for i in range(j+1)), name=f"mcl")
    model.addConstrs(((x[i, j] <= x[i, n])  for j in range(n) for i in range(j+1)), name=f"mci")
    model.addConstrs(((x[i, j] <= x[j, n])  for j in range(n) for i in range(j)), name=f"mcj")


def solve_gurobi(sdpinst: tuple, soc: bool = True, verbose: bool = False):
    """ solve the gurobi relaxed model of the sdp instance given as a tuple (s, n, m, c, a, b, t)
    given 'soc' option: if soc: SOC model, else: LP model """

    name = "soc" if soc else "lp"
    trace = []

    with gp.Model(name) as model:
        x = sdprelax_gurobi(model, sdpinst, soc)
        model.params.OutputFlag = 1 if verbose else 0
        model.optimize()
        if model.Status != GRB.OPTIMAL:
            raise gp.GurobiError(10000, f"Optimization ended with status {model.Status}")
        trace.append((model.ObjVal, model.Runtime))

    return trace


def parsecutmode(mode: str):
    mode = mode.upper()
    parts = mode.split("-")
    if len(parts) < 2 or parts[0] not in ("SOC", "LP") or parts[1] not in ("EIG", "FROB"):
        raise ValueError(f"invalid mode='{mode}'")

    issoc = (parts[0] == "SOC")
    cutname = parts[1]
    cutnorm = 2 if cutname == 'FROB' else 0

    cutrand = False
    cutupdate = False
    cutranks = None

    # no specified ranks: the rank is upated dynamically starting from 'all'
    if len(parts) < 3:
        cutranks = [0]
        cutupdate = True
    else:
        ranks = parts[2]
        cutname = f"{cutname}-{ranks}"
        # the rank is chosen randomly in a list
        if '|' in ranks:
            cutranks = [int(x) for x in ranks.split("|")]
            cutrand = True
        # a cut is generated for each rank in the list in decreasing order (i.e. '0' commes first)
        elif '+' in ranks:
            tmp0 = ranks.split("+")
            tmp = [int(x) for x in tmp0 if x != '0']
            tmp.sort(reverse=True)
            cutranks = tmp if len(tmp) == len(tmp0) else [0] + tmp
            assert len(cutranks) == len(tmp0)
        # one specified rand
        elif ranks.startswith('S'):
            cutranks = ['S']
        else:
            cutranks = [int(ranks)]

    return issoc, cutname, cutnorm, cutranks, cutrand, cutupdate


def cutgen(sdpinst, mode: str = "SOC-PROJ", verbose: bool = False, maxit=-1, maxcpu=-1, eigtol=1e-3, ub=None):
    """ run cutting-plane on the sdp instance given as a tuple (s, n, m, c, a, b, t)
    'mode' is a string made of 3 parts separated by -:
     1: the relaxation mode 'SOC' or 'LP'
     2: the gauge cut norm 'EIG' (norm=inf) or 'FROB' (norm=2)
     3: the max rank e.g. 0 (all), 1, 10 or a list of max ranks either:
     separated by '|" if generate 1 (pick randomly)/iteration, or separated by '+" if generate all/iteration
     e.g: 'LP-FROB-0', 'SOC-EIG-1', 'LP-EIG-0|1|10', 'LP-EIG-0+1+10',
    """

    s = sdpinst[0]
    n = sdpinst[1]

    issoc, cutname, cutnorm, cutranks, cutrand, cutupdate = parsecutmode(mode)
    cutlastit = 0
    cuttol = 1e-3

    trace = {}
    sol = np.zeros([n, n])
    maxit = 100000 if maxit < 0 else maxit
    nsoccuts = 0
    gap0 = None

    print(f"=============== solve cutgen {s} {mode}")
    with gp.Model(mode) as model:
        cpu00 = time.time()
        xvar = sdprelax_gurobi(model, sdpinst, issoc)
        model.params.OutputFlag = 1 if verbose else 0
        # model.params.Method = 1
        model.params.BarQCPConvTol = 1e-5   # required for removing some suboptimality (code 13)
        model.params.BarHomogeneous = 1     # slower than original Barrier but seems more robust
        cpu0 = time.time()
        print(f"model built in {cpu0-cpu00} s.")

        for it in range(maxit):
            model.optimize()
            if it == 0 and ub:
                gap0 = ub - model.objVal
                if gap0 < 1e-4:
                    gap0 = 0
            if model.Status != GRB.OPTIMAL:
                print(f"STOP gurobi status={model.Status}")
                trace[it] = {
                    "lb": trace[it-1]["lb"] if it > 0 else -GRB.MAXINT,
                    "cpuit": model.Runtime,
                    "cpu": time.time() - cpu0,
                    "status": -model.Status
                }
                trace[it]["gc"] = (trace[it]["lb"] - trace[0]["lb"]) / gap0 if gap0 and it > 0 else 0
                break

            for j in range(n):
                sol[j, j] = xvar[j, j].x
                for i in range(j):
                    sol[i, j] = sol[j, i] = xvar[i, j].x

            ranks = [random.choice(cutranks)] if cutrand else cutranks
            lmin, depth, density, lnb, cutlist = gaugecuts(sol, xvar, eigtol, ranks, cutnorm)
            # if not issoc:
            #    nsoccuts += soclincut(n, sol, xvar, model, 1e-2, 20)
            trace[it] = {
                "lb": model.ObjVal,
                "eigmin": float(lmin),
                "depth": float(depth),
                "density": round(density * 100),
                "nbnegeig": lnb,
                "nbcuts": len(cutlist),
                "nbsoccuts": nsoccuts,
                "cpuit": model.Runtime,
                "cpu": time.time() - cpu0,
                "status": 0,
                "cuttype": cutname
            }
            trace[it]["gc"] = (trace[it]["lb"] - trace[0]["lb"]) / gap0 if gap0 and it > 0 else 0
            print(f"iteration {it}, {trace[it]}")

            if lmin > -eigtol:
                print(f"STOP: lmin={lmin} > -{eigtol}")
                trace[it]["status"] = 1
                # trace[it]["sol"] = sol
                break

            if 0 < maxcpu < time.time() - cpu00:
                print(f"STOP: cpu > {maxcpu}")
                break

            for cut in cutlist:
                model.addConstr(cut >= 0)

            if cutupdate:
                if cutranks[0] != 1 and it - cutlastit > 30:
                    delta = (trace[it-2]["lb"] - trace[it]["lb"]) / (trace[cutlastit]["lb"])  # - trace[it]["lb"])
                    print(f"change -2 = {delta}")
                    if delta < cuttol:
                        cutlastit = it
                        cutranks[0] = max(1, int(lnb / 2))
                        cuttol = cuttol / 2
                        print(f"half rank = {cutranks}")
                        if cutranks[0] == 1:
                            cutupdate = False

    if len(trace) > 0:
        trace[0]["cpub"] = cpu0 - cpu00
    return trace
