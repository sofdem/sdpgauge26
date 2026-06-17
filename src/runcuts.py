#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sophie Demassey (sofdem@gmail.com), Antonio Sasaki, Valentina Sessa

run benchmarks for the Spectral-Gauge Cutting-Plane method
for SDP: min_{x PSD, tr(x)<=t} <c, x> : <a_k, x> = b_k for k in [m]
"""
from pathlib import Path
import csv
import time
from graphics import trace_progresses
from datetime import datetime
import pandas as pd
import inout as si
from sdp import solve_mosek
from cuts import cutgen
import genmatrix as gm


def testinst(sdpinst: tuple, instname: str, modes, maxit=-1, maxcpu=-1, eigtol=1e-3, graphics=1):
    """ run mosek SDP  vs gurobi cutting-plane for LP/SOC relaxations + spectral-gauge cuts
    on instance 'sdpinst' given as a tuple (n, m, c, a, b, t) see gen_sdpinstance.py
    for a limited number 'maxit' of iterations
    and given a tolerance 'eigtol' to measure the negativity of eigenvalues """

    traces = {}
    print(f"instance={instname}, modes={modes}, maxcpu={maxcpu}")
    compareSDP = (maxcpu == -2)
    assert not compareSDP or modes[0].startswith("SDP")
    sdpval = None

    for m in modes:
        cpu00 = time.time()
        traces[m] = solve_mosek(sdpinst, maxcpu=maxcpu, eigtol=eigtol, verbose=True, mode=m) if m.startswith("SDP") \
            else cutgen(sdpinst, verbose=False, maxit=maxit, maxcpu=maxcpu, eigtol=eigtol, mode=m, ub=sdpval)
        cputotal = time.time() - cpu00
        traces[m][len(traces[m])-1]["cputotal"] = cputotal

        if m.startswith('SDP'):
            sdpval = traces[m][0]["lb"]
            if compareSDP and m.startswith('SDP'):
                maxcpu = cputotal
                print(f"maxcpu = {maxcpu} (SDP time), ub={sdpval} (SDP val)")

    print("-----------------------------")
    print(f"instance={instname}, modes={modes}")
    print(f"dim={sdpinst[1]}\t cts={sdpinst[2]}\t maxit={maxit}\t maxcpu={maxcpu}\t tol={eigtol}")
    printstats(traces)

    if graphics:
        ts = datetime.now().strftime("%y%m%d-%H%M")
        pngfile = Path(gm.OUTDIR / f"{instname}-{ts}.png") if graphics == 2 else None
        viewgraphics(instname, traces, pngfile=pngfile, gc=False)


def runinst(sdpinst: tuple, mode: str, maxit=-1, maxcpu=-1, eigtol=1e-3, resfile: Path = None, ub=None):
    """ run mosek SDP  vs gurobi cutting-plane for LP/SOC relaxations + spectral-gauge cuts
    on instance 'sdpinst' given as a tuple (s, n, m, c, a, b, t) see gen_sdpinstance.py
    for a limited number 'maxit' of iterations
    and given a tolerance 'eigtol' to measure the negativity of eigenvalues """

    cpu00 = time.time()
    trace = solve_mosek(sdpinst, maxcpu=maxcpu, eigtol=eigtol, verbose=False, mode=mode) if mode.startswith("SDP") \
        else cutgen(sdpinst, verbose=False, maxit=maxit, maxcpu=maxcpu, eigtol=eigtol, mode=mode, ub=ub)
    cputotal = time.time() - cpu00
    if resfile:
        fieldnames = list(trace[0].keys())
        with open(resfile, mode='w') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(trace.values())

    it = len(trace)
    avgcpuit = sum([tr["cpuit"] for tr in trace.values()]) / it if (trace[0].get("cpuit")) \
        else trace[it-1].get("cpu")/it
    res = trace[it-1]
    res["it"] = it
    res["cpu/it"] = avgcpuit
    res["lb0"] = trace[0].get("lb", "")
    res["cpu0"] = trace[0]["cpu"]
    res["cpub"] = trace[0].get("cpub", "0")
    res["iters"] = parsetrace(mode, trace)
    res["cputotal"] = cputotal
    return res


def testbench(collection: str, modes, maxit=100, maxcpu=-1, eigtol=1e-3, graphics=0) -> pd.DataFrame:
    """ run methods in 'modes' on all instancesin 'collection' """

    collection = collection.upper()
    ts = datetime.now().strftime("%y%m%d-%H%M")
    csv_path = gm.OUTDIR / f"{collection.lower()}_{ts}.csv"
    resfile = gm.OUTDIR / f"{collection.lower()}_{ts}.csv"
    resdir = Path(gm.OUTDIR / f"{collection.lower()}_{ts}") if graphics >= 2 else None
    if resdir:
        resdir.mkdir(parents=True)

    cols = ("lb", "cpu", "it", "cpu/it", "eigmin", "status", "lb0", "cpu0", "cpub", "cputotal", "gc")
    with open(resfile, 'w') as f:
        f.write("inst,mode," + ",".join(cols) + "\n")

    rows = []

    compareSDP = (maxcpu == -2)
    computeGC = modes[0].startswith("SDP")
    assert not compareSDP or computeGC

    instances = gm.parsebench(collection)
    print(f"bench={collection}: {len(instances)} instances, modes={modes}, maxcpu={maxcpu}")

    iters = {inst.stem: {"lb": {}, "cpuit": {}, "gc": {}} for inst in instances} if graphics else None

    for inst in instances:
        print(inst)
        sdpinst = si.readsdp(inst, collection)
        print(f"test instance: {inst.stem}, s={sdpinst[0]}, n={sdpinst[1]}, m={sdpinst[2]}")
        outinst = {"dataset": collection, "matrix": inst.stem, "s": sdpinst[0], "n": sdpinst[1], "m": sdpinst[2]}
        sdpval = None

        for mode in modes:
            if compareSDP and mode.startswith('SDP'):
                maxcpu = -1
                sdpval = None

            outmode = {"mode": mode, "maxit": maxit, "maxcpu": maxcpu, "eigtol": eigtol}
            instresfile = Path(resdir / f"{inst.stem}_{mode}.csv") if resdir else None

            res = runinst(sdpinst, mode, maxit=maxit, maxcpu=maxcpu, eigtol=eigtol, resfile=instresfile, ub=sdpval)
            outres = {k: res.get(k, None) for k in cols}
            rows.append({**outinst, **outmode, **outres})
            with open(resfile, 'a') as f:
                f.write(f"{inst.stem},{mode}," + ",".join(str(res.get(k, "")) for k in cols) + "\n")
            if iters and (len(res) > 1 or res[0]["status"] in (0, 1)):
                iters[inst.stem]["lb"][mode] = res["iters"][0]
                iters[inst.stem]["cpuit"][mode] = res["iters"][1]
                iters[inst.stem]["gc"][mode] = res["iters"][2]

            if computeGC and mode.startswith('SDP'):
                if res["status"] == 1:
                    sdpval = res["lb"]
                if compareSDP:
                    maxcpu = res["cputotal"]
                    print(f"maxcpu = {maxcpu} (SDP time), ub={sdpval} (SDP val)")

        if iters:
            pngfile = Path(resdir / f"{inst.stem}.png") if resdir else None
            if sdpval is None:
                trace_progresses(iters[inst.stem]["lb"], iters[inst.stem]["cpuit"], inst.stem, pngfile=pngfile)
            else:
                trace_progresses(iters[inst.stem]["gc"], iters[inst.stem]["cpuit"], inst.stem, pngfile=pngfile,
                                 y1label="gc", g1label="gap closed")

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"wrote results to {csv_path}")
    return df


def printstats(traces):

    initdict = {i: next((k for k in traces.keys() if k.startswith(i)), None) for i in ("SDP", "SOC", "LP")}
    print(f"lb: \t" + "\t".join(f"{i}={traces[v][0]['lb']:.5f}" for i, v in initdict.items() if v))
    print(f"cpu: \t" + "\t".join(f"{i}={traces[v][0]['cpu']:.5f}" for i, v in initdict.items() if v))

    # sdpval = None
    # if "SDP" in initdict.keys():
    #     sdpval = traces[initdict["SDP"]][0]["lb"]

    for k, trace in traces.items():
        it = len(trace)
        # gc = 0 if (sdpval is None or sdpval < trace[it - 1]["lb0"] + 1e-3) \
        #    else (trace[it - 1]["lb"] - trace[it - 1]["lb0"]) / (sdpval - trace[it - 1]["lb0"])
        avgcpuit = 0 if not (trace[0].get("cpuit")) else sum(tr["cpuit"] for tr in trace.values()) / it
        print(f"{k:10s}\t stus={trace[it-1]["status"]}\t it={it}\t "
              f"lb={trace[it - 1]["lb"]:.5f}\t gc={trace[it - 1]["gc"]:.2f}\t"
              f"cpu={trace[it - 1]["cpu"]:.5f}\t cpu/it={avgcpuit:.5f}\t"
              f"cpub={trace[0]["cpub"]:.2f}\t"
              f"nsoc={trace[it - 1].get("nbsoccuts", 0)}")


def viewgraphics(instname, traces, pngfile: Path = None, gc=False):
    lbs = {}
    gcs = {}
    cpus = {}
    for m, trace in traces.items():
        l, c, g = parsetrace(m, trace)
        if l:
            lbs[m] = l
            gcs[m] = g
            cpus[m] = c
    if gc:
        trace_progresses(gcs, cpus, instname, pngfile, y1label="gc", g1label="gap closed")
    else:
        trace_progresses(lbs, cpus, instname, pngfile)


def parsetrace(m, trace):
    lbs = None
    cpus = None
    gcs = None
    cpub = trace[0]["cpub"]
    if m.startswith("SDP"):
        assert len(trace) == 1
        niter = trace[0]["niter"]
        cpuit = trace[0]["cpu"] / niter
        if trace[0]["status"]:
            lbs = [(trace[0]["cpu"] + cpub, trace[0]["lb"])]
            gcs = [(trace[0]["cpu"] + cpub, 1)]
        cpus = [(0, cpub), (cpub, cpub)] + [(cpub + (k+1) * cpuit, cpuit) for k in range(niter)]
    elif len(trace) > 1 or trace[0]["status"] in (0, 1):
        lbs = [(t["cpu"] + cpub, t["lb"]) for t in trace.values()]
        gcs = [(t["cpu"] + cpub, t["gc"]) for t in trace.values()]
        cpus = [(0, cpub), (cpub, cpub)] + [(t["cpu"] + cpub, t["cpuit"]) for t in trace.values()]
    return lbs, cpus, gcs


if __name__ == "__main__":

    ### collections =
    # BOXQP, small/large/larger/all, instance number (1,2, or 3), e.g: "BOXQP_larger_2"
    # SPCA, small/large/all, sparsity measure K=int, e.g: "SPCA_large_K=5"

    ### cutting-plane method = RELAXATION-GAUGE-RANKS
    # relaxation = LP/SOC, gauge cut = EIG (inf-norm)/ FROB (2-norm)
    # cut rank = max number of negative eigenvalues considered in the cut, ex with LP-EIG:
    # 0 =  all eigenvalues (aka PROJ)
    # 1 =  the most negative eigenvalue (aka EIG)
    # K: int = the K most negative eigenvalues  (aka LOWRANK-K)
    # K1+K2+K3 =  3 cuts/it with rank = K1, K2 and K3
    # K1|K2|K3 =  1 cut/it with rank randomly chosen btw K1, K2 and K3
    # default = 1 cut/it with rank decreasing dynamically
    # S = the negative eigenvalues between lmin (the smallest one) and lmin/2

    # modes = ("LP-EIG-1", "LP-FROB-0", "LP-FROB-S", "SOC-EIG-1", "SOC-FROB-0")
    # name = "arrhythmia_274"
    # testinst(si.spca(name, k=10), name, modes=modes, maxit=-1, maxcpu=50, eigtol=1e-6, graphics=1)

    # name = "030-100-1"
    # testinst(si.boxqpnew(name), name, modes=modes, maxit=-1, maxcpu=-2, eigtol=1e-4, graphics=1)

    modes = ["SOC-EIG-0"]
    testbench("SPCA_large_K=10", modes=modes, maxit=-1, maxcpu=1800, eigtol=1e-6, graphics=2)
    # testbench("BOXQP_largest_3", modes=modes, maxit=-1, maxcpu=1800, eigtol=1e-6, graphics=2)
    # testbench("BOXQP_smallest", modes=modes, maxit=-1, maxcpu=60, eigtol=1e-6, graphics=2)
