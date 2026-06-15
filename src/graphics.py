#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sophie Demassey (sofdem@gmail.com), Antonio Sasaki, Valentina Sessa

stats and graphics
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

OUTDIR: Path = Path("../output/")

font1 = {'family': 'serif', 'color': 'black', 'size': 11}
font2 = {'family': 'serif', 'color': 'darkred', 'size': 10}


def trace_progress(ax, iters, xlabel, ylabel, glabel):
    maxx = max(its[-1][0] for its in iters.values())
    for mode, its in iters.items():
        y = [it[1] for it in its]
        x = [it[0] for it in its] if xlabel != "it" else list(range(len(y)))
        if xlabel != "it" and len(x) == 1:
            ax.plot([x[0], maxx], [y[0], y[0]], label=mode)
        else:
            ax.plot(x, y, label=mode)

    ax.set_title(glabel, fontdict=font1)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    return ax.get_legend_handles_labels()
    # ax.legend(loc="lower right")


def trace_progresses(lbs, cpus, instname: str, pngfile: Path = None, y1label="lb", g1label="lower bounds"):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"{instname}", fontsize=12)
    trace_progress(axs[0, 0], lbs, "cpu", y1label, g1label)
    trace_progress(axs[0, 1], lbs, "it", y1label, g1label)
    trace_progress(axs[1, 0], cpus, "cpu", "cpu/it", f"cpu per iteration")
    handles, labels = trace_progress(axs[1, 1], cpus, "it", "cpu/it", f"cpu per iteration")
    fig.legend(handles, labels, loc='outside right upper')
    for ax in axs.flat:
        ax.label_outer()

    if pngfile:
        fig.savefig(pngfile)
    else:
        plt.show()
    plt.close()


def gapclosed(csvfile):
    df = pd.read_csv(csvfile)
    # df.set_index(["matrix"], inplace=True)
    dfub = None  # parse_ubs(csvfile)
    # df = dfcsv if dfub is None else pd.concat([dfcsv, dfub])
    # df = dfcsv if dfub is None else dfcsv.join(dfub)
    # df.reset_index(inplace=True)
    # df.set_index(["matrix", "mode"], inplace=True)
    df.set_index(["mode"], inplace=True)
    df['dlb'] = df['lb'] - df['lb0']
    idx = set(df.index)
    ubname = "ub" if dfub is not None else next((k for k in ("SDP", "SDP_NOMC") if k in idx), "best")
    print(f"best = {ubname}")
    idxs = {i: next((k for k in idx if k.startswith(i)), None) for i in ("SOC", "LP")}
    lbidx = {i: (k, 'lb0') for (i, k) in idxs.items() if k} | {i: (i, 'lb') for i in idx}

    grouped = df.groupby(["matrix"])
    gcs = []
    for inst, group in grouped:
        if "boxqp" in csvfile:
            ii = tuple(int(k) for k in inst[0].replace("spar", "").split('-'))
            iii = {'n': ii[0], 'density': ii[1], 'i': ii[2]}
        else:
            iii = {'n': group.loc[('LP-EIG-1', 'n')], 'name': inst[0]}
        cpu = {'maxcpu': group.loc[('LP-EIG-1', 'maxcpu')]}
        vals = {i: group.loc[k] for (i, k) in lbidx.items()}
        vals['best'] = max(vals.values())
        if ubname == 'ub':
            vals['ub'] = -dfub.loc[inst, "ub"] if inst[0] in dfub.index else vals['best']
            if vals['ub'] < vals['best']:
                print(f"error ? ub={vals['ub']} < bestlb={vals['best']}")
                print(f"{inst[0]}: {vals}")
        ub = vals[ubname]
        lp = vals['LP']
        gcs.append(iii | {i: 100 * (v - lp) / (ub - lp) for (i, v) in vals.items()} | cpu)
    # print(m[list(idx)])
    print(f"{csvfile}\n gap closed: (LB - LP) / ({ubname} - LP):")
    print("-----------------")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)

    gcdf = pd.DataFrame(gcs)
    print(gcdf.drop(columns=['LP', ubname]))
    print("-----------------")

    if "boxqp" not in csvfile:
        return

    print(f"{csvfile}\n gap closed: (LB - LP) / ({ubname} - LP):")
    print("-----------------")

    grouped = gcdf.groupby(['n'])
    d = pd.concat([grouped.size(), grouped.mean()], axis=1)
    print(d.drop(columns=['LP', ubname, 'density', 'i']))

    grouped = gcdf.groupby(['density'])
    d = pd.concat([grouped.size(), grouped.mean()], axis=1)
    print(d.drop(columns=['LP', ubname, 'n', 'i']))


def lbathalftimesdp(dirname, limittime=-1, withsdp=True):
    assert dirname.is_dir(), f"folder not found: {dirname}"
    resfile = Path(f"{dirname}.csv")
    assert resfile.is_file(), f"file not found: {resfile}"
    dfr = pd.read_csv(resfile, dtype={"cpub": float, "cpu": float, "lb": float})
    modes = dfr["mode"].unique().tolist()
    sdpdf = dfr.loc[dfr['mode'] == "SDP"] if withsdp else dfr.loc[dfr['mode'] == modes[0]]
    sdpdf.set_index('matrix', inplace=True)
    sdpcpu = sdpdf.loc[:, 'cputotal'].to_dict()
    sdplb = sdpdf.loc[:, 'lb'].to_dict()
    print(sdpcpu)
    best = {}
    for inst, sdptime in sdpcpu.items():
        if "boxqp" in str(dirname):
            idx = tuple(int(k) for k in inst.replace("spar", "").split('-'))
        else:
            ii = inst.replace("sigma_", "").split('_')
            idx = (ii[0], int(ii[1]))

        best[idx] = {"sdpcpu": sdptime, "bestlb": -float("inf"), "res": {}}
        if withsdp:
            best[idx]["bestlb"] = sdplb[inst]
        files = sorted(p for p in dirname.glob(f"{inst}_*.csv") if p.is_file())
        for f in files:
            template = f.stem.split("_")[-1]
            if template == "SDP":
                continue
            df = pd.read_csv(f, dtype={"cpub": float, "cpu": float, "lb": float})
            t = sdptime if limittime == -1 else sdptime/2 if limittime == -2 else limittime
            cpu = t - df.loc[0, "cpub"]
            best[idx]["maxcpu"] = t
            closest = df.iloc[(df['cpu']-cpu).abs().argsort()[:1]]
            it = closest.iloc[0].name
            bcpu = df.loc[it, "cpu"]
            if bcpu > cpu:
                if it == 0:
                    print(f"{idx}-{template} not solved !! {df.loc[it, :]}")
                it -= 1
            if it >= 0:
                best[idx]["res"][template] = {k: df.loc[it, k] for k in ("lb", "gc", "cpu")} | {"it": it}
            lastlb = df.iloc[-1]["lb"]
            firstlb = df.loc[0, "lb"]
            if lastlb > best[idx]["bestlb"]:
                best[idx]["bestlb"] = lastlb
            if "lp0" not in best[idx].keys() and template.startswith("LP"):
                best[idx]["lp0"] = firstlb
    return best


def printgclatex(best, modes):
    n = 30
    print(modes)
    for d in (25, 50, 75, 100):
        s = f"30 &{d} &{3} "
        sdpcpu = 0
        res = {k: 0 for k in modes}
        for i in (1, 2, 3):
            b = best[(n, d, i)]
            sdpcpu += b["sdpcpu"]
            for k in res.keys():
                res[k] += b["res"][k]["gc"]
        s += f"&{sdpcpu/3:.1f} &"
        s += ' &'.join(f"{r*100/3:.0f}" for r in res.values()) + "\\\\"
        print(s)


def printgclatex2(best, modes):
    ns = (k[0] for k in best.keys())
    inst = {n: [(k[1], k[2]) for k in best.keys() if k[0] == n] for n in ns}
    print(inst)
    print(modes)
    for n, v in inst.items():
        nb = len(v)
        s = f"{n} &{v[0][0]} &{nb} "
        sdpcpu = 0
        res = {k: 0 for k in modes}
        for i in v:
            b = best[(n, i[0], i[1])]
            sdpcpu += b["sdpcpu"]
            for k in res.keys():
                res[k] += b["res"][k]["gc"]
        s += f"&{sdpcpu/nb:.1f} &"
        s += ' &'.join(f"{r*100/nb:.1f}" for r in res.values()) + "\\\\"
        print(s)


def printgclatex3(best, modes, n, t):
    print(modes)
    # for idx, b in best.items():
    #    bestlb = max(b["res"][m]["lb"] for m in modes)
    #    best[idx]["bestlb"] = bestlb
    for d in (25, 50, 75, 100):
        s = f"{n} &{d} &{3} "
        res = {k: 0 for k in modes}
        for i in (1, 2, 3):
            b = best[(n, d, i)]
            for k in res.keys():
                gc = (b["res"][k]["lb"] - b["lp0"]) / (b["bestlb"] - b["lp0"])
                res[k] += gc
        s += f"&{t:.0f} &"
        s += ' &'.join(f"{r*100/3:.1f}" for r in res.values()) + "\\\\"
        print(s)


def printgclatex4(best, modes, t, digit=1):
    print(modes)
    for idx, b in best.items():
        s = f"{idx[0]} &{idx[1]} "
        res = {}
        gcs = {k: round(100 * (b["res"][k]["lb"] - b["lp0"]) / (b["bestlb"] - b["lp0"]), digit) for k in b["res"]}
        maxgc = max(gcs.values())
        for k in modes:
            if k in gcs:
                gc = gcs[k]
                res[k] = str(gc) if gc < maxgc else f"{{\\bf {gc}}}"
            else:
                res[k] = "-"
        s += f"&{t:.0f} &"
        s += ' &'.join(r for r in res.values()) + "\\\\"
        print(s)


if __name__ == "__main__":

    modes = ("LP-EIG-1", "LP-EIG-0", "LP-FROB-0", "LP-FROB-S", "SOC-EIG-1", "SOC-FROB-0", "SOC-FROB-S")
    printgclatex4(lbathalftimesdp(Path(OUTDIR / "spca_large_k=10_260526"), limittime=60, withsdp=False), modes, 60)

    # modes = ("LP-EIG-1", "LP-EIG-0", "LP-FROB-0", "LP-FROB-S", "SOC-EIG-1", "SOC-EIG-0", "SOC-FROB-0", "SOC-FROB-S")
    # printgclatex4(lbathalftimesdp(Path("OUTDIR / "spca_small_k=10_260504-1117"), limittime=-2, withsdp=True), modes, -2)

    # modes = ("LP-EIG-1", "LP-EIG-0", "LP-FROB-0", "LP-EIG-S", "LP-FROB-S", "SOC-EIG-1", "SOC-EIG-0", "SOC-FROB-0")
    # printgclatex(lbathalftimesdp(Path("OUTDIR / "boxqp_smallest_260422-1747"), limittime=-1), modes)

    # modes = ("LP-EIG-1", "LP-EIG-0", "LP-FROB-0", "LP-EIG-S", "SOC-EIG-1", "SOC-FROB-0")
    # printgclatex2(lbathalftimesdp(Path("OUTDIR / "boxqp_small_260422-1911"), limittime=-1), modes)

    # modes = ("LP-EIG-1", "LP-EIG-0", "LP-FROB-0", "LP-FROB-S", "LP-EIG-0|50|20|10|5|1")
    # printgclatex3(lbathalftimesdp(Path("OUTDIR / "boxqp_largest_260422"), limittime=60, withsdp=False), modes, 250, 60)
