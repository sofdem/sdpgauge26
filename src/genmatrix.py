#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Antonio Sasaki, Sophie Demassey, Valentina Sessa

generate and parse matrices
"""
from pathlib import Path
import numpy as np
import gzip

BASEDIR: Path = Path("../input/")
BOXQP_DIR: Path = BASEDIR / "boxqp"
BOXQPNEW_DIR: Path = BASEDIR / "boxqp" / "newsmall"
SPCA_DIR: Path = BASEDIR / "spca"

OUTDIR: Path = Path("../output/")
OUTDIR.mkdir(parents=True, exist_ok=True)

spca = {"small": ("pitprops_13", "wine_13", "ionosphere_34", "lung_54", "geography_68", "communities_101"),
        "large": ("arrhythmia_257", "voice_310", "gait_320", "gastro_466", "parkinson_754", "micromass_1139")}
spcamp = {"large": ["arrhythmia_257", "arrhythmia_274", "gastro_466"]}
boxqpf = {
    "smallest": {30: [25, 50, 75, 100]},
    "small": {20: [100], 30: range(60, 101, 10), 40: range(30, 101, 10), 50: (30, 40, 50), 60: [20]},
    "large": {n: [25, 50, 75] for n in [70, 80, 90, 100, 125]},
    "larger": {n: [25, 50, 75, 100] for n in range(150, 251, 25)},
    "largest": {250: [25, 50, 75, 100]}}
boxqpall = dict(p for d in boxqpf.values() for p in d.items())


def spca_select(size="small") -> list[Path]:
    if spca.get(size):
        return [Path(SPCA_DIR / f"sigma_{f}.csv") for f in spca[size]]
    return getallfiles(SPCA_DIR, "*.csv")


def boxqpfile(n, d, i, newsmall=False) -> Path:
    fname = f"spar{str(n).rjust(3, '0')}-{str(d).rjust(3, '0')}-{i}.in"
    dirname = BOXQPNEW_DIR if newsmall else BOXQP_DIR
    return Path(dirname / fname)


def qp_select(size="small", suf="123") -> list[Path]:
    boxqpdict = boxqpf.get(size, boxqpall)
    sample = [int(a) for a in suf]
    newsmall = (size == "smallest")
    return [boxqpfile(n, d, i, newsmall) for n, dlist in boxqpdict.items() for d in dlist for i in sample]


def getallfiles(dirname: Path, ext: str) -> list[Path]:
    assert dirname.is_dir(), f"folder not found: {dirname}"
    files = sorted(p for p in dirname.glob(ext) if p.is_file())
    assert files, f"no {ext} files found in {dirname}"
    return files


def load_csv_as_sigma(path: Path) -> np.ndarray:
    """ load a matrix in .csv format as a dense numpy array """
    A = np.loadtxt(path, delimiter=',')
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"matrix not square in {path}")
    return np.asarray(A, dtype=float)


def parsebench(collection: str):
    size = next((k for k in ("smallest", "small", "largest", "larger", "large") if k in collection.lower()), "all")

    if collection.startswith("BOXQP"):
        suf = collection.lower().split('_')[-1]
        return qp_select(size, suf) if suf.isnumeric() else qp_select(size)

    if collection.startswith("SPCA"):
        return spca_select(size)


def openfile(filename: Path):
    if filename.suffix == ".gz":
        return gzip.open(filename, "r")
    return open(filename, "r")


def parseboxqpinfile(filename):
    """ parse Box QP instance from file in the IN format (https://github.com/sburer/BoxQP_instances):
    Problem is max  0.5*x'*Q*x + c'*x s.t. 0 <= x <= e
    File format is n, [c'], [Q]
    min_{x PSD} <c, x> : <a_k, x> = b_k for k in [m] with matrices a_k are block diagonals with sparse blocks
    """
    with openfile(filename) as f:
        n = int(f.readline())
        b = np.fromstring(f.readline().rstrip(), dtype=float, sep=' ')
        assert b.shape[0] == n
        f.close()
    c = np.loadtxt(filename, skiprows=2, dtype=float, delimiter=' ', usecols=range(n))
    assert c.shape[0] == c.shape[1] == n, c.shape
    return c, b
