#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sophie Demassey (sofdem@gmail.com), Antonio Sasaki, Valentina Sessa

SDP instance: min_{x PSD[n], tr(x) <= t} <c, x> : <a_k, x> = b_k for k in [m]
represented as a tuple (n, m, c, a, b, t) with:
n: int = dimension of the squared matrices x, c, a_k
m: int = number of linear constraints <a_k, x> = b_k
c: np.ndarray(n, n) = symmetric matrix in S[n]
a: np.ndarray(m, n, n) = list of m symmetric matrices in S[n]
b: np.ndarray(m) = RHS list of m floats
t: float = trace upper bound


"""
import numpy as np
import genmatrix as gm

MAXINT = 1e5

CAT = ("SDP", "PCA", "SDPA")


def check_sdpinstance(snmcabt: tuple) -> (str, int, int, np.ndarray, np.ndarray, np.ndarray, float, float, float):
    """ check the format (dimensions and symmetry) of an SDP instance as a tuple """
    snmcabtlu = snmcabt + (-MAXINT, MAXINT) if len(snmcabt) == 7 else snmcabt
    s, n, m, c, a, b, t, l, u = snmcabtlu
    assert n == c.shape[0] == c.shape[1], f"n={n}, c shape={c.shape}"
    assert np.allclose(c, c.T)

    if m > 0:
        assert m == a.shape[0] == b.shape[0]
        assert n == a.shape[1] == a.shape[2]
        for ak in a:
            assert np.allclose(ak, ak.T)

    return s, n, m, c, a, b, t, l, u


def gen_sdp_from_pca(c: np.ndarray, k: int):
    """ generate a PCA instance min <-c,x>: tr(x)=1, l1(x) <= k """
    s = f"PCA{k}"
    n = c.shape[0]
    c = c # * (-1.0)
    return s, n, 0, c, None, None, k


def gen_sdp_fromqp(q: np.ndarray, b: np.ndarray, costfactor: float, nobounds=False):
    """ generate SDP-Schur relaxation min_{x PSD[n+1]} <c, x> = <q, z> + by: x=[(z,y),(y,1)], 0 <= x <= 1
    with c=[(q/2,b/2),(b/2, 0)]"""
    s = "QP"
    n = q.shape[0]
    q = q * (costfactor / 2)
    b = b * (costfactor / 2)
    a = np.c_[q, b]
    r = np.append(b.T, 0)
    c = np.r_[a, [r]]
    t = -1
    return (s, n+1, 0, c, None, None, t) if nobounds else (s, n+1, 0, c, None, None, t, 0, 1)


def tostr_sdpinstance(nmcabt: tuple):
    keys = ("s", "n", "m", "c", "a", "b", "t", "l", "u")
    nmcabtlu = check_sdpinstance(nmcabt)
    return "\n".join(f"{k} = {nmcabtlu[i]}" for i, k in enumerate(keys))


def readsdp(file, collection: str):
    nobounds = collection.endswith("NOBOUNDS")
    if collection.startswith("SPCA"):
        k = 10
        subs = collection.split("_")
        if len(subs) > 1 and subs[1].startswith("K="):
            k = int(subs[1][2:])
        c = gm.load_csv_as_sigma(file)
        return gen_sdp_from_pca(c, k)

    if collection.startswith("BOXQP"):
        c, b = gm.parseboxqpinfile(file)
        return gen_sdp_fromqp(c, b, -1.0, nobounds)


def spca(name, k=10):
    filename = f"sigma_{name}.csv"
    collection = f"SPCA={k}"
    return readsdp(gm.SPCA_DIR / filename, collection)


def boxqp(name):
    return readsdp(gm.BOXQP_DIR / f"spar{name}.in", "BOXQP")


def boxqpnew(name):
    return readsdp(gm.BOXQP_DIR / "newsmall" / f"spar{name}.in", "BOXQP")
