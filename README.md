# README
Code and experiments for the paper:

SPECTRAL-GAUGE CUTS FOR SEMIDEFINITE PROGRAMMING (2026)
ANTONIO SASAKI, SOPHIE DEMASSEY, AND VALENTINA SESSA
CMA, MINES-PARIS-PSL

----
## Benchmarks and SDP relaxations

Comparison of SDP models (in Mosek 11.2.2) with LP or SOC relaxations (in Gurobi 13.0.2) augmented with spectral-gauge cuts.
Model implementation:

#### SDP:

given symmetric matrices $C \in S^n$, polyedron $P\subseteq S^n$

$$\min_{x\in S^n} \langle C, X \rangle : X\in P, X\succeq 0$$
###### SOC relaxation: 
1x1 and 2x2 principal submatrices of $X\succeq 0$ are PSD, and their determinants are nonnegative
 $\implies$ $X_{ii}\ge 0$, $X_{ii}X_{jj} - X_{ij}^2\ge 0$
 
$$\min \sum_j C_{jj}X_{jj} + 2\sum_{i<j} C_{ij}X_{ij} : X\in P, X_{jj}\ge 0, X_{ij}^2\le X_{ii}X_{jj} \quad \forall j, i<j$$

###### LP relaxation:
AM-GM inequality on SOC model:
$0\le (X_{ii}- X_{jj})^2=(X_{ii}+X_{jj})^2 - 4X_{ii}X_{jj}$ with $X_{ii},X_{jj}\ge 0$ $\implies X_{ii}+X_{jj}\ge 2\sqrt{X_{ii}X_{jj}}\ge 2|X_{ij}|\ge \pm 2X_{ij}$

$$\min \sum_j C_{jj}X_{jj} + 2\sum_{i<j} C_{ij}X_{ij} : X\in P, X_{jj}\ge 0, X_{ii}+X_{jj}\pm 2X_{ij}\ge 0\quad \forall j, i<j$$


#### Sparse principal component analysis:

$$\max_{y\in R^n}\ y^\top Cy : y^\top y=1, \|y\|_0\le k$$


###### SDP relaxation: 
$X=yy^\top\implies X\succeq 0$ and $tr(X)=y^\top y=1$ $\implies$ 
$\|X\|_F = \sqrt{tr(X^\top X)} = \sqrt{tr(yy^\top yy^\top)}=\sqrt{tr(X)}=1$

$\|x\|_1\le \sqrt{\|x\|_0}\|x\|_2$ $\forall x\in R^n$ $\implies$ $\|X\|_1\le \sqrt{\|X\|_0}\|X\|_F \le \sqrt{k^2}\|X\|_F=k$

Dropping condition $rank(X)=1$ and relaxing $\|y\|_0\le k$ by $\|X\|_1\le k$, we get the SDP relaxation:

$$\min_{X\in S^n} \langle -C, X \rangle : tr(X)=1, \|X\|_1\le k$$

###### norm-1 model:
- attach new nonnegative variables $Y\ge 0$ and $\sum Y_{ij}\le k$ 
- gurobi:  $Y_{jj}=X_{jj}$, $Y_{ij}\ge X_{ij}$, $Y_{ij}\ge -X_{ij}$ $(\forall j, i<j)$
- mosek: $diag(Y)=diag(X)$, $(Y_{ij},X_{ij})$ inQcone, $(Y_{ji},X_{ji})$ inQcone $(\forall j, i<j)$

###### ref:
- d’Aspremont et al. (2005) *A direct
formulation for sparse pca using semidefinite programming*
- Bertsimas & Cory-Wright (2019) *On Polyhedral and Second-Order Cone Decompositions of Semidefinite Optimization Problems*



#### BoxQP:


$$\min_{y\in R^n}\ y^\top Qy + b^\top y : 0\le y\le 1$$

###### SDP relaxation: 
$Z=yy^\top$ $\implies$ $Z-yy^\top\succeq 0$ drop condition $rank(Z)=1$

Schur's complement: let $X=\big((Z,y),(y^\top, 1)\big)$ then $Z-yy^\top\succeq 0 \iff X\succeq 0$. 

Furthermore $\langle C, X \rangle = y^\top Qy + b^\top y$ given  $C=\big((Q/2, b/2),(b^\top/2, 0)\big)$  

$$\min_{X\in S^{n+1}}\ \langle C, X \rangle: 0\le X\le 1, X_{nn}=1$$

###### McCormick inequalities: 
$0\le(1-y_i)(1-y_j)=Z_{ij}-(y_i+y_j-1)$ and $0\le(1-y_i)y_j=Z_{ij}-y_i$ then

$$MC(X): X_{in}+X_{jn}-1 \le X_{ij}\le \min(X_{in},X_{jn}) \quad \forall i\le j<n$$

###### LP relaxation:
- $X_{ii}+X_{jj}+2 X_{ij}\ge 0$ is redundant with $X\ge 0$
- $X_{ii}+X_{jj}- 2 X_{ij}\ge 0$ is stronger than $MC(X)$, for example $X_{in}=X_{jn}=X_{ij}=1/2$ and $X_{ii}=X_{jj}=0$ satisfies $MC(X)$ but not $LP$
- $X_{ij}^2\le \min(X_{ii},X_{jj})$ is redundant with SOC and $u=1$ 


## spectral-gauge cuts

	# gauges
		# EIG = infinite-norm 
		# FROB = euclidean-norm
	# cut rank = max number of negative eigenvalues and associated eigenvectors considered in the cut:
    	# 0 =  all eigenvalues
    	# K: int = the K most negative ones
    	# K1+K2+K3 =  3 cuts/iteration with ranks = K1, K2 and K3
    	# K1|K2|K3 =  1 cut/iteration with rank randomly chosen btw K1, K2 and K3
    	# default = 1 cut/iteration with rank decreasing dynamically
    	# S = the most negative eigenvalues between lmin and 90%.lmin
	# mode = RELAXATION-GAUGE-RANK, example (REL being either LP or SOC):
		# REL-EIG-1: eigencut
		# REL-EIG-0: nuclear cut
		# REL-FROB-0: frobenius cut
		# REL-FROB-S: low-rank frobenius cut

----
## code structure

- runcuts.py: ```testinst``` and ```testbench``` to run different SDP/LP/QP relaxations on different benchmark collections. see usage below.
- sdp.py: SDP models in Mosek
- cuts.py: LP/SOC models and gauge cuts in Gurobi
- genmatrix.py: generate or parse matrices from collections
- inout.py: generate SDP instances from matrices

----
## usage examples:

see runcuts.py

#### ex1
    testbench("SPCA_large_K=5", modes=("SOC-EIG-1", "LP-FROB-0"), maxit=-1, maxcpu=100, eigtol=1e-5, graphics=2)
    
- "SPCA\_large\_K=5": test all the largest SPCA instances with sparsity k=5
- run 2 cut templates on each instance: SOC relaxation with eigencuts, LP relaxation with frobenius cuts
- halt condition for each run: cpu=100s or smallest eigenvalue > -1e-5
- graphics=2: draw graphics and save all result files in a directory

#### ex2

    testbench("BOXQP_small_1", modes=("SDP","LP-EIG-0"), maxit=20, maxcpu=-1, eigtol=1e-3, graphics=0)
    
- "BOXQP\_small\_1": the smallest Burer's boxQP instances indexed with `_1`
- halt condition: nb of iterations=20 or smallest eigenvalue > -1e-3
- graphics=0: generate only one summary result file
