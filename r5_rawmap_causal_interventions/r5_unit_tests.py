# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import spearmanr

def top1(x):
    x=np.asarray(x,float).ravel()
    k=max(1,int(np.ceil(.01*len(x))))
    return np.partition(x,len(x)-k)[-k:].mean()
def q(x):
    x=np.asarray(x,float).ravel()
    return np.log((x.max()+1e-12)/(top1(x)+1e-12))

rng=np.random.default_rng(7)
x=np.abs(rng.normal(.2,.03,1000))
G=np.zeros(1000,bool); G[:20]=True
# Strength: top1 must not decrease when only defect coordinates are increased.
ts=[]
for lam in [0,.25,.5,1,2,4]:
    z=x.copy(); z[G]+=lam*.03; ts.append(top1(z))
assert np.all(np.diff(ts)>=-1e-12)

# Extent: implant fixed h > original max into increasing numbers of lowest coords.
h=x.max()+x.std()
order=np.argsort(x)
alphas=np.array([.001,.0025,.005,.01,.02,.05,.10])
qs=[]; tops=[]; maxs=[]
for a in alphas:
    k=max(1,int(round(a*len(x))))
    z=x.copy(); z[order[:k]]=h
    qs.append(q(z)); tops.append(top1(z)); maxs.append(z.max())
assert np.all(np.diff(tops)>=-1e-12)
assert np.allclose(maxs,maxs[0])
assert spearmanr(alphas,qs).statistic < 0

# Spatial permutation preserves histogram-only readouts exactly.
p=rng.permutation(len(x))
xp=x[p]
assert abs(top1(x)-top1(xp))<1e-12
assert abs(q(x)-q(xp))<1e-12
print("R5 intervention unit tests: PASS")
