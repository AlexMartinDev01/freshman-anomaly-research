import numpy as np
def mean_top1p(x):
    x=np.asarray(x,float).ravel()
    k=max(1,int(np.ceil(.01*len(x))))
    return np.partition(x,len(x)-k)[-k:].mean()
def qshape(d):
    d=np.asarray(d,float)
    return np.log((d.max()+1e-12)/(mean_top1p(d)+1e-12))
rng=np.random.default_rng(0)
d=rng.uniform(.01,.7,5000)
assert abs(qshape(d)-qshape(7.3*d)) < 1e-10
print("TSN math unit test PASS")
