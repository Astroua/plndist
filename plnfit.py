import numpy as np
import scipy.stats as ss
from scipy.stats import norm
from scipy.special import erf, erfc
import matplotlib.pyplot as plt
import scipy.optimize as opt
from pln_distrib import pln


def objfunc(x, p):
    if p[2] <= 0:
        return np.inf
    vals = -np.sum(pln.logpdf(x, *p))
    return vals

sampledata = np.exp(np.random.randn(1000) + 2)
params = opt.minimize(
    lambda p: objfunc(sampledata, p), [0, 0, 1], method='Powell')

p = params['x']
plt.hist(sampledata, normed=1, log=True)
trialx = np.linspace(sampledata.min(), sampledata.max(), 1000)
plt.plot(trialx, pln.pdf(trialx, *p))
