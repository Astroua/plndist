
'''
Implementation in scipy form of the Double Pareto-Lognormal Distribution
'''

import numpy as np
from scipy.stats import rv_continuous, norm


def _pln_pdf(x, alpha, nu, tau2):
    A1 = np.exp(alpha * nu + alpha ** 2 * tau2 / 2)
    fofx = alpha * A1 * x ** (-alpha - 1) *\
        norm.cdf((np.log(x) - nu - alpha * tau2) / np.sqrt(tau2))
    return fofx


def _pln_cdf(x, alpha, nu, tau2):
    A1 = np.exp(alpha * nu + alpha ** 2 * tau2 / 2)
    term1 = norm.cdf((np.log(x) - nu) / np.sqrt(tau2))
    term2 = x ** (-alpha) * A1 * \
        norm.cdf((np.log(x) - nu - alpha * tau2) / np.sqrt(tau2))
    return term1 - term2


def _pln_logpdf(x, alpha, nu, tau2):
    return np.log(alpha) + alpha * nu + alpha * tau2 / 2 - \
        (alpha + 1) * np.log(x) + \
        norm.logcdf((np.log(x) - nu - alpha * tau2) / np.sqrt(tau2))


class pln_gen(rv_continuous):

    def _pdf(self, x, alpha, nu, tau2):
        return _pln_pdf(x, alpha, nu, tau2)

    def _logpdf(self, x, alpha, nu, tau2):
        return _pln_logpdf(x, alpha, nu, tau2)

    def _cdf(self, x, alpha, nu, tau2):
        return _pln_cdf(x, alpha, nu, tau2)

pln = pln_gen(name="pln")
