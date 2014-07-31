
'''
Implementation in scipy form of the Double Pareto-Lognormal Distribution
'''

import numpy as np
from scipy.stats import rv_continuous, norm


def _dpln_pdf(x, alpha, beta, nu, tau2):
    A1 = np.exp(alpha * nu + alpha**2 * tau2/2)
    A2 = np.exp(-beta*nu + beta**2 * tau2/2)
    term1 = A1 * x**(-alpha-1) * \
        norm.cdf((np.log(x)-nu-alpha * tau2)/np.sqrt(tau2))
    term2 = A2*x**(beta-1) * \
        norm.sf((np.log(x)-nu+beta*tau2)/np.sqrt(tau2))
    return alpha*beta/(alpha+beta)*(term2+term1)


def _dlpn_cdf(x, alpha, beta, nu, tau2):
    A1 = np.exp(alpha * nu + alpha**2 * tau2/2)
    A2 = np.exp(-beta*nu + beta**2 * tau2/2)

    term1 = norm.cdf((np.log(x) - nu) / np.sqrt(tau2))
    term2 = beta * x**-alpha * A1 * \
        norm.cdf((np.log(x)-nu-alpha * tau2)/np.sqrt(tau2))
    term3 = alpha * x**beta * A2 * \
        norm.sf((np.log(x)-nu+beta*tau2)/np.sqrt(tau2))

    return term1 - (alpha + beta)**-1 * (term2 + term3)


class dpln_gen(rv_continuous):
    def _pdf(self, x, alpha, beta, nu, tau2):
        return _dpln_pdf(x, alpha, beta, nu, tau2)

    def _logpdf(self, x, alpha, beta, nu, tau2):
        return np.log(_dpln_pdf(x, alpha, beta, nu, tau2))

    def _cdf(self, x, alpha, beta, nu, tau2):
        return _dlpn_cdf(x, alpha, beta, nu, tau2)


dpln = dpln_gen(name="dpln", a=0.0)
