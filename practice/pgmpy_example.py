# Starting with defining the network structure
from pgmpy.models import BayesianModel
from scipy.stats import uniform
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.special import beta
from pgmpy.factors.continuous import ContinuousFactor
import numpy as np
import matplotlib.pyplot as plt


cancer_model = BayesianModel([('Pollution', 'Cancer'),
                              ('Smoker', 'Cancer'),
                              ('Cancer', 'Xray'),
                              ('Cancer', 'Dyspnoea')])

# Now defining the parameters.
from pgmpy.factors.discrete import TabularCPD

cpd_poll = TabularCPD(variable='Pollution', variable_card=2,
                      values=[[0.9], [0.1]])
print(cpd_poll)
cpd_smoke = TabularCPD(variable='Smoker', variable_card=2,
                       values=[[0.3], [0.7]])
print(cpd_smoke)
cpd_cancer = TabularCPD(variable='Cancer', variable_card=2,
                        values=[[0.03, 0.05, 0.001, 0.02],
                                [0.97, 0.95, 0.999, 0.98]],
                        evidence=['Smoker', 'Pollution'],
                        evidence_card=[2, 2])
print(cpd_cancer)
cpd_xray = TabularCPD(variable='Xray', variable_card=2,
                      values=[[0.9, 0.2], [0.1, 0.8]],
                      evidence=['Cancer'], evidence_card=[2])
print(cpd_xray)
cpd_dysp = TabularCPD(variable='Dyspnoea', variable_card=2,
                      values=[[0.65, 0.3], [0.35, 0.7]],
                      evidence=['Cancer'], evidence_card=[2])
print(cpd_dysp)

# Associating the parameters with the model structure.
cancer_model.add_cpds(cpd_poll, cpd_smoke, cpd_cancer, cpd_xray, cpd_dysp)

# Checking if the cpds are valid for the model.
print(cancer_model.check_model())

# Doing some simple queries on the network
print(cancer_model.is_active_trail('Pollution', 'Smoker'))

print(cancer_model.is_active_trail('Pollution', 'Smoker', observed=['Cancer']))

print(cancer_model.local_independencies('Xray'))

print(cancer_model.get_independencies())


def uniform_pdf(u, v):
    return uniform.pdf(x=[u, v], loc=(0,0), scale=(200, 300))


def custom_pdf(x, y, z):
    return z*(np.power(x, 1)*np.power(y, 2))/beta(x, y)


def drichlet_pdf(x, y):
    return (np.power(x, 1) * np.power(y, 2)) / beta(x, y)


def normal_pdf(x1, x2):
    return multivariate_normal.pdf((x1, x2), [0, 0], [[1, 0], [0, 1]])


uniform_factor = ContinuousFactor(['g_x', 'g_y'], uniform_pdf)
custom_factor = ContinuousFactor(['x', 'y', 'z'], custom_pdf)
drichlet_factor = ContinuousFactor(['x', 'y'], drichlet_pdf)
multigauss_factor = ContinuousFactor(['x1', 'x2'], normal_pdf)

print(uniform_factor.scope(), uniform_factor.assignment(0.5, 0.5))
print(custom_factor.scope(), custom_factor.assignment(0.0, 0.0, 0.0))
print(drichlet_factor.scope(), drichlet_factor.assignment(0.0, 0.0))
print(multigauss_factor.scope(), multigauss_factor.assignment(0.0, 0.0))

print(uniform_pdf(0.5, 0.5))
print(custom_pdf(0.0, 0.0, 0.0))
print(drichlet_pdf(0.0, 0.0))
print(normal_pdf(0.0, 0.0))

fig, ax = plt.subplots(1, 1)
mean, var, skew, kurt = uniform.stats(moments='mvsk')
x = np.linspace(uniform.ppf(0.01), uniform.ppf(0.99), 100)
ax.plot(x, uniform.pdf(x), 'r-', lw=5, alpha=0.6, label='uniform pdf')
rv = uniform()
ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
vals = uniform.ppf([0.001, 0.5, 0.999])
np.allclose([0.001, 0.5, 0.999], uniform.cdf(vals))
r = uniform.rvs(size=1000)
ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()