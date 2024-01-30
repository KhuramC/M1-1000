import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize


def gaussian(x, mean=0., stdev=1., pmax=1.):
    """Gaussian function. Default is the PDF of standard normal distribution"""
    x = (x - mean) / stdev
    return pmax * np.exp(- x * x / 2)


def pmf(K, P):
    return np.array([p if k else 1 - p for k, p in zip(K, P)])


def nll(p):
    return -np.sum(np.log(p))


data_config = np.array(((25, 15, 35),
                        (75, 15, 37),
                        (125, 10, 29),
                        (175, 2, 25)))

data = []
for d in data_config:
    dat = np.zeros((d[2], 2))
    dat[:, 0] = d[0]
    dat[:d[1], 1] = 1.
    data.append(dat)
data = np.vstack(data)


def negloglikelihood(params):
    return nll(pmf(data[:, 1], gaussian(
        data[:, 0], pmax=params[0], stdev=params[1], mean=20.)))


initParams = [.5, 50.]
results = minimize(negloglikelihood, initParams, method='Nelder-Mead')
params = results.x

print(params)

hist = np.array([(d[0], d[1] / d[2]) for d in data_config])
x = np.linspace(0, hist[-1, 0] + 50, 51)
plt.bar(hist[:, 0], hist[:, 1], width=50.)
plt.plot(x, gaussian(x, pmax=params[0], stdev=params[1]), 'r')
plt.show()
