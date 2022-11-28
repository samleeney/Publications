#| # Bayesian RFI Mitigation - Example
#| This is a simple script showing how to integrate a likelihood capable of correcting for RFI into a Bayesian data analysis pipeline.

#| First, generate some mock data.

import numpy as np
import matplotlib.pyplot as plt
N = 25
x = np.linspace(0, 25, N)
m = 1
c = 1
sig = 5
y = m * x + c + np.random.randn(N) * sig
plt.plot(x, y, 'o')

#| Add some RFI

y[10] += 100
y[15] += 100
plt.plot(x, y, 'ro')

#| Now, define a traditional likelihood function and fit the data without modeling the RFI.

def likelihood(theta):
    m=theta[0]
    c=theta[1]
    sig=theta[2]
    y_=m * x + c
    return (-(y_-y)**2/sig**2/2 - np.log(2*np.pi*sig**2)/2).sum(), []

#| Now, we'll define a likelihood function that can correct for RFI.
#| Note the difference between the two likelihoods. Notice the condition imposed on the likelihood by `emax'.

def rfi_corrected_likelihood(theta):
    m=theta[0]
    c=theta[1]
    sig=theta[2]
    y_= m * x + c
    logL=-(y_-y)**2/sig**2/2 - np.log(2*np.pi*sig**2)/2 + np.log(1-p)
    emax = logL > logp - np.log(delta)
    logPmax=np.where(emax, logL, logp - np.log(delta)).sum()
    return logPmax, []

#| Definte a prior. Notice that the prior range which encapsulates the full range of possible values from the data as defined by delta.

from pypolychord.priors import UniformPrior

def prior(hypercube):
    theta = np.zeros_like(hypercube)
    theta[0]=UniformPrior(-delta/(np.max(x)-np.min(x)), delta/(np.max(x)-np.min(x)))(hypercube[1])  # m
    theta[1] = UniformPrior(-delta, delta)(hypercube[0])  # c
    theta[2]=UniformPrior(0, delta)(hypercube[2])  # sig
    return theta

#| Set $p$ (the probability thresholding term) and $\Delta$ (the length scale in units of data)

delta = np.max(y)
logp = -2.5
p = np.exp(logp)

#| Fit with a Bayesian numerical solver. We use the Nested Sampler Polychord but any others will also work.
#| Polychord settings.

import pypolychord
from pypolychord.settings import PolyChordSettings

nDims=3
nDerived=0
settings=PolyChordSettings(nDims, nDerived)
settings.nlive=200
settings.read_resume=False

#| We first fit the data using the traditional likelihood

settings.file_root='rfi_nocorr'
output=pypolychord.run_polychord(
likelihood, nDims, nDerived, settings, prior)

#| Process the results using anesthetic. Again, any other chains evaluator could be used here.

from anesthetic import NestedSamples
norfi_nocorr=NestedSamples(
    root = './chains/rfi_nocorr', columns = ['$m$', '$c$', r'$\sigma$'])
fig, ax=norfi_nocorr.plot_2d(['$m$', '$c$', r'$\sigma$'],
                               label = 'RFI No Correction', alpha = 0.6)

#| Notice that $\sigma$ is estimated completely wrong, and the confidence in the other parameters is low.

#| Fit the data, this time using the correcting likelihood

settings.file_root='rfi_corr'
output=pypolychord.run_polychord(
    rfi_corrected_likelihood, nDims, nDerived, settings, prior)

#| Compare the results,

from anesthetic import NestedSamples
rfi_corr=NestedSamples(
    root = './chains/rfi_corr', columns = ['$m$', '$c$', r'$\sigma$'])
fig, ax=norfi_nocorr.plot_2d(['$m$', '$c$', r'$\sigma$'],
                               label = 'No RFI No Correction', alpha = 0.6)
rfi_corr.plot_2d(ax, label = 'RFI Corrected', alpha = 0.6)
plt.legend(loc='upper right', markerscale=5,
           bbox_to_anchor=(1.3, 1.6), fontsize=7)

#| # Key Point
#|
#| The vast majority of this example makes up the general 'pipeline' for simulating and analysing the data. Only three lines of code are modified (inside rfi_corrected_likelihood) to impliment the RFI correction.

