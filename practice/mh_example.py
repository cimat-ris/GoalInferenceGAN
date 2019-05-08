from typing import Any, Callable, Union, List

import numpy as np
import scipy
import scipy.stats
import matplotlib.pyplot as plt
from gigan.core import metropolis_hastings

# --------------------------------------------------------------------------------
# STEP 1: DATA GENERATION
# We generate 30,000 samples from a normal distribution with
# $\mu$ = 10, and $\sigma$= 3, but we can only observe 1000 of them.
# --------------------------------------------------------------------------------
from numpy.core.multiarray import ndarray

mod1: Callable[[Any], Union[ndarray, int, float, complex]] = lambda t: np.random.normal(10, 3, t)

# Form a population of 30,000 individual, with average=10 and scale=3
population = mod1(30000)
# Assume we are only able to observe 1,000 of these individuals.
observation = population[np.random.randint(0, 30000, 1000)]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.hist(observation, bins=35, )
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
ax.set_title("Figure 1: Distribution of 1000 observations sampled from population of 30,000 with $\mu$=10, $\sigma$=3")
fig.show()
mu_obs = observation.mean()
print(mu_obs)

# --------------------------------------------------------------------------------
# STEP 2: What do we want?
# We would like to find a distribution for $\sigma_{obs}$ using the 1000 observed
# samples
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# STEP 3: Define the PDF and the transition model.
# From Figure 1, we can see that the data is normally distributed. The mean can
# be easily computed by taking the average of the values of the 1000 samples.
# By doing that, we get for example $\mu_{obs}=9.8$.
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# STEP 4: Define when we accept or reject $\sigma_{new}$
# --------------------------------------------------------------------------------

# --------------------------------------------------------------------------------
# STEP 5: Define the prior and the likelihood
# --------------------------------------------------------------------------------

# TRANSITION MODEL ( SAMPLE-ABLE DISTRIBUTION MODEL )

# The transition model defines how to move from sigma_current to sigma_new
transition_model: Callable[[Any], List[Union[Union[ndarray, int, float, complex], Any]]] \
    = lambda x: [x[0], np.random.normal(x[1], 0.5, (1,))]


def prior(x):
    # x[0] = mu, x[1]=sigma (new or current)
    # returns 1 for all valid values of sigma. Log(1) =0, so it does not affect the summation.
    # returns 0 for all invalid values of sigma (<=0). Log(0)=-infinity, and Log(negative number) is undefined.
    # It makes the new sigma infinitely unlikely.
    if x[1] <= 0:
        return 0
    return 1


# Computes the likelihood of the data given a sigma (new or current) according to equation (2)
def manual_log_like_normal(x, data):
    # x[0]=mu, x[1]=sigma (new or current)
    # data = the observation
    return np.sum(-np.log(x[1] * np.sqrt(2 * np.pi)) - ((data - x[0]) ** 2) / (2 * x[1] ** 2))


# Same as manual_log_like_normal(x,data), but using scipy implementation. It's pretty slow.
def log_lik_normal(x, data):
    # x[0]=mu, x[1]=sigma (new or current)
    # data = the observation
    return np.sum(np.log(scipy.stats.norm(x[0], x[1]).pdf(data)))


# Defines whether to accept or reject the new sample
def acceptance(x, x_new):
    if x_new > x:
        return True
    else:
        accept = np.random.uniform(0, 1)
        # Since we did a log likelihood, we need to exponentiate in order to compare to the random number
        # less likely x_new are less likely to be accepted
        return accept < (np.exp(x_new - x))


# STEP 6. Run the algorithm with initial parameters and collect accepted and rejected samples
accepted, rejected = metropolis_hastings(manual_log_like_normal, prior, transition_model, [mu_obs, 0.1], 50000,
                                         observation, acceptance)

print(accepted.shape)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(2, 1, 1)

ax.plot(rejected[0:50, 1], 'rx', label='Rejected', alpha=0.5)
ax.plot(accepted[0:50, 1], 'b.', label='Accepted', alpha=0.5)
ax.set_xlabel("Iteration")
ax.set_ylabel("$\sigma$")
ax.set_title("Figure 2: MCMC sampling for $\sigma$ with Metropolis-Hastings. First 50 samples are shown.")
ax.grid()
ax.legend()

ax2 = fig.add_subplot(2, 1, 2)
to_show = -accepted.shape[0]
ax2.plot(rejected[to_show:, 1], 'rx', label='Rejected', alpha=0.5)
ax2.plot(accepted[to_show:, 1], 'b.', label='Accepted', alpha=0.5)
ax2.set_xlabel("Iteration")
ax2.set_ylabel("$\sigma$")
ax2.set_title("Figure 3: MCMC sampling for $\sigma$ with Metropolis-Hastings. All samples are shown.")
ax2.grid()
ax2.legend()

fig.tight_layout()
fig.show()
print(accepted.shape)

show = int(-0.75 * accepted.shape[0])
hist_show = int(-0.75 * accepted.shape[0])

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 2, 1)
ax.plot(accepted[show:, 1])
ax.set_title("Figure 4: Trace for $\sigma$")
ax.set_ylabel("$\sigma$")
ax.set_xlabel("Iteration")
ax = fig.add_subplot(1, 2, 2)
ax.hist(accepted[hist_show:, 1], bins=20, density=True)
ax.set_ylabel("Frequency (normed)")
ax.set_xlabel("$\sigma$")
ax.set_title("Figure 5: Histogram of $\sigma$")
fig.tight_layout()
fig.show()

ax.grid("off")

#  PREDICTIONS

mu = accepted[show:, 0].mean()
sigma = accepted[show:, 1].mean()
print(mu, sigma)
model = lambda t, mu, sigma: np.random.normal(mu, sigma, t)
observation_gen = model(population.shape[0], mu, sigma)
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
ax.hist(observation_gen, bins=70, label="Predicted distribution of 30,000 individuals")
ax.hist(population, bins=70, alpha=0.5, label="Original values of the 30,000 individuals")
ax.set_xlabel("Mean")
ax.set_ylabel("Frequency")
ax.set_title("Figure 6: Posterior distribution of predicitons")
ax.legend()
fig.show()
