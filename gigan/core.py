# -*- coding: utf-8 -*-
from . import helpers
import numpy as np
from pgmpy.models import BayesianModel
from gigan.extensions.pgmpy import RandomChoice

import sgan.utils

def get_hmm():
    """Get a thought."""
    return 'hmmm...'


def hmm():
    """Contemplation..."""
    if helpers.get_answer():
        print(get_hmm())


def metropolis_hastings(likelihood_computer, prior, transition_model, param_init, iterations, data, acceptance_rule):
    """
    Metropolis-Hastings uses transition_model to randomly walk in the distribution space, accepting or rejecting jumps
    to new positions based on how likely the sample is. This “memory-less” random walk is the “Markov Chain” part of
    MCMC, which is a type of algorithm for sampling from a probability distribution and can be used to estimate the
    distribution of parameters given a set of observations.
    The likelihood of each new sample is decided by a function f() that must be proportional to the posterior we want to
    sample from. f is commonly chosen to be a probability function that expresses this proportionality.

    :param likelihood_computer: P(Y|X) returns the likelihood that these parameters generated the data Y
    :param prior: P(X) measures how likely is parameter X regardless of evidence Y data.
    :param transition_model: proposal distribution Q(X'|X) that draws samples from an intractable posterior
                            distribution P(X|Y) that we wish to compute.
    :param param_init: a starting sample for parameters
    :param iterations: number of accepted samples to generate
    :param data: the data Y that we wish to model
    :param acceptance_rule: (x,x_new) decides whether to accept or reject the new sample
    :return: Array of accepted samples
    """
    x = param_init
    accepted = []
    rejected = []
    for i in range(iterations):
        x_new = transition_model(x)  # X_{i+1} = Q(X_i)
        x_lik = likelihood_computer(x, data)  # P(Y|X_i)
        x_new_lik = likelihood_computer(x_new, data)  # P(Y|X_{i+1})
        if acceptance_rule(x_lik + np.log(prior(x)), x_new_lik + np.log(prior(x_new))):
            x = x_new
            accepted.append(x_new)
        else:
            rejected.append(x_new)
    return np.array(accepted), np.array(rejected)


def cascading_resimulation_mh(graph_model: BayesianModel, iterations: int, data, acceptance_rule):
    """
        Cusumano's Goal Inference (2017) presented a Cascading Resimulation Metropolis-Hastings strategy.
        TODO: change description
        uses transition_model to randomly walk in the distribution space, accepting or rejecting jumps
        to new positions based on how likely the sample is. This “memory-less” random walk is the “Markov Chain” part of
        MCMC, which is a type of algorithm for sampling from a probability distribution and can be used to estimate the
        distribution of parameters given a set of observations.
        The likelihood of each new sample is decided by a function f() that must be proportional to the posterior we
        want to sample from. f is commonly chosen to be a probability function that expresses this proportionality.

        :param graph_model: Bayesian graphical model as a Pgmpy object.
        :param likelihood_computer: P(Y|X) returns the likelihood that these parameters generated the data Y
        :param prior: P(X) measures how likely is parameter X regardless of evidence Y data.
        :param transition_model: proposal distribution Q(X'|X) that draws samples from an intractable posterior
                                distribution P(X|Y) that we wish to compute.
        :param param_init: a starting sample for parameters
        :param iterations: number of accepted samples to generate
        :param data: the data Y that we wish to model
        :param acceptance_rule: (x,x_new) decides whether to accept or reject the new sample
        :return: Array of accepted samples
        """
    accepted = []
    rejected = []
    G = graph_model
    root_nodes = G.get_roots()
    ith_node: RandomChoice = root_nodes[0]
    for i in range(iterations):
        # 1 propose a new value for choice i
        z = ith_node.samples[ith_node.samples.count() - 1]
        z_new = ith_node.transition_model(z)
        # 2 Initially, no change to other choices
        # z'_I\i <-- z_I\i
        # 3 Unnormalized target density for previous values
        z_lik = ith_node.likelihood(z, data)  # P(Y|X_i)
        # 4 Unnormalized target density for proposed values
        z_new_lik = ith_node.likelihood(z_new, data)  # P(Y|X_{i+1})
        # 5 Ask for likelihoods from j \in B
        B = [ith_node, G.get_children(ith_node)]
        # 6 Likelihood-free cascade participants
        H = []
        # 7 Visited choices with tracktable likelihoods
        A = [ith_node]
        while len(B) > 0:
            jth_node = B.pop()   # Pop in topological order
            if jth_node.likelihood is None:  # Choice j is likelihood-free
                # Propose from prior
                # z'_j ~ prior(*;f_j(z_p))
                # Ask for child likelihoods
                H = H
            else:
                # likelihood
                z_lik = 0
                # new likelihood
                z_new_lik = 0
                A = A
        # MH ratio
        if acceptance_rule(z_new_lik/z_lik):
            z = z_new
            accepted.append(z_new)
        else:
            rejected.append(z_new)
    return np.array(accepted), np.array(rejected)