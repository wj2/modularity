
import numpy as np
import tensorflow as tf
import itertools as it
import functools as ft
import scipy.stats as sts

import sklearn.svm as skm
import sklearn.multioutput as skmo
import sklearn.model_selection as skms

import general.utility as u
import general.tf.networks as gtf
import general.tasks.classification as gtc


def make_coloring_task_group(n_vars, n_groups=2, n_tasks=10, share_vars=False):
    if share_vars:
        t_inds = np.repeat(np.arange(n_vars).reshape((1, -1)), n_groups, axis=0)
    else:
        t_inds = np.arange(n_vars*n_groups).reshape((n_groups, -1))

    tasks = []
    for i in range(n_tasks):
        tasks_i = []
        for t_j in t_inds:
            tasks_i.append(gtc.ColoringTask(t_j))
        tasks.append(gtc.CompositeTask(gtc.parity, *tasks_i))
    return gtc.TaskGroup(*tasks)


def analyze_information(network, n_samps=500, decoder=skm.LinearSVC):
    stim, inp_rep, lrs = network.sample_layer_reps(n_samps, add_noise=True)
    network_targets = network.get_target(stim)

    sub0_tasks = gtc.TaskGroup(*list(t.tasks[0] for t in network.tasks.tasks))
    sub1_tasks = gtc.TaskGroup(*list(t.tasks[1] for t in network.tasks.tasks))
    s0_targ = sub0_tasks(stim)
    s1_targ = sub1_tasks(stim)
    targets = (network_targets, s0_targ, s1_targ)
    corrs = {}
    for i, lr in enumerate(lrs):
        lr = np.array(lr)
        for j, targ in enumerate(targets):
            corr_ij = np.zeros(targ.shape[1])
            for k in range(targ.shape[1]):
                m = decoder(dual="auto", max_iter=5000)
                out = skms.cross_validate(m, lr, targ[:, k], cv=10)
                corr_ij[k] = np.mean(out["test_score"])
            corrs[(i, j)] = corr_ij
    return corrs


class CompositionalNetwork(gtf.GenericFFNetwork):
    def __init__(self, *args, tasks=None, **kwargs):
        if tasks is None:
            tasks = gtc.ParityTask(n_task_vars)
        super().__init__(*args, tasks=tasks, **kwargs)




    
