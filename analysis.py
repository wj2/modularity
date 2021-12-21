
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as skd
import itertools as it

import general.utility as u
import modularity.simple as ms

@u.arg_list_decorator
def train_variable_models(group_size, tasks_per_group, group_maker, model_type,
                          n_reps=2, **kwargs):
    out_ms = np.zeros((len(group_size), len(tasks_per_group), len(group_maker),
                       len(model_type), n_reps), dtype=object)
    print(out_ms.shape)
    for (i, j, k, l) in u.make_array_ind_iterator(out_ms.shape[:-1]):
        out_ms[i, j, k, l] = train_n_models(group_size[i], tasks_per_group[j],
                                            group_maker=group_maker[k],
                                            model_type=model_type[l],
                                            n_reps=n_reps, **kwargs)
    return out_ms    

def train_n_models(group_size, tasks_per_group, group_width=200, fdg=None,
                   n_reps=2, n_groups=5, group_maker=ms.random_groups,
                   model_type=ms.ColoringModularizer, epochs=5, verbose=False,
                   **training_kwargs):
    if fdg is None:
        use_mixer = False
    else:
        use_mixer = True
    inp_dim = fdg.input_dim
    out = []
    for i in range(n_reps):
         m_i = model_type(inp_dim, group_size=group_size, n_groups=n_groups,
                          group_maker=group_maker, use_dg=fdg,
                          group_width=group_width, use_mixer=use_mixer,
                          tasks_per_group=tasks_per_group)
         m_i.fit(epochs=epochs, verbose=verbose, **training_kwargs)
         out.append(m_i)
    return out

def correlate_clusters(groups, w_matrix):
    w = u.make_unit_vector(np.array(w_matrix).T)
    w_abs = np.abs(w)
    w_sort_inds = np.argsort(w_abs, axis=1).astype(float)
    sort_inds = u.make_unit_vector(w_sort_inds)
    w_sim = np.dot(sort_inds, sort_inds.T)
    w_sim[np.identity(w_sim.shape[0], dtype=bool)] = np.nan
    u_groups = np.unique(groups)
    n_groups = len(u_groups)
    overlap = np.zeros((n_groups, n_groups))
    for i in u_groups:
        i_mask = groups == i
        for j in u_groups:
            j_mask = groups == j
            m_ij = w_sim[i_mask]
            m_ij = m_ij[:, j_mask]
            overlap[i, j] = np.nanmean(np.abs(m_ij))
    mask = np.identity(n_groups, dtype=bool)
    avg_in = np.mean(overlap[mask])
    avg_out = np.mean(overlap[np.logical_not(mask)])
    return overlap, avg_in - avg_out

def threshold_clusters(groups, w_matrix, cumu_weight=.9):
    w_abs = np.abs(np.array(w_matrix).T)
    w_abs = w_abs/np.sum(w_abs, axis=1, keepdims=True)
    
    w_sim = np.zeros((w_abs.shape[0], w_abs.shape[0]))
    for i, j in it.product(range(w_abs.shape[0]), repeat=2):
        inds_i = np.argsort(w_abs[i])[::-1]
        wi_sum = np.cumsum(w_abs[i, inds_i])
        ind_set_i = set(inds_i[wi_sum < cumu_weight])
        inds_j = np.argsort(w_abs[j])[::-1]
        wj_sum = np.cumsum(w_abs[j, inds_j])
        ind_set_j = set(inds_j[wj_sum < cumu_weight])
        avg_len = np.mean([len(ind_set_i), len(ind_set_j)])
        w_sim[i, j] = len(ind_set_i.intersection(ind_set_j))/avg_len 
    u_groups = np.unique(groups)
    n_groups = len(u_groups)
    overlap = np.zeros((n_groups, n_groups))
    for i in u_groups:
        i_mask = groups == i
        for j in u_groups:
            j_mask = groups == j
            m_ij = w_sim[i_mask]
            m_ij = m_ij[:, j_mask]
            overlap[i, j] = np.nanmean(np.abs(m_ij))
    mask = np.identity(n_groups, dtype=bool)
    avg_in = np.mean(overlap[mask])
    avg_out = np.mean(overlap[np.logical_not(mask)])
    return overlap, avg_in - avg_out    

def quantify_clusters(groups, w_matrix, absolute=True):
    w = u.make_unit_vector(np.array(w_matrix).T)
    if absolute:
        w = np.abs(w)
    w_sim = np.dot(w, w.T)
    w_sim[np.identity(w_sim.shape[0], dtype=bool)] = np.nan
    u_groups = np.unique(groups)
    n_groups = len(u_groups)
    overlap = np.zeros((n_groups, n_groups))
    for i in u_groups:
        i_mask = groups == i
        for j in u_groups:
            j_mask = groups == j
            m_ij = w_sim[i_mask]
            m_ij = m_ij[:, j_mask]
            overlap[i, j] = np.nanmean(np.abs(m_ij))
    mask = np.identity(n_groups, dtype=bool)
    avg_in = np.mean(overlap[mask])
    avg_out = np.mean(overlap[np.logical_not(mask)])
    return overlap, avg_in - avg_out

def apply_clusters_model_list(ml, func=quantify_clusters, **kwargs):
    ml = np.array(ml)
    mats = np.zeros_like(ml, dtype=object)
    diffs = np.zeros_like(mats)
    for ind in u.make_array_ind_iterator(ml.shape):
        m = ml[ind]
        mat, diff = func(m.out_group_labels, m.model.weights[-2],
                         **kwargs)
        mats[ind] = mat
        diffs[ind] = diff
    return mats, diffs
