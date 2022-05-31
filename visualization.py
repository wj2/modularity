
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import general.utility as u 
import general.plotting as gpl
import modularity.analysis as ma

def _get_output_clusters(ws):
    idents = np.argmax(ws, axis=1)
    return idents

def _get_alg_clusters(ws, n_groups, alg):
    m = alg(n_groups)
    m.fit(ws)
    idents = m.predict(ws)
    return idents

def _get_cluster_order(ws, n_groups=None, alg=None):
    if alg is None:
        idents = _get_output_clusters(ws)
    else:
        idents = _get_output_clusters(ws, n_groups, alg)
    if n_groups is None:
        n_groups = len(np.unique(idents))
    tot = 0
    order = np.zeros(len(idents), dtype=int)
    for i in range(n_groups):
        inds = np.where(idents == i)[0]
        order[tot:tot + len(inds)] = inds
        tot = tot + len(inds)
    return order

geometry_metrics = ('shattering', 'within_ccgp', 'across_ccgp')
def plot_geometry_metrics(*args, geometry_names=geometry_metrics, **kwargs):
    return plot_clustering_metrics(*args, clustering_names=geometry_names,
                                   **kwargs)

clustering_metrics_all = ('cosine_sim_diffs',
                          'cosine_sim_absolute_diffs',
                          'threshold_diffs', 'brim')
clustering_metrics = ('brim', 'threshold_diffs')
def plot_clustering_metrics(df, x='tasks_per_group',
                            clustering_names=clustering_metrics,
                            axs=None, fwid=3, **kwargs):
    if axs is None:
        n_plots = len(clustering_names)
        f, axs = plt.subplots(1, n_plots, figsize=(fwid*n_plots, fwid))
    for i, cn in enumerate(clustering_names):
        sns.scatterplot(data=df, x=x, y=cn, ax=axs[i], **kwargs)
    return axs

@gpl.ax_adder
def plot_param_sweep(mod_mat, x_values, x_label='', y_label='',
                     x_dim=0, kind_dim=1, line_labels=None, ax=None):
    if line_labels is None:
        line_labels = ('',)*mod_mat.shape[kind_dim]
    mod_mat = np.moveaxis(mod_mat, (x_dim, kind_dim),
                          (0, 1))
    dims = tuple(np.arange(len(mod_mat.shape), dtype=int))
    mean_mat = np.mean(mod_mat, axis=dims[2:])
    for i in range(mod_mat.shape[1]):
        l = gpl.plot_trace_werr(x_values, mean_mat[:, i], label=line_labels[i],
                                ax=ax)
        col = l[0].get_color()
        for ind in u.make_array_ind_iterator(mod_mat.shape[2:]):
            full_ind = (slice(None), i) + ind
            ax.plot(x_values, mod_mat[full_ind], 'o', color=col)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)

def plot_clusters(*ms, axs=None, func=ma.quantify_clusters, fwid=3, **kwargs):
    if axs is None:
        f, axs = plt.subplots(1, len(ms), figsize=(fwid*len(ms), fwid))
    mins = []
    maxs = []
    outs = []
    for i, m in enumerate(ms):
        out = func(m.out_group_labels, m.model.weights[-2],
                   **kwargs)
        cluster, diff = out
        outs.append(out)
        mins.append(np.min(cluster))
        maxs.append(np.max(cluster))
    min_all = np.min(mins)
    max_all = np.max(maxs)
    for i, (cluster, diff) in enumerate(outs):
        axs[i].set_title('diff = {:.2f}'.format(diff))
        axs[i].imshow(cluster, vmin=min_all, vmax=max_all)
    return axs

def plot_weight_maps(*ms, axs=None, fhei=10, fwid=3, **kwargs):
    if axs is None:
        f, axs = plt.subplots(1, 2*len(ms), figsize=(fwid*2*len(ms), fhei))
    for i, m in enumerate(ms):
        plot_weight_map(m, axs=axs[2*i:2*(i+1)])
    return axs


@gpl.ax_adder
def plot_weight_distribution(m, ax=None, **kwargs):
    ws = m.model.weights[2]
    for i in range(ws.shape[1]):
        ax.hist(np.abs(ws[:, i]), histtype='step', **kwargs)

def plot_weight_map(m, fwid=3, axs=None, clustering=None):
    if axs is None:
        fwid = 3
        f, axs = plt.subplots(1, 2, figsize=(fwid*2, 4*fwid))

    w_inp = np.transpose(m.model.weights[0])
    w_out = np.array(m.model.weights[2])

    abs_coeffs = np.abs(w_inp)
    if clustering is None:
        hidden_order = _get_cluster_order(np.abs(w_out))
    else:
        hidden_order = _get_cluster_order(abs_coeffs, len(m.groups),
                                          clustering)
    w_inp = w_inp[hidden_order]
    
    inp_order = np.argsort(np.argmax(np.abs(w_inp), axis=0))
    w_inp = w_inp[:, inp_order]
        
    axs[0].pcolormesh(w_inp)
    axs[1].pcolormesh(w_out[hidden_order])

    axs[0].set_xlabel('inputs')
    axs[0].set_ylabel('hidden')
    axs[1].set_xlabel('outputs')
    return axs
