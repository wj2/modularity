import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.decomposition as skd
import itertools as it

import general.utility as u
import general.plotting as gpl
import modularity.analysis as ma
import modularity.auxiliary as maux


@gpl.ax_adder()
def plot_mt_learning(*outs, mixing=0, ax=None, vis_key="val_loss"):
    args_list = []
    mixing_list = []
    for out in outs:
        rel_weights, nm_strs, args, same_ds, flip_ds = out
        args_list.append(args)
        nm_ind = np.argmin(np.abs(nm_strs - mixing))
        mixing_list.append(nm_strs[nm_ind])
        epochs = np.arange(same_ds[vis_key].shape[2])
        gpl.plot_trace_werr(epochs, same_ds[vis_key][nm_ind], ax=ax, label="same")
        gpl.plot_trace_werr(epochs, flip_ds[vis_key][nm_ind], ax=ax, label="flip")
    return u.merge_dict(args_list), mixing_list


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
        order[tot : tot + len(inds)] = inds
        tot = tot + len(inds)
    return order


def visualize_activity(
    inputs,
    activity,
    targs,
    con_inds=(-2, -1),
    c_colors=("r", "g"),
    r1_color="b",
    r2_color="m",
    trs=None,
    ax=None,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"})
    cons = np.argmax(inputs[:, con_inds], axis=1)
    u_cons = np.unique(cons)
    _, trs = gpl.plot_highdim_points(
        activity,
        ax=ax,
        p=trs,
        dim_red_mean=False,
        colors=((0.1, 0.1, 0.1),),
    )
    for i, c in enumerate(u_cons):
        c_mask = cons == c
        con_act = activity[c_mask]
        lvs = inputs[c_mask]
        for j, k in it.combinations(range(len(lvs)), 2):
            if np.sum((lvs[j] - lvs[k]) ** 2) == 1:
                rel_points = con_act[np.array([j, k])]
                gpl.plot_highdim_trace(
                    rel_points,
                    ax=ax,
                    p=trs,
                    colors=(c_colors[i]),
                )
    r1_mask = targs[:, 0] == 0
    r2_mask = targs[:, 0] == 1
    gpl.plot_highdim_points(
        activity[r1_mask],
        activity[r2_mask],
        ax=ax,
        p=trs,
        colors=(r1_color, r2_color),
    )
    return ax


def visualize_model_order(orders, out_scores, ax=None, color_dict=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if color_dict is None:
        color_dict = {}
    for k, scores in out_scores.items():
        ax.plot(orders, scores, color=color_dict.get(k), label=k)
    ax.legend(frameon=False)
    return ax


def plot_zs_generalization(
    con_run,
    ax=None,
    boots=1000,
    key="zero shot",
    gen_color=None,
    dec_color=None,
    x_denom=100,
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    xs = np.zeros(len(con_run))
    vs_gen = np.zeros_like(xs)
    vs_dec = np.zeros_like(xs)
    for i, (k, v) in enumerate(con_run.items()):
        k = k / x_denom
        xs[i] = k
        mu = np.mean(v[key], axis=(2, 3))
        gen, dec = mu[:, 0], mu[:, 1]
        l_ = ax.plot((k,) * len(dec), gen, "o", color=gen_color)
        gen_color = l_[0].get_color()
        vs_gen[i] = np.mean(gen)
        vs_dec[i] = np.mean(dec)
    inds = np.argsort(xs)
    xs = xs[inds]
    vs_gen = vs_gen[inds]
    ax.plot(xs, vs_gen, color=gen_color, label="generalization")
    vs_dec = vs_dec[inds]
    # ax.plot(xs, vs_dec, color=dec_color, label="performance")

    gpl.add_hlines(0.5, ax)
    ax.set_ylabel("task performance")
    ax.set_xlabel("nonlinear mixing")


def plot_desired_order_runs(df, key_dict, **kwargs):
    runs = maux.find_key_runs(df, key_dict)
    ax = kwargs.pop("ax", None)
    for i, r in enumerate(runs):
        run_df = df[df["runind"] == r]
        ax = plot_order_run(run_df, ax=ax, set_labels=i == 0, **kwargs)
    return ax


def plot_order_spectrum(spectrum_dict, axs=None, key_order=None, shares=None):
    if key_order is None:
        if "model_rep_dynamic" in spectrum_dict.keys():
            key_order = ("input", "model_rep_dynamic", "model_rep", "task")
            shares = (0.05, 0.8, 0.05, 0.05)
        else:
            key_order = ("input", "model_rep", "task")
    if shares is None:
        shares = (1 / len(key_order),) * len(key_order)
    if axs is None:
        f = plt.figure()
        gs = f.add_gridspec(100, 100)
        axs = []
        start = 0
        sharey = None
        for i, s in enumerate(shares):
            end = int(np.round(s * 100)) + start
            axs.append(f.add_subplot(gs[:, start:end], sharey=sharey))
            start = end
            sharey = axs[i]
    for i, k in enumerate(key_order):
        v = spectrum_dict[k]
        if len(v.shape) == 1:
            v = np.expand_dims(v, 0)
            axs[i].set_xticks([0])
            axs[i].set_xticklabels([k], rotation=90)
        axs[i].plot(v, "o-")
        gpl.clean_plot(axs[i], i)
    return axs


def plot_spectrum(x_spec, spectrum, ax=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    ax.bar(x_spec, spectrum, **kwargs)
    gpl.clean_plot(ax, 0)
    gpl.make_yaxis_scale_bar(ax, 0.5, double=False, label=r"$r^{2}$", text_buff=0.5)
    ax.set_xlabel("order")
    return ax


def plot_order_run(
    run_df,
    x_key="mixing_strength",
    pks=("out_scores_input", "out_scores_model_rep", "out_scores_task"),
    labels=("input", "rep", "tasks"),
    fwid=2,
    ax=None,
    thr=0.99,
    set_labels=True,
):
    if not set_labels:
        labels = ("",) * len(labels)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
    run_df = run_df.sort_values(x_key)
    order_list = {}
    x_list = run_df[x_key]
    for i, (_, row) in enumerate(run_df.iterrows()):
        orders = row["orders"]
        for j, pk in enumerate(pks):
            o_l_j = order_list.get(pk, [])
            o_pk = np.where(row[pk] > thr)[0][0]
            o_l_j.append(orders[o_pk])
            order_list[pk] = o_l_j
    for i, (pk, l_) in enumerate(order_list.items()):
        l_ = ax.plot(x_list, order_list[pk], label=labels[i])
        ax.plot(x_list, order_list[pk], "o", color=l_[0].get_color())
    ax.legend(frameon=False)
    gpl.clean_plot(ax, 0)
    ax.set_xlabel(x_key)
    ax.set_ylabel("order explaining > {}".format(thr))
    return ax


def visualize_splitting_likelihood(
    n_tasks,
    n_latents,
    ests,
    probs=None,
    ax=None,
    label_templ="D = {}",
    pred_ls="dashed",
    colors=None,
):
    if ax is None:
        f, ax = plt.subplots()
    if colors is None:
        colors = (None,) * len(n_latents)

    for i, lat in enumerate(n_latents):
        l_ = gpl.plot_trace_werr(
            n_tasks,
            ests[:, i],
            label=label_templ.format(lat),
            ax=ax,
            log_y=True,
            color=colors[i],
        )
        if probs is not None:
            color = l_[0].get_color()
            ax.plot(n_tasks, probs[i], color=color, ls=pred_ls)
    ax.set_xlabel("N tasks")
    ax.set_ylabel("alternate decomposition\nprobability")


def visualize_excess_dimensionality(
    tasks_per_group, task_dims, ax=None, label_templ="{} task, {}"
):
    if ax is None:
        f, ax = plt.subplots()

    for k, (_, dim_wi, dim_nc) in task_dims.items():
        delta = np.squeeze(np.mean(dim_nc - dim_wi, axis=-1))
        excess_overlap = delta[0]
        excess_no_overlap = delta[1]

        ax.plot(tasks_per_group, excess_overlap, label=label_templ.format(k, "overlap"))
        ax.plot(
            tasks_per_group,
            excess_no_overlap,
            label=label_templ.format(k, "no overlap"),
        )

    ax.set_xscale("log")
    gpl.clean_plot(ax, 0)
    ax.set_xlabel("N tasks")
    ax.set_ylabel("excess dimensionality")
    ax.legend(frameon=False)


def plot_training(hist, ax=None, training_key="val_loss", func=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    tc = np.array(hist.history[training_key])
    epochs = np.arange(tc.shape[-1])
    gpl.plot_trace_werr(epochs, tc, ax=ax, **kwargs)
    return ax


def visualize_training_dimensionality(
    mhs,
    label_templ=r"$\sigma_{weights} = $",
    ax=None,
    labels=None,
    lin_color=None,
    nl_color=None,
    ms=4,
):
    if ax is None:
        f, ax = plt.subplots()
    if labels is None:
        labels = ("",) * len(mhs)

    for i, (m, h) in enumerate(mhs):
        ax.plot(
            h.history["dimensionality"], label=label_templ + " {}".format(labels[i])
        )
        c = m.n_groups
        l_ = m.group_size
        targ_lin = c * l_
        targ_nl = (c**2) * l_ - c

    x_e = ax.get_xlim()[-1]
    ax.plot(x_e, targ_lin, "o", color=lin_color, ms=ms)
    ax.plot(x_e, targ_nl, "o", color=nl_color, ms=ms)

    ax.set_xlabel("training epoch")
    ax.set_ylabel("representation\ndimensionality")
    gpl.clean_plot(ax, 0)
    ax.legend(frameon=False)
    return ax


def visualize_decoder_weights(
    w1, w2, ax=None, n_pts=100, contour_color="blue", cluster_labels=None, **kwargs
):
    if ax is None:
        f, ax = plt.subplots()

    x = np.abs(w1)
    y = np.abs(w2)

    if cluster_labels is None:
        cluster_labels = np.zeros(w1.shape)

    ax.plot(x, y, "o", **kwargs)
    upper_lim = max(np.max(x), np.max(y))
    pts_x, pts_y = np.meshgrid(
        np.linspace(0, upper_lim, n_pts), np.linspace(0, upper_lim, n_pts)
    )
    # zz = kd_pdf(np.stack([pts_x.flatten(), pts_y.flatten()],
    #                     axis=0))
    # ax.contour(pts_x, pts_y, zz.reshape(100, 100, order='A'),
    #            colors=contour_color)


def plot_task_object(
    model,
    task_ind=0,
    split_axes=(),
    axs=None,
    fwid=3,
    n_samps=1000,
    plot_3d=True,
    excl_last=True,
    ms=5,
    colors=None,
):
    if colors is None:
        colors = {0: "r", 1: "b"}
    _, stim, targs = model.get_x_true(n_train=n_samps)
    rel_stim = maux.get_relevant_dims(stim, model)

    masks = ma.task_masks(split_axes)
    if axs is None:
        if plot_3d:
            subplot_kw = {"projection": "3d"}
        else:
            subplot_kw = {}
        n_plots_h = int(np.ceil(np.sqrt(len(masks))))
        f, axs = plt.subplots(
            n_plots_h,
            n_plots_h,
            figsize=(fwid * n_plots_h, fwid * n_plots_h),
            subplot_kw=subplot_kw,
            squeeze=False,
        )
        axs = axs.flatten()
    for i, (k, mf) in enumerate(masks.items()):
        mask = mf(rel_stim)
        plot_stim = rel_stim[mask]
        incl_vars = np.var(plot_stim, axis=0) > 0
        if excl_last:
            incl_vars[-1] = False
        pts = plot_stim[:, incl_vars]
        ax_names = np.where(incl_vars)[0]

        cols = targs[mask, task_ind]
        u_cols = np.unique(cols)
        for col in u_cols:
            sub_mask = cols == col
            axs[i].plot(*pts[sub_mask].T, "o", ms=ms, color=colors[col])
        axs[i].set_title(k)
        labelers = (axs[i].set_xlabel, axs[i].set_ylabel, axs[i].set_zlabel)
        for j, an in enumerate(ax_names):
            labelers[j]("F{}".format(an))


def plot_sequential_loss(hists, axs=None, cmap="Blues", fwid=2, con_seq=None):
    if axs is None:
        f, axs = plt.subplots(
            1, len(hists), figsize=(len(hists) * fwid, fwid), sharey=True
        )
    cm = plt.get_cmap(cmap)
    colors = cm(np.linspace(0.3, 1, len(hists)))
    if con_seq is None:
        con_seq = list(range(len(hists)))
    for i, hist in enumerate(hists):
        for j in range(len(hists)):
            hist_ij = hist.history["corr_tracking"][j]
            epochs = np.arange(len(hist_ij))
            if con_seq[i] == j:
                axs[i].plot(epochs, hist_ij, color=colors[j])
            else:
                axs[i].plot(epochs, hist_ij, color=colors[j], linestyle="dashed")
        gpl.add_hlines(0.5, axs[i])
        gpl.clean_plot(axs[i], i)


geometry_metrics = ("shattering", "within_ccgp", "across_ccgp")


def plot_geometry_metrics(*args, geometry_names=geometry_metrics, **kwargs):
    return plot_clustering_metrics(
        *args, clustering_names=geometry_names, ms=None, **kwargs
    )


def visualize_gate_angle(stim, targs, gates, ws=None, ax=None, ref_vector=None):
    if ax is None:
        f, ax = plt.subplots(subplot_kw={"projection": "3d"})
    if ref_vector is None:
        rv = np.zeros(stim.shape[1])
        rv[-1] = 1
        rv = np.expand_dims(rv, 0)
    gates = u.make_unit_vector(gates)
    print(rv.shape, gates.shape, ws.shape)
    ax.hist(np.sum(gates * rv, axis=1), density=True)
    if ws is not None:
        ax.hist(np.sum(ws * rv, axis=1), density=True, histtype="step")


def visualize_stable_gates(
    stim,
    targs,
    gates,
    ws=None,
    ax=None,
    pt_ms=5,
    split_dim=-1,
    stable_color=(0.6, 0.6, 0.6),
):
    if ax is None:
        f, ax = plt.subplots(subplot_kw={"projection": "3d"})
    gates_uv = u.make_unit_vector(gates)
    for uv in gates_uv:
        traj = np.stack((np.zeros(len(uv)), uv), axis=0)
        ax.plot(*traj.T, color=stable_color)
    if ws is not None:
        for uv in ws:
            traj = np.stack((np.zeros(len(uv)), uv), axis=0)
            ax.plot(*traj.T, alpha=0.1)

    uv = np.unique(stim[:, split_dim])
    for val in uv:
        mask = val == stim[:, split_dim]
        ax.plot(*stim[mask].T, "o", ms=pt_ms)
    # ax.view_init(0, 0)
    return ax


def visualize_module_activity(
    model,
    context,
    ax=None,
    n_samps=1000,
    line_color=None,
    line_colors=None,
    pt_color=None,
    pt_alpha=1,
    resp_colors=(None, None),
    plot_resp_cats=True,
    task_ind=0,
    ms=None,
    p=None,
    fix_vars=None,
    fix_value=0,
    linestyle="solid",
    **kwargs,
):
    inp_rep, stim, targ = model.get_x_true(
        n_train=n_samps,
        group_inds=context,
        fix_vars=fix_vars,
        fix_value=fix_value,
    )
    stim, ind = np.unique(stim, axis=0, return_index=True)
    targ = targ[ind]
    inp_rep = inp_rep[ind]

    rep = model.get_representation(inp_rep)
    rel_stim = stim[:, model.groups[context]]
    centroids = np.unique(rel_stim, axis=0)

    ax, p = gpl.plot_highdim_trace(
        rep,
        plot_line=False,
        plot_points=True,
        ax=ax,
        p=p,
        color=pt_color,
        alpha=pt_alpha,
        ms=ms,
    )

    rep_cents = {}
    for c in centroids:
        mask = np.all(rel_stim == np.expand_dims(c, 0), axis=1)
        rep_cent = np.mean(rep[mask], axis=0, keepdims=True)
        gpl.plot_highdim_trace(
            rep_cent,
            plot_line=False,
            plot_points=True,
            ax=ax,
            p=p,
            color=pt_color,
            ms=ms,
        )
        rep_cents[tuple(c)] = rep_cent
    if line_colors is None:
        line_colors = (line_color,) * centroids.shape[1]
    for c1, c2 in it.combinations(centroids, 2):
        if np.sum((c1 - c2) ** 2) == 1:
            ind = np.argmax((c1 - c2) ** 2)
            rc1 = rep_cents[tuple(c1)]
            rc2 = rep_cents[tuple(c2)]
            comb = np.concatenate((rc1, rc2), axis=0)
            gpl.plot_highdim_trace(
                comb,
                ax=ax,
                p=p,
                color=line_colors[ind],
                linestyle=linestyle,
            )

    if plot_resp_cats:
        cats = model.group_func[context](rel_stim)[:, task_ind]

        ax, p = gpl.plot_highdim_trace(
            rep[cats],
            plot_line=False,
            plot_points=True,
            ax=ax,
            p=p,
            color=resp_colors[0],
            alpha=pt_alpha,
            ms=ms,
        )
        ax, p = gpl.plot_highdim_trace(
            rep[~cats],
            plot_line=False,
            plot_points=True,
            ax=ax,
            p=p,
            color=resp_colors[1],
            alpha=pt_alpha,
            ms=ms,
        )
    gpl.clean_3d_plot(ax)
    gpl.make_3d_bars(ax, bar_len=1)
    ax.set_aspect("equal")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")
    return ax, p


clustering_metrics_all = (
    "cosine_sim_diffs",
    "cosine_sim_absolute_diffs",
    "threshold_diffs",
    "brim",
)
clustering_metrics = ("brim_diffs", "threshold_diffs")


def plot_clustering_metrics(
    df,
    x="tasks_per_group",
    clustering_names=clustering_metrics,
    axs=None,
    fwid=3,
    **kwargs,
):
    if axs is None:
        n_plots = len(clustering_names)
        f, axs = plt.subplots(1, n_plots, figsize=(fwid * n_plots, fwid), squeeze=False)
    for i, cn in enumerate(clustering_names):
        sns.scatterplot(data=df, x=x, y=cn, ax=axs[0, i], **kwargs)
    return axs


def plot_linear_model(
    coef_dict, targ_fields, inter=None, axs=None, fwid=3, label="", use_num=False
):
    n_rows = len(targ_fields)
    n_cols = len(coef_dict)
    if inter is not None:
        n_cols = n_cols + 1
    if axs is None:
        f, axs = plt.subplots(n_rows, n_cols, figsize=(fwid * n_cols, fwid * n_rows))
    else:
        f = None
    for i, (fn, (x_vals, weights)) in enumerate(coef_dict.items()):
        if use_num:
            x_num = np.arange(len(x_vals))
        else:
            x_num = np.array(x_vals, dtype=float)
        for j, w_ij in enumerate(weights):
            if inter is not None and i == 0:
                axs[j, -1].plot([0], [inter[j]], "o")
                gpl.clean_plot(axs[j, -1], 0)
                gpl.clean_plot_bottom(axs[j, -1])
                axs[j, -1].set_xlabel("intercept")
            if i == 0:
                axs[j, i].set_ylabel(targ_fields[j])
            l_ = axs[j, i].plot(x_num, w_ij)
            if i == 0 and j == 0:
                use_label = label
            else:
                use_label = ""
            axs[j, i].plot(x_num, w_ij, "o", color=l_[0].get_color(), label=use_label)
            if use_num:
                axs[j, i].set_xticks(x_num)
                axs[j, i].set_xticklabels(x_vals)
            gpl.clean_plot(axs[j, i], i)
            if i > 0:
                axs[j, i].sharey(axs[j, i - 1])
                axs[j, i].autoscale()
            if j == len(weights) - 1:
                axs[j, i].set_xlabel(fn)
            if i == 0 and j == 0 and len(label) > 0:
                axs[j, i].legend(frameon=False)
            gpl.add_hlines(0, axs[j, i])
    return f, axs


def plot_optimal_context_scatter(
    *args,
    **kwargs,
):
    out = plot_context_scatter(
        *args, cluster_func=ma.infer_optimal_activity_clusters, **kwargs
    )
    return out


def plot_context_scatter(
    m,
    n_samps=1000,
    ax=None,
    fwid=3,
    from_layer=None,
    colors=None,
    cluster_func=ma.infer_activity_clusters,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
    labels, act = cluster_func(
        m, n_samps=n_samps, use_mean=True, ret_act=True, from_layer=from_layer
    )
    xy_labels = ("con 1 activity", "con 2 activity")
    u_labels = np.unique(labels)
    if colors is None:
        colors = (None,) * len(u_labels)
    if len(u_labels) == 1:
        colors = (colors[1],)
    if len(u_labels) == 2:
        colors = (colors[0], colors[-1])

    if act.shape[1] > 2:
        p = skd.PCA(2)
        act = p.fit_transform(act)
        xy_labels = ("PC 1", "PC 2")

    mean_diff = list(
        np.mean(act[labels == l_, 1] - act[labels == l_, 0]) for l_ in u_labels
    )
    u_labels = u_labels[np.argsort(mean_diff)]
    for i, l_ in enumerate(u_labels):
        mask = labels == l_
        ax.plot(act[mask, 0], act[mask, 1], "o", color=colors[i])
    ax.set_xlabel(xy_labels[0])
    ax.set_ylabel(xy_labels[1])
    gpl.clean_plot(ax, 0)
    return ax


def _remove_singleton(li):
    if u.check_list(li):
        li = li[0]
    return li


def accumulate_run_quants(
    ri_list,
    quant_key="model_frac",
    templ="modularizer_nls([0-9]+)-{run_ind}",
    legend_keys=("date",),
    **kwargs,
):
    dms = []
    quants = {}
    for ri in ri_list:
        run = maux.load_run(ri, file_template=templ, **kwargs)
        dm = run[1]
        dms.append(dm)
        key = tuple(_remove_singleton(run[2][k]) for k in legend_keys)
        quants[key] = (run[1], run[0][quant_key])
    return quants


def visualize_ri_list(
    ri_list,
    quant_key="model_frac",
    templ="modularizer_nls([0-9]+)-{run_ind}",
    legend_keys=("date",),
    ax=None,
    fwid=3,
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))

    quants = accumulate_run_quants(
        ri_list, quant_key=quant_key, templ=templ, legend_keys=legend_keys
    )
    print(quant_key)
    for k, (xs, qs) in quants.items():
        l_text = list(
            lk + " = {}".format(np.squeeze(k[i])) for i, lk in enumerate(legend_keys)
        )
        l_text = "\n".join(l_text)
        gpl.plot_trace_werr(xs, qs.T, ax=ax, label=l_text)
    return ax


def plot_context_clusters(
    m,
    n_samps=1000,
    ax=None,
    fwid=3,
    from_layer=None,
    cmap="Blues",
    context_colors=None,
    gap=0.05,
    fontsize="small",
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
    labels = ma.infer_activity_clusters(
        m, n_samps=n_samps, use_mean=True, from_layer=from_layer
    )
    activity = ma.sample_all_contexts(
        m, n_samps=n_samps, use_mean=False, from_layer=from_layer
    )
    if context_colors is None:
        context_colors = (None,) * len(activity)
    sort_inds = np.argsort(labels)
    a_full = np.concatenate(activity, axis=0)
    gap = int(np.round(gap * n_samps))
    vmax = np.mean(a_full) + np.std(a_full)
    # gpl.pcolormesh(
    #     a_full[:, sort_inds], vmax=vmax, cmap=cmap, rasterized=True, ax=ax,
    # )
    # xts = np.arange(0, a_full.shape[1] + 1, 50)
    # ax.set_xticks(xts)
    ax.imshow(a_full[:, sort_inds], aspect="auto", vmax=vmax, cmap=cmap)
    for i, a in enumerate(activity):
        n_samps = a.shape[0]
        gpl.make_yaxis_scale_bar(
            ax,
            anchor=(n_samps / 2) + n_samps * i,
            magnitude=n_samps / 2 - gap,
            label="con {}".format(i + 1),
            color=context_colors[i],
            fontsize=fontsize,
        )
    gpl.clean_plot(ax, 1)
    ax.set_xlabel("hidden units")
    return ax


def plot_simple_tuning(m, xs, hist, ind=0, ax=None, thr=0.0001, xs_tr=None, ys=None):
    if ax is None:
        f, ax = plt.subplots()
    if xs_tr is None:
        xs_tr = xs

    reps = np.array(m(xs_tr))
    x_mask = reps[:, ind] > thr
    xs_in = xs[x_mask]

    final_weights = u.make_unit_vector(np.array(m.weights[0]).T)[ind]
    ax.plot([0, final_weights[0]], [0, final_weights[1]])
    if ys is not None:
        cv2 = ma.compute_svd(
            m, xs_tr, ys, ind=ind, thr=thr, renorm_targs=False, unit=True
        )
        cv2 = u.make_unit_vector(np.sum(np.abs(cv2), axis=0) * np.sign(final_weights))
        ax.plot([0, cv2[0]], [0, cv2[1]], linestyle="dashed")

    if "weights" in hist.history:
        ws = hist.history["weights"]
        w_traj = u.make_unit_vector(np.array(list(w[:, ind] for w in ws)))
        ax.plot(*w_traj.T, "o")

    ax.plot(*xs_in.T, "o", ms=5)

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_aspect("equal")


def plot_func_clusters(
    m, func_list, n_clusters=None, ax=None, cmap="Blues", n_samps_per_func=2000
):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    rep, stim, targ = m.get_x_true(n_train=n_samps_per_func * len(func_list))
    rel_stim = maux.get_relevant_dims(stim, m)
    m_rep = m.get_representation(rep)
    func_resps = []
    for i, f in enumerate(func_list):
        mask = f(rel_stim)
        func_resps.append(m_rep[mask])
    a_ms = np.concatenate(
        list(np.mean(x, axis=0, keepdims=True) for x in func_resps), axis=0
    )
    if n_clusters is None:
        n_clusters = len(func_list) + 1
    _, inds = ma._fit_clusters(a_ms, n_clusters)
    sort_inds = np.argsort(inds)

    a_full = np.concatenate(func_resps, axis=0)
    vmax = np.mean(a_full) + np.std(a_full)
    ax.imshow(a_full[:, sort_inds], aspect="auto", vmax=vmax, cmap=cmap)
    gpl.clean_plot(ax, 0)


def plot_metrics(
    *metrics, labels=None, axs=None, fwid=1.5, metric_names=None, flat=True, **kwargs
):
    if labels is None:
        labels = ("",) * len(metrics[0])
    if metric_names is None:
        metric_names = ("",) * len(metrics)

    if flat:
        n_rows = 1
    else:
        n_rows = metrics[0].shape[1]

    if axs is None:
        f, axs = plt.subplots(
            n_rows,
            len(metrics),
            figsize=(fwid * len(metrics), n_rows * fwid),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
    for i, metric in enumerate(metrics):
        for j in range(metric.shape[1]):
            if flat:
                j_ax = 0
            else:
                j_ax = j
            axs[j_ax, i].violinplot(metric[:, j], positions=[0, 1, 2], **kwargs)
            gpl.clean_plot(axs[j_ax, i], i)
            axs[j_ax, i].set_xticks([0, 1, 2])
            axs[j_ax, i].set_xticklabels(labels, rotation=90)
            gpl.add_hlines(0.5, axs[j_ax, i])
            axs[j_ax, i].set_title(metric_names[i])
            axs[j_ax, 0].set_ylabel("decoding performance")
    return f, axs


def compare_act_weight_clusters(
    m,
    n_samps=1000,
    axs=None,
    fwid=3,
    n_clusters=None,
    methods=(ma.cluster_graph, ma.act_cluster, ma.cluster_max_corr),
):
    act = np.concatenate(
        ma.sample_all_contexts(m, use_mean=False, n_samps=n_samps), axis=0
    )

    vmax = np.mean(act) + np.std(act)

    if axs is None:
        f, axs = plt.subplots(
            len(methods), 1, figsize=(fwid, fwid * len(methods)), squeeze=False
        )
    for i, method in enumerate(methods):
        m_cluster = method(m, n_clusters=n_clusters)
        m_sort = np.argsort(m_cluster)
        axs[i, 0].imshow(act[:, m_sort], vmax=vmax)

        axs[i, 0].set_aspect("auto")
        axs[i, 0].set_ylabel(method.__name__)


def plot_model_list_activity(m_list, fwid=3, axs=None, f=None, cmap="Blues", **kwargs):
    n_plots = len(m_list)
    if axs is None:
        f, axs = plt.subplots(
            2, n_plots, figsize=(n_plots * fwid, 2 * fwid), sharey="row", sharex="row"
        )
    for i, m in enumerate(m_list):
        plot_context_clusters(m, ax=axs[0, i], cmap=cmap, **kwargs)
        plot_context_scatter(m, ax=axs[1, i], **kwargs)
        diff = ma.quantify_activity_clusters(m)
        axs[0, i].set_xlabel("units")
        axs[1, i].set_title("cluster diff = {:.2f}".format(diff))
        axs[1, i].set_xlabel("activity in context 1")
        axs[1, i].set_ylabel("activity in context 2")
    axs[0, 0].set_ylabel("stimuli")
    return f, axs


def plot_ablation(*mats, axs=None, fwid=3, boundzero=True):
    if axs is None:
        f, axs = plt.subplots(1, len(mats), figsize=(len(mats) * fwid, fwid))
        if len(mats) == 1:
            axs = [axs]
    full_mat = np.stack(mats, axis=0)
    if boundzero:
        vmin = 0
    else:
        vmin = np.min(full_mat)
    vmax = np.max(full_mat)
    for i, mat in enumerate(mats):
        mat[mat < 0] = 0
        m = axs[i].imshow(mat, vmin=vmin, vmax=vmax, cmap="Blues")
    f.colorbar(m, ax=axs, label="normalized\nperformance change")
    return axs


def plot_2context_activity_diff(
    fdg, m, n_samps=1000, ax=None, integrate_context=False, n_groups=2, fwid=3
):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid * 1.5, fwid))

    m_rep0 = m.sample_reps(n_samps, context=0)[2].numpy()
    m_rep1 = m.sample_reps(n_samps, context=1)[2].numpy()

    mr0 = np.mean(m_rep0**2, axis=0)
    mr1 = np.mean(m_rep1**2, axis=0)

    sort_inds = np.argsort(mr0 - mr1)
    mask = (mr0 - mr1) > 0

    plot_arr = np.concatenate((m_rep0[:, sort_inds], m_rep1[:, sort_inds]))

    # m_arr = np.mean(plot_arr, axis=0)
    # ax_corr.plot(mr0 - m_arr, mr1 - m_arr, 'o')
    vmax = np.mean(plot_arr) + np.std(plot_arr)
    ax.imshow(plot_arr, aspect="auto", vmax=vmax)
    ax.set_xlabel("units")
    ax.set_ylabel("trials")
    return ax, mask


def plot_task_heat(
    data,
    task_field="tasks_per_group",
    dim_field="args_group_width",
    heat_field="gm",
    fax=None,
    fwid=3,
    ng_field="group_size",
    val_field="val_loss",
    n_groups_field="n_groups",
    val_thr=0.1,
    skip_y=0,
    vmax=None,
):
    if fax is None:
        fax = plt.subplots(1, 1, figsize=(fwid, fwid))
    f, ax = fax
    n_tasks = np.unique(data[task_field])
    widths = np.unique(data[dim_field])

    combos = it.product(range(len(n_tasks)), range(len(widths)))
    plot_arr = np.zeros((len(n_tasks), len(widths)))
    for nt_ind, wi_ind in combos:
        nt = n_tasks[nt_ind]
        wi = widths[wi_ind]
        mask = np.logical_and(data[task_field] == nt, data[dim_field] == wi)
        out = data[heat_field][mask]
        val_mask = data[val_field][mask] < val_thr
        plot_arr[nt_ind, wi_ind] = np.nanmean(out[val_mask])

    m = gpl.pcolormesh(
        widths, n_tasks[skip_y:], plot_arr[skip_y:], ax=ax, equal_bins=True, vmax=vmax
    )
    ax.set_xlabel("group width")
    ax.set_ylabel("n tasks")
    f.colorbar(m, ax=ax)
    ng = np.unique(data[ng_field])[0]
    n_groups = np.unique(data[n_groups_field])[0]
    x_ind = np.argmin(np.abs(2**ng - widths))
    y_ind = np.argmin(np.abs(2**ng - n_tasks[skip_y:]))
    gpl.add_vlines(x_ind, ax)
    gpl.add_hlines(y_ind, ax)

    ax.plot([0, len(n_tasks)], [1, len(n_tasks) + 2])
    x_ind2 = np.argmin(np.abs(n_groups * 2 ** (ng) - widths))
    gpl.add_vlines(x_ind2, ax)
    return fax


@gpl.ax_adder()
def plot_param_sweep(
    mod_mat,
    x_values,
    x_label="",
    y_label="",
    x_dim=0,
    kind_dim=1,
    line_labels=None,
    ax=None,
):
    if line_labels is None:
        line_labels = ("",) * mod_mat.shape[kind_dim]
    mod_mat = np.moveaxis(mod_mat, (x_dim, kind_dim), (0, 1))
    dims = tuple(np.arange(len(mod_mat.shape), dtype=int))
    mean_mat = np.mean(mod_mat, axis=dims[2:])
    for i in range(mod_mat.shape[1]):
        l_ = gpl.plot_trace_werr(x_values, mean_mat[:, i], label=line_labels[i], ax=ax)
        col = l_[0].get_color()
        for ind in u.make_array_ind_iterator(mod_mat.shape[2:]):
            full_ind = (slice(None), i) + ind
            ax.plot(x_values, mod_mat[full_ind], "o", color=col)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(frameon=False)


def plot_clusters(*ms, axs=None, func=ma.quantify_clusters, fwid=3, **kwargs):
    if axs is None:
        f, axs = plt.subplots(1, len(ms), figsize=(fwid * len(ms), fwid))
    mins = []
    maxs = []
    outs = []
    for i, m in enumerate(ms):
        out = func(m.out_group_labels, m.model.weights[-2], **kwargs)
        cluster, diff = out
        outs.append(out)
        mins.append(np.min(cluster))
        maxs.append(np.max(cluster))
    min_all = np.min(mins)
    max_all = np.max(maxs)
    for i, (cluster, diff) in enumerate(outs):
        axs[i].set_title("diff = {:.2f}".format(diff))
        axs[i].imshow(cluster, vmin=min_all, vmax=max_all)
    return axs


def plot_weight_maps(*ms, axs=None, fhei=10, fwid=3, **kwargs):
    if axs is None:
        f, axs = plt.subplots(1, 2 * len(ms), figsize=(fwid * 2 * len(ms), fhei))
    for i, m in enumerate(ms):
        plot_weight_map(m, axs=axs[2 * i : 2 * (i + 1)])
    return axs


@gpl.ax_adder()
def plot_weight_distribution(m, ax=None, **kwargs):
    ws = m.model.weights[2]
    for i in range(ws.shape[1]):
        ax.hist(np.abs(ws[:, i]), histtype="step", **kwargs)


def plot_weight_map(m, fwid=3, axs=None, clustering=None):
    if axs is None:
        fwid = 3
        f, axs = plt.subplots(1, 2, figsize=(fwid * 2, 4 * fwid))

    w_inp = np.transpose(m.model.weights[0])
    w_out = np.array(m.model.weights[2])

    abs_coeffs = np.abs(w_inp)
    if clustering is None:
        hidden_order = _get_cluster_order(np.abs(w_out))
    else:
        hidden_order = _get_cluster_order(abs_coeffs, len(m.groups), clustering)
    w_inp = w_inp[hidden_order]

    inp_order = np.argsort(np.argmax(np.abs(w_inp), axis=0))
    w_inp = w_inp[:, inp_order]

    axs[0].pcolormesh(w_inp)
    axs[1].pcolormesh(w_out[hidden_order])

    axs[0].set_xlabel("inputs")
    axs[0].set_ylabel("hidden")
    axs[1].set_xlabel("outputs")
    return axs
