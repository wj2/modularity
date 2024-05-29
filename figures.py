import itertools as it
import numpy as np
import scipy.stats as sts
import scipy.special as ss
import matplotlib.pyplot as plt
import sklearn.decomposition as skd

import modularity.simple as ms
import modularity.analysis as ma
import modularity.visualization as mv
import modularity.auxiliary as maux
import disentangled.data_generation as dg

import general.utility as u
import general.paper_utilities as pu
import general.plotting as gpl

config_path = "modularity/figures.conf"

colors = (
    np.array(
        [
            (127, 205, 187),
            (65, 182, 196),
            (29, 145, 192),
            (34, 94, 168),
            (37, 52, 148),
            (8, 29, 88),
        ]
    )
    / 256
)


class ModularizerFigure(pu.Figure):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, find_panel_keys=False, **kwargs)

    def make_fdg(self, retrain=False, dg_dim=None):
        if self.data.get("trained_fdg") is None or retrain:
            inp_dim = self.params.getint("inp_dim")
            if dg_dim is None:
                dg_dim = self.params.getint("dg_dim")

            rescale = self.params.getboolean("rescale_fdg")
            dg_epochs = self.params.getint("dg_epochs")
            dg_noise = self.params.getfloat("dg_noise")
            dg_regweight = self.params.getlist("dg_regweight", typefunc=float)
            dg_layers = self.params.get("dg_layers")
            dg_layers = self.params.getlist("dg_layers", typefunc=int)
            dg_train_egs = self.params.getint("dg_train_egs")
            dg_pr_reg = self.params.getboolean("dg_pr_reg")
            dg_bs = self.params.getint("dg_batch_size")

            continuous = self.params.getboolean("continuous", False)
            if continuous:
                source_distr = sts.multivariate_normal([0] * inp_dim, 1)
            else:
                source_distr = u.MultiBernoulli(0.5, inp_dim)
            fdg = dg.FunctionalDataGenerator(
                inp_dim,
                dg_layers,
                dg_dim,
                noise=dg_noise,
                use_pr_reg=dg_pr_reg,
                l2_weight=dg_regweight,
                rescale=rescale,
            )
            fdg.fit(
                source_distribution=source_distr,
                epochs=dg_epochs,
                train_samples=dg_train_egs,
                batch_size=dg_bs,
                verbose=False,
            )
            self.data["trained_fdg"] = fdg
        return self.data["trained_fdg"]

    def make_mddg(self, nl, **kwargs):
        n_units = kwargs.get("n_units", self.params.getint("n_units"))
        n_feats = kwargs.get("n_feats", self.params.getint("n_feats"))
        mddg = dg.MixedDiscreteDataGenerator(
            n_feats,
            mix_strength=nl,
            n_units=n_units,
        )
        return mddg

    def _load_consequences_sweep(
        self,
        key="consequences_simulations",
        reload=False,
        run_key="run_inds_sweep",
        template_key="controlled_template",
        folder_key="controlled_folder",
    ):
        if self.data.get(key) is None or reload:
            run_inds = self.params.getlist(run_key)
            controlled_template = self.params.get(template_key)
            controlled_folder = self.params.get(folder_key)

            mixing = []
            mix_dicts = []
            for i, ri in enumerate(run_inds):
                out_dict = maux.load_consequence_runs(
                    ri,
                    folder=controlled_folder,
                    template=controlled_template,
                    ref_key="tasks_per_group",
                )
                m = out_dict[1]["args"]["dm_input_mixing"]
                m = m / out_dict[1]["args"]["dm_input_mixing_denom"]
                mix_dicts.append(out_dict)
                mixing.append(m)
            self.data[key] = (mixing, mix_dicts)
        return self.data[key]

    def load_and_organize_con_sweep(self, plot_keys, *args, load_num=2, **kwargs):
        out = self._load_consequences_sweep(*args, **kwargs)
        mixing, mix_dicts = out
        inds = np.argsort(mixing)
        mix_sort = np.array(mixing)[inds]

        metric_dict = {}
        for i, ind in enumerate(inds):
            out_dict = mix_dicts[ind]
            task_arr = np.array(list(out_dict.keys()))
            task_inds = np.argsort(task_arr)
            task_sort = task_arr[task_inds]

            n_ts = len(task_arr)
            for pk in plot_keys:
                for j, nt in enumerate(task_sort):
                    group = out_dict[nt][pk]
                    if load_num == 1:
                        group = (group,)
                    if len(group[0].shape) > 2:
                        group = list(np.mean(g, axis=-1) for g in group)
                    group_arr = metric_dict.get(pk, (None,) * len(group))
                    if group_arr[0] is None:
                        shape = (
                            len(mix_sort),
                            n_ts,
                        ) + group[0].shape
                        group_arr = list(np.zeros(shape) for g in group)
                    for k, g in enumerate(group):
                        group_arr[k][i, j] = g
                    metric_dict[pk] = group_arr
        out = (mix_sort, task_sort, metric_dict)
        return out

    def train_eg_networks(self):
        n_tasks = self.params.getlist("eg_n_tasks", typefunc=int)
        nl_strengths = self.params.getlist("eg_nl_strs", typefunc=float)
        n_units = self.params.getint("n_units")
        n_feats = self.params.getint("n_feats")

        models = np.zeros((len(n_tasks), len(nl_strengths)), dtype=object)
        hists = np.zeros_like(models)
        for i, nt in enumerate(n_tasks):
            for j, nl in enumerate(nl_strengths):
                mddg = dg.MixedDiscreteDataGenerator(
                    n_feats,
                    mix_strength=nl,
                    n_units=n_units,
                )
                m_ij, h_ij = self.train_modularizer(fdg=mddg, tasks_per_group=nt)
                models[i, j] = m_ij
                hists[i, j] = h_ij
        out = (n_tasks, nl_strengths), models, hists
        return out

    def make_ident_modularizer(self, linear=False, **kwargs):
        if linear:
            m_type = ms.LinearIdentityModularizer
        else:
            m_type = ms.IdentityModularizer
        m_ident, h = self.train_modularizer(model_type=m_type, train_epochs=0, **kwargs)
        return m_ident

    def _plot_learning_cons(
        self,
        plot_dict,
        plot_dict_rand,
        axs,
        colors=None,
        mod_name="modular network",
        naive_name="naive network",
        plot_keys=(
            "new task tasks",
            "new context tasks",
            "related context tasks",
            "related context inference tasks",
        ),
        plot_labels=(
            "novel",
            "new context",
            "related context",
            "related context untrained",
        ),
        log_y=False,
    ):
        if colors is None:
            colors = (None, None)
        naive_color, mod_color = colors
        for i, key in enumerate(plot_keys):
            loss_pre, loss_null = plot_dict[key]
            if len(loss_pre.shape) > 2:
                loss_pre = np.mean(loss_pre, axis=2)
                loss_null = np.mean(loss_null, axis=2)
            ax = axs[i]
            xs = np.arange(0, loss_pre.shape[1])
            if i == len(plot_keys) - 1:
                l_mod = mod_name
                l_naive = naive_name
            else:
                l_mod = ""
                l_naive = ""
            gpl.plot_trace_werr(
                xs,
                loss_null,
                ax=ax,
                color=naive_color,
                log_y=log_y,
                label=l_naive,
            )
            gpl.plot_trace_werr(
                xs,
                loss_pre,
                ax=ax,
                color=mod_color,
                log_y=log_y,
                label=l_mod,
            )
            if key == plot_keys[-1]:
                loss_pre_rand, _ = plot_dict_rand[key]
                if len(loss_pre_rand.shape) > 2:
                    loss_pre_rand = np.mean(loss_pre_rand, axis=2)
                gpl.plot_trace_werr(
                    xs,
                    loss_pre_rand,
                    ax=ax,
                    color=mod_color,
                    log_y=log_y,
                    label="orthogonal tasks",
                    linestyle="dashed",
                )
            ax.set_ylabel("{}\ntask performance".format(plot_labels[i]))
            ax.set_xlabel("training epochs")
            gpl.add_hlines(0.5, ax)

    def make_modularizers(self, retrain=False):
        if self.data.get("trained_models") is None or retrain:
            act_reg = self.params.getfloat("act_reg")

            m_noreg, h_noreg = self.train_modularizer(act_reg_weight=0)

            # m_reg, h_reg = self.train_modularizer(act_reg_weight=act_reg)
            labels = ("",)  #  'regularization')
            self.data["trained_models"] = (
                (m_noreg,),  # m_reg),
                (h_noreg,),  # h_reg),
                labels,
            )
        return self.data["trained_models"]

    def train_modularizer(self, verbose=False, fdg=None, **kwargs):
        if fdg is None:
            fdg = self.make_fdg()
        return ms.train_modularizer(fdg, verbose=verbose, params=self.params, **kwargs)

    def load_run(self, run_ind, **kwargs):
        folder = self.params.get("sim_folder")
        out = maux.load_run(run_ind, folder=folder, **kwargs)
        return out

    def load_nls_runs(self, *args, **kwargs):
        folder = self.params.get("sim_folder")
        out = maux.load_nls_param_sweep(*args, folder=folder, **kwargs)
        return out

    def _quantification_panel(
        self,
        quant_keys,
        ri_list,
        axs,
        label_dict=None,
        nulls=None,
        legend_keys=("group_size",),
        plot_ylabels=None,
        colors=None,
    ):
        if colors is None:
            colors = {}
        if plot_ylabels is None:
            plot_ylabels = quant_keys
        model_templ = self.params.get("model_template")
        if nulls is None:
            nulls = (0.5,) * len(quant_keys)
        if label_dict is None:
            label_dict = {}
        for i, qk in enumerate(quant_keys):
            qk_ri = mv.accumulate_run_quants(
                ri_list,
                templ=model_templ,
                quant_key=qk,
                legend_keys=legend_keys,
            )
            for k, (xs, qs) in qk_ri.items():
                gpl.plot_trace_werr(
                    xs,
                    qs.T,
                    ax=axs[i],
                    label=label_dict[k],
                    log_x=True,
                    color=colors.get(k),
                )
            axs[i].set_ylabel(plot_ylabels[i])
            gpl.add_hlines(nulls[i], axs[i])


class NichoSimoneFigure(ModularizerFigure):
    def __init__(self, fig_key="nicho_simone_figure", colors=colors, **kwargs):
        fsize = (4.4, 3.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        ps_grid = pu.make_mxn_gridspec(self.gs, 1, 3, 0, 45, 0, 100, 8, 2)
        perf_axs = self.get_axs(ps_grid, sharey="all", squeeze=True)

        g11 = self.gs[60:90, :20]
        g12 = self.gs[60:90, 25:75]
        g13 = self.gs[60:90, 80:]
        g21 = self.gs[93:, :20]
        g22 = self.gs[93:, 25:75]
        g23 = self.gs[93:, 80:]
        gs_arr = np.array([(g11, g12, g13), (g21, g22, g23)])
        diff_axs = self.get_axs(gs_arr, sharey="horizontal")

        gss["panel_performance"] = (perf_axs, diff_axs)
        self.gss = gss

    def panel_performance(self):
        key = "panel_performance"
        perf_axs, diff_axs = self.gss[key]

        aligned_ri = self.params.get("aligned_runind")
        same_ri = self.params.get("same_runind")
        align = self.params.getint("eg_align")
        s_color = self.params.getcolor("same_color")
        d_color = self.params.getcolor("different_color")
        diff_color = self.params.getcolor("diff_color")

        aligned_out = maux.load_mt_run(aligned_ri, gd_func=np.float64, sort_key="corr")
        same_out = maux.load_mt_run(same_ri, gd_func=np.float64, sort_key="corr")
        _ = mv.plot_mt_learning(
            aligned_out,
            key_targ=0,
            ax=perf_axs[0],
            vis_key="corr_rate",
            same_color=s_color,
            flip_color=d_color,
        )
        _ = mv.plot_mt_learning(
            aligned_out,
            key_targ=align,
            ax=perf_axs[1],
            vis_key="corr_rate",
            same_color=s_color,
            flip_color=d_color,
            same_label="",
            flip_label="",
        )
        _ = mv.plot_mt_learning(
            same_out,
            key_targ=0,
            ax=perf_axs[2],
            vis_key="corr_rate",
            same_color=s_color,
            flip_color=d_color,
            same_label="",
            flip_label="",
        )
        list(pa.set_xlabel("training epoch") for pa in perf_axs)
        list(gpl.add_hlines(0.5, pa) for pa in perf_axs)
        list(gpl.clean_plot(ax, i) for i, ax in enumerate(perf_axs))
        perf_axs[0].set_ylabel("fraction correct")

        mv.plot_mt_diff(
            aligned_out, key_targ=0, color=diff_color, ax=diff_axs[0, 0], points=True
        )
        mv.plot_mt_diff(aligned_out, color=diff_color, ax=diff_axs[0, 1])
        mv.plot_mt_diff(
            same_out, key_targ=0, color=diff_color, ax=diff_axs[0, 2], points=True
        )
        list(gpl.clean_plot_bottom(da) for da in diff_axs[0])
        diff_axs[0, 0].set_ylim([-0.1, 0.6])
        mv.plot_mt_diff(
            aligned_out, key_targ=0, color=diff_color, ax=diff_axs[1, 0], points=True
        )
        mv.plot_mt_diff(aligned_out, color=diff_color, ax=diff_axs[1, 1])
        mv.plot_mt_diff(
            same_out, key_targ=0, color=diff_color, ax=diff_axs[1, 2], points=True
        )
        diff_axs[1, 0].set_ylim([-5, -4])
        diff_axs[0, 0].set_ylabel("learning speed difference\n(different - same)")
        diff_axs[1, 0].set_xticks([0])
        diff_axs[1, 0].set_xticklabels(["orthogonal"])
        diff_axs[1, 2].set_xticks([0])
        diff_axs[1, 2].set_xticklabels(["same"])
        diff_axs[1, 1].set_xlabel("feature similarity")

        for i, j in u.make_array_ind_iterator(diff_axs.shape):
            gpl.clean_plot(diff_axs[i, j], j)


class FigureWorldIntro(ModularizerFigure):
    def __init__(self, fig_key="task_intro_figure", colors=colors, **kwargs):
        fsize = (4.4, 3.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_minimal_order",
            "panel_input_spectrum",
            "panel_generalization",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}
        ps_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 100, 60, 100, 8, 2)

        gss["panel_scaling"] = self.get_axs(
            ps_grid,
            squeeze=True,
            sharex="all",
            sharey="all",
        )

        self.gss = gss

    def panel_scaling(self):
        key = "panel_scaling"
        ax_lin, ax_nonlin = self.gss[key]

        max_rel = self.params.getint("max_relevant_dims")
        n_cons = self.params.getint("n_contexts")
        n_vals = self.params.getint("n_vals")
        relevant_dims = np.arange(2, max_rel + 1)
        orders = np.arange(1, max_rel)

        cmap = self.params.get("order_cmap")
        mf_num = self.params.getfloat("max_flex_color")
        cm = plt.get_cmap(cmap)
        mf_color = cm(mf_num)
        o_colors = cm(np.linspace(0.2, 0.9, len(orders)))

        con_task_color = self.params.getcolor("contextual_color")
        lin_task_color = self.params.getcolor("partition_color")

        # any binary task
        max_flex = n_vals**relevant_dims - 1

        # any second order task
        label_orders = (1, 5, 10, 15)
        for i, o in enumerate(orders):
            oi_dim = np.array(list(ma.order_dim(rd, o) for rd in relevant_dims))
            mask = oi_dim > 0
            if o in label_orders:
                label = "O = {} tasks".format(o)
            else:
                label = ""
            ax_nonlin.plot(
                relevant_dims[mask],
                oi_dim[mask],
                color=o_colors[i],
                label=label,
            )

        # any contextual linear task
        ax_lin.plot(
            relevant_dims, relevant_dims, color=(0.8, 0.8, 0.8), label="disentangled"
        )
        # linear_basis = n_cons*((relevant_dims - n_cons + 1))
        linear_basis = n_cons * (relevant_dims - n_cons + 1)
        ax_lin.plot(
            relevant_dims,
            linear_basis,
            color=con_task_color,
            label="modular",
        )
        o2_dim = np.array(list(ma.order_dim(rd, 2) for rd in relevant_dims))
        ax_lin.plot(relevant_dims, o2_dim, color=o_colors[-2], label="min unstructured")

        ax_lin.plot(relevant_dims, max_flex, color=mf_color, label="max unstructured")
        ax_nonlin.plot(relevant_dims, max_flex, color=mf_color, label="arbitrary tasks")

        # ax_lin.set_yscale("log")
        ax_lin.set_ylim([0, 80])

        ax_lin.legend(frameon=False)
        ax_nonlin.legend(frameon=False)
        gpl.clean_plot(ax_lin, 0)
        gpl.clean_plot(ax_nonlin, 0)
        ax_lin.set_ylabel("basis\ndimensionality")
        ax_nonlin.set_ylabel("basis\ndimensionality")
        ax_nonlin.set_xlabel("latent variables")
        ax_lin.set_xlabel("latent variables")


class FigureTaskIntro(ModularizerFigure):
    def __init__(self, fig_key="task_intro_figure", colors=colors, **kwargs):
        fsize = (6, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_minimal_order",
            "panel_input_spectrum",
            "panel_generalization",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        gss["panel_task_spectrum"] = self.get_axs(
            (self.gs[10:20, 55:70],), squeeze=False, sharex="row"
        )[0, 0]

        gss["panel_minimal_order"] = self.get_axs(
            (self.gs[0:20, 80:100],), squeeze=False, sharex="row"
        )[0, 0]

        inp_rep_grid = pu.make_mxn_gridspec(self.gs, 1, 3, 25, 60, 0, 100, 8, 2)
        inp_spec_grid = pu.make_mxn_gridspec(self.gs, 1, 3, 60, 75, 10, 90, 8, 20)
        axs_rep = self.get_axs(inp_rep_grid, all_3d=True)
        axs_spec = self.get_axs(inp_spec_grid, sharex="all", sharey="all")
        gss["panel_input_spectrum"] = (axs_rep, axs_spec)

        metric_grid = pu.make_mxn_gridspec(self.gs, 1, 4, 85, 100, 0, 100, 8, 8)
        gss["panel_metrics"] = self.get_axs(metric_grid)

        self.gss = gss

    def panel_task_spectrum(self):
        key = "panel_task_spectrum"
        ax = self.gss[key]

        n_feats = self.params.getint("n_feats")
        n_tasks = self.params.getint("n_tasks")
        axis_tasks = self.params.getboolean("axis_tasks")
        if self.data.get(key) is None:
            source = u.MultiBernoulli(0.5, n_feats + 1)
            stim = source.rvs(1000)
            stim = np.concatenate((stim, np.expand_dims(1 - stim[:, -1], 1)), axis=1)

            task_func = ms.make_contextual_task_func(
                n_feats, n_tasks=n_tasks, renorm=False, axis_tasks=axis_tasks
            )
            rep = task_func(stim)
            self.data[key] = ma.compute_order_spectrum(stim, rep, n_feats)
        x_spec = np.arange(1, n_feats + 1)
        mv.plot_spectrum(x_spec, self.data[key], ax=ax)

    def panel_minimal_order(self):
        key = "panel_minimal_order"
        ax_order = self.gss[key]

        max_rel = self.params.getint("max_relevant_dims")
        order = 2
        amb_dims = self.params.getint("ambient_dims")
        n_cons = self.params.getint("n_contexts")
        n_vals = self.params.getint("n_vals")
        relevant_dims = np.arange(1, max_rel + 1)

        full_dim = ss.binom(amb_dims, relevant_dims) * n_vals**relevant_dims
        o2_dim = ss.binom(amb_dims, order) * n_vals**order
        nonlinear_basis = n_vals ** (n_cons - 1 + relevant_dims)
        linear_basis = n_cons * 2 * relevant_dims

        ax_order.plot(
            relevant_dims,
            np.ones_like(relevant_dims) * o2_dim,
            label="all second-order",
        )
        # ax_order.plot(relevant_dims, full_dim, label="relevant-order terms")
        # ax_order.plot(relevant_dims, nonlinear_basis, label="relevant nonlinear")
        ax_order.plot(relevant_dims, linear_basis, label="needed")
        ax_order.set_ylim([0, 200])
        ax_order.legend(frameon=False)
        gpl.clean_plot(ax_order, 0)
        ax_order.set_ylabel("dimensionality")
        ax_order.set_xlabel("relevant dimensions")

    def panel_input_spectrum(self):
        key = "panel_input_spectrum"
        axs_rep, axs_spec = self.gss[key]

        mixing_levels = self.params.getlist("mixing_levels", typefunc=float)
        n_feats = self.params.getint("n_feats")
        n_units = self.params.getint("n_units")
        if self.data.get(key) is None:
            spectrums = np.zeros((len(mixing_levels), n_feats))
            for i, ml in enumerate(mixing_levels):
                mddg = dg.MixedDiscreteDataGenerator(
                    n_feats, mix_strength=ml, n_units=n_units
                )
                stim, rep = mddg.sample_reps(1000, add_noise=False)
                spectrums[i] = ma.compute_order_spectrum(stim, rep, n_feats)
            self.data[key] = spectrums
        spectrums = self.data[key]
        x_spec = np.arange(1, n_feats + 1)

        lv1_color = self.params.getcolor("lv_color")
        lv2_color = self.params.getcolor("noncon_color")
        lv3_color = self.params.getcolor("con1_color")
        cols = (lv1_color, lv2_color, lv3_color)
        for i, ml in enumerate(mixing_levels):
            mddg = dg.MixedDiscreteDataGenerator(
                n_feats, mix_strength=ml, n_units=n_units
            )
            rd = mddg.code.rep_dict
            stims = []
            reps = []
            list((stims.append(k[0]), reps.append(v)) for k, v in rd.items())
            stims = np.array(stims)
            reps = np.array(reps)
            _, p = gpl.plot_highdim_points(reps, ax=axs_rep[0, i])
            for j, k in it.combinations(range(reps.shape[0]), 2):
                diff = (stims[j] - stims[k]) ** 2
                if np.sum(diff) == 4:
                    col_ind = np.argmax(diff)
                    color = cols[col_ind]
                    tr = np.stack((reps[j], reps[k]), axis=0)
                    gpl.plot_highdim_trace(tr, ax=axs_rep[0, i], p=p, color=color)
            _, p = gpl.plot_highdim_points(reps, ax=axs_rep[0, i], p=p)
            gpl.clean_3d_plot(axs_rep[0, i])
            gpl.make_3d_bars(axs_rep[0, i], bar_len=0.5)
            mv.plot_spectrum(x_spec, spectrums[i], ax=axs_spec[0, i])


class FigureControlledInput(ModularizerFigure):
    def __init__(self, fig_key="input_controlled_figure", colors=colors, **kwargs):
        fsize = (4, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_vis",
            "panel_quant",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        nl_strengths = self.params.getlist("nl_str_egs", typefunc=float)
        vis_grid = pu.make_mxn_gridspec(
            self.gs, len(nl_strengths), 2, 0, 70, 30, 100, 10, 20
        )
        gss["panel_vis"] = self.get_axs(
            vis_grid,
            squeeze=True,
        )

        quant_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 80, 100, 10, 90, 10, 20)
        gss["panel_quant"] = self.get_axs(quant_grid, squeeze=True)

        self.gss = gss

    def panel_quant(self, recompute=False):
        key = "panel_quant"
        ax_dim, ax_sep = self.gss[key]

        color = self.params.getcolor("dg_color")
        nls_args = self.params.getlist("nl_bounds", typefunc=float)
        n_nls = self.params.getint("n_nls")
        nl_strengths = np.linspace(*nls_args, n_nls)
        n_units = self.params.getint("n_units")
        n_feats = self.params.getint("n_feats")
        n_folds = self.params.getint("n_folds")
        total_power = self.params.getfloat("total_power")
        rep_kwargs = {"add_noise": True}
        if self.data.get(key) is None or recompute:
            dims = np.zeros(len(nl_strengths))
            seps = np.zeros((len(nl_strengths), n_folds))
            for i, nl in enumerate(nl_strengths):
                mddg = dg.MixedDiscreteDataGenerator(
                    n_feats,
                    mix_strength=nl,
                    n_units=n_units,
                    total_power=total_power,
                )
                dims[i] = mddg.representation_dimensionality(participation_ratio=True)
                seps[i] = ma.contextual_performance(
                    mddg, n_folds=n_folds, rep_kwargs=rep_kwargs
                )
            self.data[key] = (dims, seps)
        dims, seps = self.data[key]
        gpl.plot_trace_werr(1 - nl_strengths, dims, ax=ax_dim, color=color)
        gpl.plot_trace_werr(1 - nl_strengths, seps.T, ax=ax_sep, color=color)
        ax_dim.set_xlabel("input structure")
        ax_sep.set_xlabel("input structure")
        ax_sep.invert_xaxis()
        ax_dim.invert_xaxis()

        ax_dim.set_ylabel("dimensionality")
        ax_sep.set_ylabel("linear separability")
        gpl.add_hlines(0.5, ax_sep)

    def panel_vis(self, recompute=False):
        key = "panel_vis"
        axs = self.gss[key]

        nl_strengths = self.params.getlist("nl_str_egs", typefunc=float)
        n_units = self.params.getint("n_units")
        n_feats = self.params.getint("n_feats")
        if self.data.get(key) is None or recompute:
            inputs = {}
            for j, nl in enumerate(nl_strengths):
                mddg = dg.MixedDiscreteDataGenerator(
                    n_feats,
                    mix_strength=nl,
                    n_units=n_units,
                )
                inputs[nl] = mddg
            self.data[key] = inputs

        cmap = self.params.get("cluster_cmap")

        c1_color = self.params.getcolor("con1_color")
        c2_color = self.params.getcolor("con2_color")
        neutral_color = self.params.getcolor("noncon_color")

        input_models = self.data[key]
        for i, nl in enumerate(nl_strengths):
            im = input_models[nl]
            im_ident = self.make_ident_modularizer(fdg=im)
            ax_clust, ax_scatt = axs[i]

            mv.plot_context_clusters(
                im_ident, ax=ax_clust, cmap=cmap, context_colors=(c1_color, c2_color)
            )
            mv.plot_optimal_context_scatter(
                im_ident, ax=ax_scatt, colors=(c1_color, neutral_color, c2_color)
            )


class FigureInput(ModularizerFigure):
    def __init__(self, fig_key="input_figure", colors=colors, **kwargs):
        fsize = (4, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_act",
            "panel_tasks",
            "panel_dim_sparse",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        dim_sparse_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 70, 100, 0, 40, 10, 20)
        gss["panel_dim_sparse"] = self.get_axs(
            dim_sparse_grid, squeeze=True, sharex="row"
        )

        eg_act_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 40, 0, 40, 20, 30)
        gss["panel_eg_act"] = self.get_axs(eg_act_grid, squeeze=True, sharex="row")

        act_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 100, 60, 100, 20, 10)
        gss["panel_act"] = self.get_axs(act_grid, squeeze=True)

        # tasks_grid = self.gs[85:100, 70:]
        # gss["panel_tasks"] = self.get_axs((tasks_grid,), squeeze=False)[0, 0]

        self.gss = gss

    def panel_eg_act(self):
        key = "panel_eg_act"
        inp_ax, rep_ax = self.gss[key]

        fdg = self.make_fdg()
        n_egs = 3
        lvs, reps = fdg.sample_reps(n_egs)

        cmap = self.params.get("cluster_cmap")

        vs = np.arange(lvs.shape[1])
        us = np.arange(reps.shape[1])
        egs = np.arange(n_egs)
        gpl.pcolormesh(vs, egs, lvs, ax=inp_ax, cmap=cmap, rasterized=True)
        gpl.pcolormesh(us, egs, reps, ax=rep_ax, cmap=cmap, rasterized=True)

        gpl.clean_plot(inp_ax, 0)
        gpl.clean_plot(rep_ax, 0)
        inp_ax.set_xticks([0, 10, 20])
        rep_ax.set_xticks([0, 200, 400])
        inp_ax.set_xlabel("latent variables")
        inp_ax.set_ylabel("stimuli")
        rep_ax.set_xlabel("input units")
        rep_ax.set_ylabel("input reps")

    def panel_dim_sparse(self):
        key = "panel_dim_sparse"
        ax_sparse, ax_dims = self.gss[key]

        fdg = self.make_fdg()
        lv_dim = fdg.input_dim
        lvs, reps = fdg.sample_reps(10000)

        lv_sparse = u.quantify_sparseness(lvs)

        rep_sparse = u.quantify_sparseness(reps)
        rep_sparse = rep_sparse[~np.isnan(rep_sparse)]
        rep_pr = u.participation_ratio(reps)

        lv_color = self.params.getcolor("lv_color")
        rep_color = self.params.getcolor("dg_color")

        gpl.violinplot(
            [lv_sparse],
            [0],
            color=[lv_color],
            ax=ax_sparse,
            showextrema=False,
            showmedians=True,
        )
        gpl.violinplot(
            [rep_sparse],
            [1],
            color=[rep_color],
            ax=ax_sparse,
            showextrema=False,
            showmedians=True,
        )

        ax_dims.plot([0], lv_dim, "o", color=lv_color)
        ax_dims.plot([1], rep_pr, "o", color=rep_color)

        gpl.clean_plot(ax_sparse, 0)
        gpl.clean_plot(ax_dims, 0)
        ax_sparse.set_xticks([0, 1])
        ax_sparse.set_xticklabels(["latent variables", "input model"], rotation=90)
        ax_dims.set_xticks([0, 1])
        ax_dims.set_xticklabels(["latent variables", "input model"], rotation=90)
        ax_dims.set_ylabel("dimensionality")
        ax_sparse.set_ylabel("sparseness")

        ax_dims.set_xlim([-0.2, 1.2])
        gpl.clean_plot_bottom(ax_sparse, keeplabels=True)
        gpl.clean_plot_bottom(ax_dims, keeplabels=True)

    def panel_tasks(self):
        key = "panel_tasks"
        ax = self.gss[key]

        n_tasks = self.params.getint("n_tasks")

        if self.data.get(key) is None:
            modu = self.make_ident_modularizer(
                linear=True,
                tasks_per_group=n_tasks,
            )
            self.data[key] = ma.task_performance_learned(
                modu,
            )

        perf = self.data[key]
        ax.hist(np.mean(perf, axis=1))

    def panel_act(self):
        key = "panel_act"
        ax_clust, ax_scatt = self.gss[key]

        cmap = self.params.get("cluster_cmap")

        c1_color = self.params.getcolor("con1_color")
        c2_color = self.params.getcolor("con2_color")
        neutral_color = self.params.getcolor("noncon_color")

        modu = self.make_ident_modularizer()
        mv.plot_context_clusters(
            modu, ax=ax_clust, cmap=cmap, context_colors=(c1_color, c2_color)
        )
        mv.plot_context_scatter(
            modu, ax=ax_scatt, colors=(c1_color, neutral_color, c2_color)
        )
        ax_scatt.set_xlabel("activity in context 1")
        ax_scatt.set_ylabel("activity in context 2")
        ax_clust.set_xlabel("units")
        ax_clust.set_ylabel("")


class FigureControlledGeometry(ModularizerFigure):
    def __init__(self, fig_key="geom_gen_figure", colors=colors, **kwargs):
        fsize = (7, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_eg_geom",
            "panel_geom",
            "panel_unit_align",
            "panel_zs",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        full_ax = self.get_axs(
            (self.gs[20:40, :20],),
            all_3d=True,
            squeeze=False,
        )
        eg_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 60, 20, 60, 2, 2)
        eg_axs = self.get_axs(eg_grid, all_3d=True, squeeze=True)
        gss["panel_eg_geom"] = full_ax, eg_axs

        geom_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 60, 70, 100, 10, 10)
        gss["panel_geom"] = self.get_axs(geom_grid, squeeze=True)

        zs_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 60, 100, 70, 100, 10, 10)
        gss["panel_zs"] = self.get_axs((self.gs[70:100, 70:100],), squeeze=False)[0, 0]

        zs_eg_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 60, 100, 20, 60, 2, 2)
        gss["panel_eg_zs"] = self.get_axs(
            zs_eg_grid,
            all_3d=True,
            squeeze=True,
        )

        self.gss = gss

    def panel_eg_geom(self, refit=False):
        key = "panel_eg_geom"
        f_ax, axs = self.gss[key]

        if self.data.get(key) is None or refit:
            self.data[key] = self.train_eg_networks()

        (n_tasks, nl_strs), models, hs = self.data[key]
        models = models.T

        f1_color = self.params.getcolor("f1_color")
        f2_color = self.params.getcolor("f2_color")
        f3_color = self.params.getcolor("f2_color")

        r1_color = self.params.getcolor("r1_color")
        r2_color = self.params.getcolor("r2_color")

        for ind in u.make_array_ind_iterator(axs.shape):
            ax, p = mv.visualize_module_activity(
                models[ind],
                0,
                line_colors=(f1_color, f2_color, f3_color),
                resp_colors=(r1_color, r2_color),
                ms=5,
                ax=axs[ind],
            )

    def panel_geom(self, reload_=False):
        key = "panel_geom"
        axs = self.gss[key]

        template = self.params.get("nls_template_big")
        nl_inds = self.params.getlist("nls_big_ids")
        plot_keys = ("within_ccgp", "shattering")
        labels = {
            "within_ccgp": "abstraction",
            "shattering": "shattering",
        }
        cms = ("Blues", "Oranges")
        if self.data.get(key) is None or reload_:
            out_arrs, n_parts, mixes = self.load_nls_runs(template, nl_inds, plot_keys)
            self.data[key] = out_arrs, n_parts, mixes

        out_arrs, n_parts, mixes = self.data[key]
        for i, pk in enumerate(plot_keys):
            arr = np.mean(out_arrs[pk], axis=2)
            img = gpl.pcolormesh(
                n_parts,
                1 - mixes,
                arr,
                ax=axs[i],
                cmap=cms[i],
                vmin=0.5,
                rasterized=True,
            )
            self.f.colorbar(img, ax=axs[i], label=labels[pk])

            axs[i].set_xlabel("number of tasks")
            axs[i].set_ylabel("input structure")
            axs[i].set_xticks([n_parts[0], 10, n_parts[-1]])
            axs[i].set_yticks([0, 0.5, 1])
            axs[i].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    def panel_zs(self, reload_=False):
        key = "panel_zs"
        ax = self.gss[key]

        plot_key = "zero shot"
        if self.data.get(key) is None or reload_:
            self.data[key] = self.load_and_organize_con_sweep(
                (plot_key,),
                reload=reload_,
                folder_key="sim_folder",
                run_key="zs_run_inds",
                load_num=1,
            )
        cm = plt.get_cmap(self.params.get("zs_cmap"))
        mix_sort, task_sort, metric_dict = self.data[key]
        zs_p = metric_dict[plot_key][0]
        zs_ood = zs_p[..., 0, :]
        zs_ood = np.mean(zs_ood, axis=(-2, -1))
        m = gpl.pcolormesh(
            task_sort,
            1 - mix_sort,
            zs_ood,
            ax=ax,
            cmap=cm,
            vmin=0.5,
            vmax=1,
            rasterized=True,
        )
        self.f.colorbar(m, ax=ax, label="zero shot\ngeneralization")
        ax.set_xticks([task_sort[0], 10, task_sort[-1]])
        ax.set_yticks([mix_sort[0], 0.5, mix_sort[-1]])
        ax.set_xlabel("number of tasks")
        ax.set_ylabel("input structure")
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    def panel_eg_zs(self, retrain=False):
        key = "panel_eg_zs"
        axs = self.gss[key]

        nl_zs = self.params.getlist("zs_eg_nls", typefunc=float)
        fix_n_vars = self.params.getint("zs_fix_n_vars")
        if self.data.get(key) is None or retrain:
            out_dict = {}
            for nlz in nl_zs:
                mddg_nlz = self.make_mddg(nlz)
                ood, ind, (m, h) = ma.zero_shot_training(
                    mddg_nlz,
                    n_tasks=5,
                    group_size=2,
                    n_overlap=2,
                    fix_n_irrel_vars=fix_n_vars,
                    verbose=False,
                )
                out_dict[nlz] = m
            self.data[key] = out_dict

        f1_color = self.params.getcolor("f1_color")
        f2_color = self.params.getcolor("f2_color")

        r1_color = self.params.getcolor("r1_color")
        r2_color = self.params.getcolor("r2_color")

        out_dict = self.data[key]
        for i, nlz in enumerate(nl_zs):
            m = out_dict[nlz]
            irrel = m.irrel_vars[:fix_n_vars]
            inp_rep, _, _ = m.get_x_true(group_inds=0)
            rep = m.get_representation(inp_rep)
            p = skd.PCA(3)
            p = p.fit(rep)
            _, p = mv.visualize_module_activity(
                m,
                0,
                line_colors=(f1_color, f2_color),
                resp_colors=(r1_color, r2_color),
                ms=5,
                ax=axs[i],
                fix_vars=irrel,
                fix_value=0,
                p=p,
                linestyle="dashed",
            )
            _, p = mv.visualize_module_activity(
                m,
                0,
                line_colors=(f1_color, f2_color),
                resp_colors=(r1_color, r2_color),
                ms=5,
                ax=axs[i],
                fix_vars=irrel,
                fix_value=1,
                p=p,
            )


class FigureEmergence(ModularizerFigure):
    def __init__(self, fig_key="emergence_figure", colors=colors, **kwargs):
        fsize = (7, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ("panel_vis_stable",)
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        vis_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 0, 40, 0, 100, 10, 10)
        vis_axs = self.get_axs(vis_grid, all_3d=True, squeeze=True)
        dir_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 50, 100, 0, 100, 10, 10)
        dir_axs = self.get_axs(dir_grid, squeeze=True)
        gss["panel_vis_stable"] = (vis_axs, dir_axs)

        self.gss = gss

    def panel_vis_stable(self, recompute=False):
        key = "panel_vis_stable"
        axs_vis, axs_dirs = self.gss[key]

        ntl = 5  # self.params.getint('n_tasks_low')
        nth = 100  # self.params.getint('n_tasks_high')
        n_latents = 2

        nts = (ntl, nth)
        if self.data.get(key) is None or recompute:
            stable_gates = {}
            for nt in nts:
                stims, targs, _ = ma.make_target_funcs(
                    n_latents,
                    nt,
                    renorm_targets=True,
                    renorm_stim=True,
                    axis_tasks=True,
                )
                svs, hist = ma.compute_stable_wvs(stims, targs)
                thr = 0.1
                svs = svs[hist[-1, :] < thr]
                model_out = ms.make_linear_network(
                    stims, targs, use_relu=True, verbose=False
                )
                ws_all = model_out[0].weights[0].numpy().T
                ws = u.make_unit_vector(ws_all)
                stable_gates[nt] = (stims, targs, svs, ws, model_out)
            self.data[key] = stable_gates

        stable_gates = self.data[key]
        for i, nt in enumerate(nts):
            stims, targs, gates, ws, model_out = stable_gates[nt]
            resp = model_out[1](stims)
            mask_c1 = stims[:, -1] == 1

            # print(gates)
            # print(ws[:10])
            resp_c1 = np.mean(resp[mask_c1], axis=0)
            resp_c2 = np.mean(resp[~mask_c1], axis=0)
            # axs_vis[i].plot(resp_c1, resp_c2, 'o')
            mv.visualize_stable_gates(stims, targs, gates, ws=ws, ax=axs_vis[i])
            axs_vis[i].view_init(0, 50)
            mv.visualize_gate_angle(stims, targs, gates, ws=ws, ax=axs_dirs[i])


class FigureModularity(ModularizerFigure):
    def __init__(self, fig_key="modularity_discrete", colors=colors, **kwargs):
        fsize = (8, 7)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_ri",
            "panel_task_compare",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        ri_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 60, 100, 80, 100, 10, 10)
        ri_axs = self.get_axs(ri_grid, sharex="all", squeeze=True)
        gss["panel_ri"] = ri_axs

        n_plots = len(self.params.getlist("n_eg_tasks"))
        tc_grid = pu.make_mxn_gridspec(self.gs, 3, n_plots, 50, 100, 20, 70, 10, 8)
        tc_axs = self.get_axs(tc_grid, sharey="horizontal", sharex="horizontal")
        gss["panel_task_compare"] = tc_axs

        self.gss = gss

    def panel_ri(self):
        key = "panel_ri"
        axs_all = self.gss[key]
        quant_keys = ("model_frac", "diff_act_ablation")
        y_labels = ("cluster fraction", "ablation effect")
        ri_list = self.params.getlist("ri_list")

        label_dict = {(3,): "D = 3", (5,): "D = 5", (8,): "D = 8"}
        colors = {
            (3,): self.params.getcolor("l3_color"),
            (5,): self.params.getcolor("l5_color"),
            (8,): self.params.getcolor("l8_color"),
        }
        labels = ("D = 3", "D = 5", "D = 8")

        nulls = (0, 0, 0.5, 0.5)
        self._quantification_panel(
            quant_keys,
            ri_list,
            axs_all,
            label_dict=label_dict,
            nulls=nulls,
            plot_ylabels=y_labels,
            colors=colors,
        )
        axs_all[-1].set_xlabel("tasks")

    def panel_task_compare(self, refit_models=False, recompute_ablation=False):
        key = "panel_task_compare"
        axs_all = self.gss[key]

        # maybe also add high-dim visualization?
        n_tasks = self.params.getlist("n_eg_tasks", typefunc=int)
        act_cmap = self.params.get("activity_cmap")
        ablation_cmap = self.params.get("ablation_cmap")

        if self.data.get(key) is None or refit_models:
            models = []
            hists = []
            abls = []
            fdg = self.make_fdg()
            for i, nt in enumerate(n_tasks):
                m_i, h_i = self.train_modularizer(tasks_per_group=nt)
                tc_i = ma.act_ablation(
                    m_i,
                    single_number=False,
                )

                models.append(m_i)
                hists.append(h_i)
                abls.append(tc_i)
            self.data[key] = (fdg, models, hists, abls)

        fdg, models, hists, abls = self.data[key]
        c1_color = self.params.getcolor("con1_color")
        c2_color = self.params.getcolor("con2_color")
        neutral_color = self.params.getcolor("noncon_color")

        colors = (c1_color, neutral_color, c2_color)
        for i, nt in enumerate(n_tasks):
            m_i = models[i]
            tc_i = abls[i]
            if recompute_ablation:
                tc_i = ma.act_ablation(
                    m_i,
                    single_number=False,
                )

            axs_i = axs_all[:, i]
            mv.plot_context_clusters(
                m_i,
                ax=axs_i[0],
                cmap=act_cmap,
                context_colors=(c1_color, c2_color),
            )
            mv.plot_context_scatter(m_i, ax=axs_i[1], colors=colors)

            if i > 0:
                axs_i[1].set_ylabel("")
            tc_i[tc_i < 0] = 0
            m = axs_i[2].imshow(tc_i, cmap=ablation_cmap)
            self.f.colorbar(m, ax=axs_i[2], label="performance\nchange")
            axs_i[2].set_xlabel("context")
            axs_i[2].set_ylabel("inferred cluster")
            axs_i[2].set_yticks([0, 1, 2])


class FigureControlled(ModularizerFigure):
    def __init__(self, fig_key="controlled", colors=colors, **kwargs):
        fsize = (6, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ("panel_consequences",)
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        cons_grid = pu.make_mxn_gridspec(self.gs, 4, 3, 0, 100, 0, 100, 10, 10)
        gss["panel_consequences"] = self.get_axs(cons_grid, squeeze=True)

        self.gss = gss

    def panel_consequences(self):
        key = "panel_consequences"
        axs = self.gss[key]

        run_ind = self.params.get("consequences_run_ind")
        run_ind_rand = self.params.get("consequences_run_ind_random")
        controlled_template = self.params.get("controlled_template")
        controlled_folder = self.params.get("controlled_folder")

        mod_color = self.params.getcolor("partition_color")
        naive_color = self.params.getcolor("naive_color")

        out_dict = maux.load_consequence_runs(
            run_ind,
            folder=controlled_folder,
            template=controlled_template,
            ref_key="dm_input_mixing",
        )
        out_dict_rand = maux.load_consequence_runs(
            run_ind_rand,
            folder=controlled_folder,
            template=controlled_template,
            ref_key="dm_input_mixing",
        )
        mixing = 0
        plot_dict = out_dict[mixing]
        plot_dict_rand = out_dict_rand[mixing]
        self._plot_learning_cons(
            plot_dict,
            plot_dict_rand,
            axs[:, 0],
            colors=(naive_color, mod_color),
        )
        mixing = 100
        plot_dict = out_dict[mixing]
        plot_dict_rand = out_dict_rand[mixing]
        self._plot_learning_cons(
            plot_dict,
            plot_dict_rand,
            axs[:, 2],
            colors=(naive_color, mod_color),
        )

        # mv.plot_cumulative_learning_cons(
        #     out_dict,
        #     out_dict_rand,
        #     axs=axs[:, 1]
        # )


class FigureConsequences(ModularizerFigure):
    def __init__(self, fig_key="controlled", colors=colors, **kwargs):
        fsize = (6, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ("panel_consequences",)
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        cons_grid1 = pu.make_mxn_gridspec(self.gs, 1, 3, 0, 40, 0, 100, 10, 10)
        axs1 = self.get_axs(
            cons_grid1,
            squeeze=True,
        )
        cons_grid2 = pu.make_mxn_gridspec(self.gs, 1, 3, 60, 100, 0, 100, 10, 20)
        axs2 = self.get_axs(
            cons_grid2,
            squeeze=True,
        )
        gss["panel_consequences"] = np.stack((axs1, axs2), axis=0)

        self.gss = gss

    def panel_consequences(self, recompute=False):
        key = "panel_consequences"
        axs_map, axs_trace = self.gss[key]

        plot_keys = ("new task tasks", "related context tasks", "new context tasks")
        if self.data.get(key) is None or recompute:
            self.data[key] = self.load_and_organize_con_sweep(
                plot_keys, reload=recompute
            )
        mix_sort, task_sort, metric_dict = self.data[key]

        task_fix = 2
        plot_mix = (0, 1)
        trace_colors = (
            self.params.getcolor("disentangled_color"),
            self.params.getcolor("unstructured_color"),
        )
        cm = plt.get_cmap(self.params.get("diverge_cmap"))
        labels = ("novel task", "related context", "unrelated context")

        for i, pk in enumerate(plot_keys):
            pre_pk, null_pk = metric_dict[pk]
            diff = np.mean(np.sum(pre_pk - null_pk, axis=3), axis=2)
            diff = np.mean(np.sum(pre_pk, axis=3), axis=2)
            # bound = np.max(np.abs(diff))
            m = gpl.pcolormesh(
                task_sort,
                1 - mix_sort,
                diff,
                ax=axs_map[i],
                cmap=cm,
                rasterized=True,
                # vmin=-bound,
                # vmax=bound
            )
            self.f.colorbar(m, ax=axs_map[i], label="learning speed")
            axs_map[i].set_title(labels[i])
            n_epochs = pre_pk.shape[-1]
            epochs = np.arange(n_epochs)
            for j, pm in enumerate(plot_mix):
                mix_mask = mix_sort == pm
                task_mask = task_sort == task_fix
                pre_masked = np.squeeze(pre_pk[mix_mask][:, task_mask])
                axs_trace[i].plot(
                    epochs, pre_masked.T, color=trace_colors[j], alpha=0.1
                )
                if i == 0:
                    label = "input structure = {}".format(1 - pm)
                else:
                    label = ""
                gpl.plot_trace_werr(
                    epochs,
                    pre_masked,
                    ax=axs_trace[i],
                    color=trace_colors[j],
                    label=label,
                )
            gpl.add_hlines(0.5, axs_trace[i])
            axs_trace[i].set_xlabel("training epoch")
            axs_trace[i].set_ylabel("task performance")
            axs_map[i].set_xticks([task_sort[0], 10, task_sort[-1]])
            axs_map[i].set_yticks([mix_sort[0], 0.5, mix_sort[-1]])
            axs_map[i].set_xlabel("number of tasks")
            axs_map[i].set_ylabel("input structure")


def _combine_binary_arrs(binary_arrs, colors):
    comb_arr = np.zeros(binary_arrs[0].shape)
    for i, ba in enumerate(binary_arrs):
        group = (comb_arr == 0) * ba * (i + 1)
        comb_arr = group + comb_arr
    cmap = gpl.make_discrete_cmap(*colors)
    return comb_arr, cmap


class FigureModularityIsolated(ModularizerFigure):
    def __init__(self, fig_key="isolated_rep", colors=colors, **kwargs):
        fsize = (6, 4.4)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_eg_traces",
            "panel_param_sweep",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        eg_grid = pu.make_mxn_gridspec(self.gs, 1, 3, 0, 30, 0, 100, 10, 10)
        eg_axs = self.get_axs(eg_grid, squeeze=True)
        tr_grid = pu.make_mxn_gridspec(self.gs, 1, 1, 35, 65, 0, 50, 10, 10)
        tr_axs = self.get_axs(tr_grid)[0, 0]
        ps_grid = pu.make_mxn_gridspec(self.gs, 1, 3, 70, 100, 0, 100, 10, 10)
        sweep_axs = self.get_axs(
            ps_grid,
            squeeze=True,
        )
        gss["panel_traces_sweep"] = (eg_axs, tr_axs, sweep_axs)

        self.gss = gss

    def panel_traces_sweep(self, reload_=False):
        key = "panel_traces_sweep"
        axs_eg, ax_tr, axs_sweep = self.gss[key]
        template = self.params.get("nls_template_big")
        inds_lin = self.params.getlist("run_inds_linear")
        inds_nonlin = self.params.getlist("run_inds_nonlinear")
        inds_full = self.params.getlist("run_inds_full")
        plot_keys = ("corr_rate", "model_frac", "val_loss")

        if self.data.get(key) is None or reload_:
            out_arrs_lin, n_parts, mixes = self.load_nls_runs(
                template, inds_lin, plot_keys
            )
            out_arrs_nonlin, n_parts, mixes = self.load_nls_runs(
                template, inds_nonlin, plot_keys
            )
            out_arrs_full, n_parts, mixes = self.load_nls_runs(
                template, inds_full, plot_keys
            )
            self.data[key] = (
                n_parts,
                mixes,
                out_arrs_lin,
                out_arrs_nonlin,
                out_arrs_full,
            )
        n_parts, mixes, out_arrs_lin, out_arrs_nonlin, out_arrs_full = self.data[key]
        mix_inds = (1, 10, 19)
        part_ind = 5
        tr_key = "loss"
        mod_key = "model_frac"
        for i, mi in enumerate(mix_inds):
            tr_lin = out_arrs_lin[tr_key][mi, part_ind]
            tr_nonlin = out_arrs_nonlin[tr_key][mi, part_ind]
            epochs = np.arange(tr_lin.shape[1])
            gpl.plot_trace_werr(epochs, tr_lin, ax=axs_eg[i], log_y=True)
            gpl.plot_trace_werr(epochs, tr_nonlin, ax=axs_eg[i], log_y=True)

        first_n_epochs = 3
        tr_diff = (
            out_arrs_lin[tr_key][..., :first_n_epochs]
            - out_arrs_nonlin[tr_key][..., :first_n_epochs]
        )
        tr_mu_diff = np.nanmean(tr_diff, axis=-1)
        print(mixes.shape, n_parts.shape)
        gpl.plot_trace_werr(mixes, tr_mu_diff[:, part_ind].T, ax=ax_tr)
        mods = out_arrs_full[mod_key][:, part_ind].T
        # gpl.plot_trace_werr(mixes, mods, ax=ax_tr)
        gpl.add_hlines(0, ax_tr)
        


class FigureModularityControlled(ModularizerFigure):
    def __init__(self, fig_key="controlled_rep", colors=colors, **kwargs):
        fsize = (6, 4.4)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_eg_networks",
            "panel_param_sweep",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        eg_grid = pu.make_mxn_gridspec(self.gs, 2, 4, 0, 55, 0, 100, 10, 10)
        eg_axs = self.get_axs(eg_grid, squeeze=True)
        gss["panel_eg_networks"] = eg_axs

        ps_grid = pu.make_mxn_gridspec(self.gs, 1, 3, 70, 100, 0, 100, 10, 10)
        gss["panel_param_sweep"] = self.get_axs(
            ps_grid,
            squeeze=True,
        )

        self.gss = gss

    def panel_eg_networks(self, retrain=False):
        key = "panel_eg_networks"
        axs = self.gss[key]

        if self.data.get(key) is None or retrain:
            self.data[key] = self.train_eg_networks()

        c1_color = self.params.getcolor("con1_color")
        c2_color = self.params.getcolor("con2_color")
        neutral_color = self.params.getcolor("noncon_color")
        colors = (c1_color, neutral_color, c2_color)

        (n_tasks, nl_strengths), models, _ = self.data[key]
        for i, j in u.make_array_ind_iterator(models.shape):
            n_t = n_tasks[i]
            n_l = nl_strengths[j]
            print("n_tasks", n_t, "      nl_str", n_l)
            mv.plot_context_clusters(
                models[i, j],
                ax=axs[j, i * 2],
                context_colors=(c1_color, c2_color),
            )
            mv.plot_optimal_context_scatter(
                models[i, j],
                ax=axs[j, i * 2 + 1],
                colors=colors,
            )
            print(ma.compute_alignment_index(models[i, j]))
            print(ma.compute_frac_contextual(models[i, j]))

    def panel_sweep_metrics(self, reload_=False):
        key = "panel_param_sweep"
        axs = self.gss[key]

        ax_focus = axs[-1]
        template = self.params.get("nls_template_big")
        nl_inds = self.params.getlist("nls_big_ids")
        print(nl_inds, template)
        plot_keys = ("model_frac", "alignment_index")
        axs = axs[: len(plot_keys)]
        labels = {
            "model_frac": "fraction of units",
            "alignment_index": "subspace specialization",
        }
        cms = ("Blues", "Oranges")
        if self.data.get(key) is None or reload_:
            out_arrs, n_parts, mixes = self.load_nls_runs(template, nl_inds, plot_keys)
            self.data[key] = out_arrs, n_parts, mixes
        out_arrs, n_parts, mixes = self.data[key]

        nl_mask = mixes >= 0.5
        t_mask = n_parts <= 10
        plot_thresh = {"model_frac": 0.05, "alignment_index": 0.05}
        binary_arrs = []
        arr_colors = []
        for i, pk in enumerate(plot_keys):
            arr = np.mean(out_arrs[pk], axis=2)
            img = gpl.pcolormesh(
                n_parts,
                1 - mixes,
                arr,
                ax=axs[i],
                cmap=cms[i],
                vmin=0,
                rasterized=True,
            )
            self.f.colorbar(img, ax=axs[i], label=labels[pk])

            sub_arr = arr[nl_mask][:, t_mask]
            binary_arrs.append(sub_arr > plot_thresh[pk])
            arr_colors.append(plt.get_cmap(cms[i])(0.7))
            axs[i].set_xlabel("number of tasks")
            axs[i].set_ylabel("input structure")
            axs[i].set_xticks([n_parts[0], 10, n_parts[-1]])
            axs[i].set_yticks([0, 0.5, 1])
            axs[i].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        combined_arr, cm_comb = _combine_binary_arrs(binary_arrs, arr_colors)
        gpl.pcolormesh(
            n_parts[t_mask],
            1 - mixes[nl_mask],
            combined_arr,
            cmap=cm_comb,
            vmin=0,
            ax=ax_focus,
            rasterized=True,
        )
        ax_focus.set_xticks([n_parts[0], 10])
        ax_focus.set_yticks([0, 0.5])
        ax_focus.set_xlabel("number of tasks")
        ax_focus.set_ylabel("input structure")
        ax_focus.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)

    def panel_param_sweep(self, reload_=False):
        key = "panel_param_sweep"
        ax = self.gss[key]

        template = self.params.get("model_nls_template")
        template_new = self.params.get("model_nls_template_new")
        nl_inds = self.params.getlist("nl_runs")
        task_inds = self.params.getlist("task_runs")
        if self.data.get(key) is None or reload_:
            nl_loaded = {}
            for i, ind in enumerate(nl_inds):
                out = self.load_run(
                    ind,
                    ordering_func=maux.get_nl_strength,
                    file_template=template,
                )
                nl_loaded[ind] = out
            task_loaded = {}
            for i, ind in enumerate(task_inds):
                out = self.load_run(
                    ind,
                    file_template=template_new,
                )
                task_loaded[ind] = out

            self.data[key] = nl_loaded, task_loaded
        nl_loaded, task_loaded = self.data[key]
        plot_key = "model_frac"
        label_key = "tasks_per_group"
        for k, (quants, order, args) in task_loaded.items():
            label = "T = {}".format(args[label_key][0])
            gpl.plot_trace_werr(order, quants[plot_key].T, ax=ax, label=label)


class FigureGeometryConsequences(ModularizerFigure):
    def __init__(self, fig_key="geometry_consequences", colors=colors, **kwargs):
        fsize = (7, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_new_task",
            "panel_new_context",
            "panel_geometry",
            "panel_specialization",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        side = 0
        nt_grid = pu.make_mxn_gridspec(self.gs, 1, 4, 75, 100, side, 100 - side, 10, 10)
        nt_axs = self.get_axs(nt_grid, squeeze=True, sharex="all", sharey="all")
        # gss['panel_new_task'] = nt_axs[0, 0]
        # gss['panel_new_context'] = nt_axs[0, 1]
        gss["panel_learning_consequences"] = nt_axs

        geom_grid = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 60, 30, 70, 10, 10)
        vis_grid = pu.make_mxn_gridspec(self.gs, 3, 1, 0, 60, 0, 20, 10, 10)
        geom_axs = self.get_axs(geom_grid, sharey="all", sharex="row")
        vis_axs = self.get_axs(vis_grid, all_3d=True)
        gss["panel_vis_geom"] = np.concatenate((vis_axs, geom_axs), axis=1)

        specialization_grid = pu.make_mxn_gridspec(
            self.gs, 3, 1, 0, 60, 80, 100, 10, 10
        )

        gss["panel_specialization"] = self.get_axs(specialization_grid)

        self.gss = gss

    def panel_vis_geom(self):
        key = "panel_vis_geom"
        axs = self.gss[key]

        if self.data.get(key) is None:
            modu = self.train_modularizer()
            ident_modu = self.make_ident_modularizer(linear=False)
            fdg = self.make_fdg()

            out = ma.apply_geometry_model_list([ident_modu], fdg)
            self.data[key] = (ident_modu, modu, out)

        ident_modu, (modu, modu_h), (shatter, w_ccgp, _) = self.data[key]

        dg_color = self.params.getcolor("dg_color")
        mod_color = self.params.getcolor("partition_color")

        mv.visualize_module_activity(
            ident_modu,
            0,
            ax=axs[1, 0],
            pt_color=dg_color,
            pt_alpha=0.1,
            line_color=dg_color,
            ms=1,
        )
        mv.visualize_module_activity(
            modu,
            0,
            ax=axs[2, 0],
            pt_color=mod_color,
            pt_alpha=0.1,
            line_color=mod_color,
            ms=1,
        )

        ccgp_color_wi = self.params.getcolor("ccgp_color_wi")
        ccgp_color_ac = self.params.getcolor("ccgp_color_ac")
        shattering_color = self.params.getcolor("shattering_color")

        gpl.violinplot(
            [w_ccgp[0, 0].flatten()], [0], ax=axs[1, 1], color=[ccgp_color_wi]
        )
        gpl.violinplot(
            [shatter[0, 0].flatten()], [0], ax=axs[1, 2], color=[shattering_color]
        )
        axs[1, 1].set_ylabel("classifier\ngeneralization")
        axs[1, 2].set_ylabel("shattering\ndimensionality")
        gpl.clean_plot(axs[1, 1], 0)
        gpl.clean_plot(axs[1, 2], 0)
        gpl.clean_plot_bottom(axs[1, 1])
        gpl.clean_plot_bottom(axs[1, 2])
        xl = [-0.5, 0.5]
        axs[1, 1].set_xlim(xl)
        axs[1, 2].set_xlim(xl)

        gpl.add_hlines(0.5, axs[1, 1])
        gpl.add_hlines(0.5, axs[1, 2])

        ri_list = self.params.getlist("run_inds_linear")
        colors = (
            self.params.getcolor("l3_color"),
            self.params.getcolor("l5_color"),
            self.params.getcolor("l8_color"),
        )
        labels = ("D = 3", "D = 5", "D = 8")
        for i, ri in enumerate(ri_list):
            run_data, order, _ = self.load_run(ri)

            w_ccgp = run_data["within_ccgp"].T
            gpl.plot_trace_werr(
                order,
                w_ccgp,
                ax=axs[2, 1],
                log_x=True,
                color=colors[i],
                label=labels[i],
            )
            axs[2, 1].set_ylabel("classifier\ngeneralization")
            axs[2, 1].set_xlabel("tasks")
            gpl.add_hlines(0.5, axs[2, 1])

            shatter = run_data["shattering"].T
            gpl.plot_trace_werr(
                order, shatter, ax=axs[2, 2], log_x=True, color=colors[i]
            )
            axs[2, 2].set_ylabel("shattering\ndimensionality")
            axs[2, 2].set_xlabel("tasks")
            gpl.add_hlines(0.5, axs[2, 2])

    def panel_learning_consequences(self):
        key = "panel_learning_consequences"
        axs = self.gss[key]

        run_ind = self.params.get("consequences_run_ind")
        run_ind_rand = self.params.get("consequences_run_ind_random")
        n_tasks = self.params.getint("consequences_n_tasks")

        mod_color = self.params.getcolor("partition_color")
        naive_color = self.params.getcolor("naive_color")

        out_dict = maux.load_consequence_runs(run_ind)
        out_dict_rand = maux.load_consequence_runs(run_ind_rand)
        plot_dict = out_dict[n_tasks]
        plot_dict_rand = out_dict_rand[n_tasks]

        self._plot_learning_cons(
            plot_dict,
            plot_dict_rand,
            axs,
            colors=(naive_color, mod_color),
        )

    def panel_specialization(self, recompute=False):
        key = "panel_specialization"
        axs = self.gss[key]

        if self.data.get(key) is None or recompute:
            mod, _ = self.train_modularizer()
            out = ma.compute_model_alignment(mod)
            self.data[key] = (mod, out)

        mod, out = self.data[key]
        (weights, corrs), u_inds, clusts = out
        ws_mu = np.mean(weights, axis=2)
        n_feats = weights.shape[1]
        ms = 1

        cont_color = self.params.getcolor("grey_color")

        u_clusts, counts = np.unique(clusts, return_counts=True)
        excl_clust = np.max(counts) == counts
        u_clusts = u_clusts[~excl_clust]

        cluster_colors = (
            self.params.getcolor("con1_color"),
            self.params.getcolor("con2_color"),
        )
        for i, c_i in enumerate(u_clusts):
            xs, ys = [], []
            mask = clusts == c_i
            for f1, f2 in it.combinations(range(n_feats), 2):
                xs.append(ws_mu[0, f1, mask])
                ys.append(ws_mu[0, f2, mask])
                xs.append(ws_mu[1, f1, mask])
                ys.append(ws_mu[1, f2, mask])

            mv.visualize_decoder_weights(
                np.concatenate(xs),
                np.concatenate(ys),
                ax=axs[0, 0],
                ms=ms,
                color=cluster_colors[i],
                contour_color=(cont_color,),
                cluster_labels=clusts,
            )

            xs = []
            ys = []
            for f1, f2 in it.product(range(n_feats), repeat=2):
                xs.append(ws_mu[0, f1, mask])
                ys.append(ws_mu[1, f2, mask])

            mv.visualize_decoder_weights(
                np.concatenate(xs),
                np.concatenate(ys),
                color=cluster_colors[i],
                ax=axs[1, 0],
                ms=ms,
                contour_color=(cont_color,),
                cluster_labels=clusts,
            )

        axs[0, 0].set_title("within contexts")
        axs[1, 0].set_title("across contexts")
        gpl.clean_plot(axs[0, 0], 0)
        gpl.clean_plot(axs[1, 0], 0)
        gpl.clean_plot_bottom(axs[0, 0])
        axs[1, 0].set_xlabel(r"$f_{i}$ decoder weight")
        axs[0, 0].set_ylabel(r"$f_{j}$ decoder weight")

    def panel_new_context(self):
        key = "panel_new_context"
        ax = self.gss[key]

        train_epochs = self.params.getint("context_train_epochs")
        train_samps = self.params.getint("context_train_samples")
        n_tasks = self.params.getint("n_tasks")
        fdg = self.make_fdg()
        if self.data.get(key) is None:
            self.data[key] = ma.new_context_training(
                fdg,
                total_groups=3,
                n_tasks=n_tasks,
                train_samps=train_samps,
                train_epochs=train_epochs,
            )
        (_, pretrain_history), (_, naive_history) = self.data[key]
        xs = np.arange(train_epochs)

        key = "val_loss"
        log_y = False
        gpl.plot_trace_werr(
            xs,
            pretrain_history.history[key],
            ax=ax,
            log_y=log_y,
            label="pretrained network",
        )
        gpl.plot_trace_werr(
            xs, naive_history.history[key], ax=ax, log_y=log_y, label="naive network"
        )

    def panel_new_task(self):
        key = "panel_new_task"
        ax = self.gss[key]

        train_epochs = self.params.getint("task_train_epochs")
        train_samps = self.params.getint("task_train_samples")
        n_tasks = self.params.getint("n_tasks")

        all_tasks = set(np.arange(n_tasks))
        nov_task = set([0])
        pretrain_tasks = all_tasks.difference(nov_task)
        if self.data.get(key) is None:
            out_two = self.train_modularizer(
                train_epochs=len(pretrain_tasks) * train_epochs,
                n_train=len(pretrain_tasks) * train_samps,
                tasks_per_group=n_tasks,
                only_tasks=pretrain_tasks,
            )
            h_next = out_two[0].fit(
                track_dimensionality=True,
                epochs=train_epochs,
                n_train=train_samps,
                verbose=False,
                val_only_tasks=nov_task,
            )

            out_one = self.train_modularizer(
                train_epochs=train_epochs,
                n_train=train_samps,
                tasks_per_group=n_tasks,
                only_tasks=nov_task,
            )
            self.data[key] = (h_next, out_one[1])
        pretrain_history, naive_history = self.data[key]
        xs = np.arange(train_epochs)

        key = "val_loss"
        log_y = False
        gpl.plot_trace_werr(
            xs,
            pretrain_history.history[key],
            ax=ax,
            log_y=log_y,
            label="pretrained network",
        )
        gpl.plot_trace_werr(
            xs, naive_history.history[key], ax=ax, log_y=log_y, label="naive network"
        )


class FigureIntro(ModularizerFigure):
    def __init__(self, fig_key="intro_figure", colors=colors, **kwargs):
        fsize = (7, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_schem",
            "panel_visualization",
            "panel_quantification",
            "panel_history",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        schem1_grid = self.gs[:50, :70]
        gss[self.panel_keys[0]] = self.get_axs((schem1_grid,))

        vis_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 55, 100, 0, 15, 10, 10)
        abl_grid = self.gs[55:75, 20:40]
        gss[self.panel_keys[1]] = (self.get_axs(vis_grid), self.get_axs((abl_grid,)))

        quant_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 55, 100, 60, 100, 5, 10)

        gss[self.panel_keys[2]] = self.get_axs(quant_grid)

        hist_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 50, 80, 100, 10, 0)
        gss[self.panel_keys[3]] = self.get_axs(hist_grid)

        self.gss = gss

    def panel_history(self):
        key = self.panel_keys[3]
        ax_hist, ax_dim = np.squeeze(self.gss[key])

        nets, hist_dicts, labels = self.make_modularizers()
        hist_dicts = hist_dicts[:1]
        for i, h_dict in enumerate(hist_dicts):
            hist = h_dict.history["val_loss"]
            dims = h_dict.history["dimensionality"]
            epochs = h_dict.epoch
            gpl.plot_trace_werr(epochs, hist, ax=ax_hist)
            gpl.plot_trace_werr(epochs, dims, ax=ax_dim)

        c = nets[0].n_groups
        l = nets[0].group_size
        pred_dim = c * l
        alt_dim = (c**2) * l - c

        ax_dim.plot([epochs[-1]], pred_dim, "o", ms=5, label="modular")
        ax_dim.plot([epochs[-1]], alt_dim, "o", ms=5, label="non-modular")
        ax_dim.legend(frameon=False)
        ax_dim.set_xlabel("training epoch")
        ax_dim.set_ylabel("hidden layer dim")

    def panel_visualization(self):
        key = self.panel_keys[1]
        axs, axs_abl = self.gss[key]
        axs_abl = axs_abl[0, 0]

        nets, _, labels = self.make_modularizers()

        mv.plot_model_list_activity(nets[:1], axs=axs, cmap="Blues")
        axs[1, 0].set_title("")

        ablate_layer = None
        tc_lin = ma.act_ablation(
            nets[0], n_clusters=3, single_number=False, layer=ablate_layer
        )

        tc_lin[tc_lin < 0] = 0
        m = axs_abl.imshow(tc_lin, cmap="Reds")
        self.f.colorbar(m, ax=axs_abl, label="normalized\nperformance change")
        axs_abl.set_xlabel("context")
        axs_abl.set_ylabel("inferred cluster")
        axs_abl.set_yticks([0, 1, 2])

    def panel_quantification(self):
        key = self.panel_keys[2]
        axs = self.gss[key]
        ((ax_clustering, ax_abl), (ax_ccgp, ax_shatter)) = axs

        ccgp_color_wi = self.params.getcolor("ccgp_color_wi")
        ccgp_color_ac = self.params.getcolor("ccgp_color_ac")
        shattering_color = self.params.getcolor("shattering_color")
        ablation_color = self.params.getcolor("ablation_color")

        ri_l3 = self.params.get("run_ind_l3")
        ri_l5 = self.params.get("run_ind_l5")
        labels = ("D = 3", "D = 5")
        colors = (
            self.params.getcolor("l3_color"),
            self.params.getcolor("l5_color"),
        )
        linestyles = ("solid", "solid")

        for i, run in enumerate((ri_l3, ri_l5)):
            run_data, order, args = self.load_run(run)
            print(args)

            label = labels[i]

            gm = run_data["gm"].T
            frac = run_data["model_frac"].T
            fdg_frac = run_data["fdg_frac"].T
            gpl.plot_trace_werr(
                order,
                frac,
                ax=ax_clustering,
                log_x=True,
                color=colors[i],
                ls=linestyles[i],
            )
            gpl.plot_trace_werr(
                order, fdg_frac, ax=ax_clustering, log_x=True, color=(0.9, 0.9, 0.9)
            )
            ax_clustering.set_ylabel("cluster fraction")
            ax_clustering.set_xlabel("tasks")

            waa = run_data["within_act_ablation"].T
            aaa = run_data["across_act_ablation"].T
            gpl.plot_trace_werr(
                order,
                waa - aaa,
                ax=ax_abl,
                log_x=True,
                color=colors[i],
                ls=linestyles[i],
            )
            ax_abl.set_ylabel("ablation effect")
            gpl.add_hlines(0, ax_abl)
            ax_abl.set_xlabel("N tasks")

            w_ccgp = run_data["within_ccgp"].T
            gpl.plot_trace_werr(
                order,
                w_ccgp,
                ax=ax_ccgp,
                log_x=True,
                color=colors[i],
                label="within",
                ls=linestyles[i],
            )
            ax_ccgp.set_ylabel("classifier\ngeneralization")

            # a_ccgp = run_data['across_ccgp'].T
            # gpl.plot_trace_werr(order, a_ccgp, ax=ax_ccgp, log_x=True,
            #                     color=colors[i], label='across',
            #                     ls=linestyles[i])
            gpl.add_hlines(0.5, ax_ccgp)

            shatter = run_data["shattering"].T
            gpl.plot_trace_werr(
                order,
                shatter,
                ax=ax_shatter,
                log_x=True,
                label=label,
                color=colors[i],
                ls=linestyles[i],
            )
            ax_shatter.set_ylabel("shattering\ndimensionality")
            ax_shatter.set_xlabel("tasks")
            gpl.add_hlines(0.5, ax_shatter)
            ax_ccgp.set_xlabel("tasks")


def _get_last_dim(dims):
    """
    dims is T x R x E
    where we want to find the last element on the third dimension that is not nan
    """
    mask = np.isnan(dims)
    change_mask = np.diff(mask, axis=2)
    out = dims[..., :-1][change_mask].reshape((dims.shape[:2]))
    return out


class FigureOtherCases(ModularizerFigure):
    def __init__(self, fig_key="other_param_figure", colors=colors, **kwargs):
        fsize = (7, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_nonlinear_single",
            "panel_overlap_clusters",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        lb = 40

        task_schem = self.gs[:lb, 0:100]
        self.get_axs((task_schem,))

        task_comp_grid = pu.make_mxn_gridspec(self.gs, 2, 3, lb, 100, 0, 60, 8, 8)
        gss[self.panel_keys[0]] = self.get_axs(task_comp_grid, sharey="vertical")

        cs_grid = pu.make_mxn_gridspec(self.gs, 2, 2, lb, 100, 70, 100, 5, 8)
        gss[self.panel_keys[1]] = self.get_axs(
            cs_grid, sharex="all", sharey="all", aspect="equal"
        )

        self.gss = gss

    def make_nl_modularizers(self, retrain=False):
        if self.data.get("trained_nl_models") is None or retrain:
            m_o, h_o = self.train_modularizer(model_type=ms.ColoringModularizer)
            m_no, h_no = self.train_modularizer(
                model_type=ms.ColoringModularizer, n_overlap=0
            )
            labels = ("overlap", "no overlap")
            self.data["trained_nl_models"] = (
                (
                    m_o,
                    m_no,
                ),  # m_reg),
                (
                    h_o,
                    h_no,
                ),  # h_reg),
                labels,
            )
        return self.data["trained_nl_models"]

    def panel_task_comparison(self, recompute=False):
        key = self.panel_keys[0]
        axs = self.gss[key]

        l_o_run = self.params.get("lin_overlap_ri")
        l_no_run = self.params.get("lin_no_overlap_ri")

        nl_o_run = self.params.get("color_overlap_ri")
        nl_no_run = self.params.get("color_no_overlap_ri")

        if self.data.get(key) is None or recompute:
            l_o_out = self.load_run(l_o_run)
            l_no_out = self.load_run(l_no_run)
            nl_o_out = self.load_run(nl_o_run)
            nl_no_out = self.load_run(nl_no_run)

            c = l_o_out[-1]["n_groups"]
            l = l_o_out[-1]["group_size"][0]
            o = (l, 0)

            out_dim, out_pv = ma.task_dimensionality(l, o, l_o_out[1], c, n_reps=20)

            runs = ((l_o_out, l_no_out), (nl_o_out, nl_no_out))
            self.data[key] = (runs, out_dim)

        run_groups, out_dim = self.data[key]
        group_labels = ("linear", "coloring")
        overlap_labels = ("overlap", "no overlap")
        o_color = self.params.getcolor("l3_color")
        no_color = self.params.getcolor("no_overlap_color")
        j_colors = (o_color, no_color)

        for i, group in enumerate(run_groups):
            for j, run in enumerate(group):
                run_data, order, args = run

                model_frac = run_data["model_frac"].T
                fdg_frac = run_data["fdg_frac"].T
                gpl.plot_trace_werr(
                    order,
                    model_frac,
                    ax=axs[i, 1],
                    log_x=True,
                    color=j_colors[j],
                    label=overlap_labels[j],
                )
                gpl.plot_trace_werr(
                    order, fdg_frac, ax=axs[i, 1], log_x=True, color=(0.9, 0.9, 0.9)
                )

                waa = run_data["within_act_ablation"].T
                aaa = run_data["across_act_ablation"].T
                gpl.plot_trace_werr(
                    order,
                    waa - aaa,
                    ax=axs[i, 2],
                    log_x=True,
                    color=j_colors[j],
                    label=overlap_labels[j],
                )

                dims = out_dim[group_labels[i]]
                wi_dim = np.squeeze(dims[1])
                ac_dim = np.squeeze(dims[2])
                gpl.plot_trace_werr(
                    order,
                    ac_dim[j].T - wi_dim[j].T,
                    ax=axs[i, 0],
                    log_x=True,
                    color=j_colors[j],
                )

            gpl.add_hlines(0, axs[i, 2])
            gpl.add_hlines(0, axs[i, 0])

            axs[i, 1].set_ylabel("cluster fraction")
            axs[i, 2].set_ylabel("ablation effect")
            axs[i, 0].set_ylabel("excess dimensionality")

        axs[-1, 2].set_xlabel("N tasks")
        axs[-1, 1].set_xlabel("N tasks")
        axs[-1, 0].set_xlabel("N tasks")

    def panel_nonlinear_single(self):
        key = self.panel_keys[0]
        axs_vis, axs_quant = self.gss[key]
        axs_quant = axs_quant.T

        o_run = self.params.get("overlap_ri")
        no_run = self.params.get("no_overlap_ri")
        labels = ("overlap", "no overlap")
        for i, run in enumerate((o_run, no_run)):
            run_data, order, args = self.load_run(run)
            l = args["group_size"][0]
            o = args["group_overlap"][0]
            if self.data.get(key) is None:
                self.data[key] = {}
            if self.data[key].get((l, o)) is None:
                c = args["n_groups"]
                models = {"coloring": ms.ColoringModularizer}
                out_dim, out_pv = ma.task_dimensionality(
                    l, o, order, c, n_reps=10, models=models
                )

                self.data[key][(l, o)] = out_dim["coloring"]
            out_dim = self.data[key][(l, o)]
            wi_dim = np.squeeze(out_dim[1])
            ac_dim = np.squeeze(out_dim[2])
            gpl.plot_trace_werr(
                order, ac_dim.T - wi_dim.T, ax=axs_quant[0, 0], log_x=True
            )

            tot_dim = np.squeeze(out_dim[0])
            dims = _get_last_dim(run_data["dimensionality"])
            gpl.plot_trace_werr(order, dims.T, ax=axs_vis[0, 0], ls="dashed")
            gpl.plot_trace_werr(order, tot_dim.T, ax=axs_vis[0, 0])
            print(tot_dim.shape, dims.shape)

            waa = run_data["within_act_ablation"].T
            aaa = run_data["across_act_ablation"].T
            gpl.plot_trace_werr(order, waa - aaa, ax=axs_quant[0, 1], log_x=True)
            gpl.add_hlines(0, axs_quant[0, 1])
            axs_quant[0, 1].set_xlabel("N tasks")
            axs_quant[0, 1].set_ylabel("ablation effect")
            axs_quant[0, 0].set_ylabel("excess dimensionality")

    def panel_overlap_clusters(self):
        key = self.panel_keys[1]
        axs_scatters = self.gss[key]
        axs_scatters = axs_scatters.flatten()

        model_types = ("linear", "linear", "coloring", "coloring")
        overlaps = (3, 0, 3, 0)

        if self.data.get(key) is None:
            mod_dict = {}
            for i, mt in enumerate(model_types):
                ov = overlaps[i]

                out = self.train_modularizer(n_overlap=ov, model_type_str=mt)

                mod_dict[(mt, ov)] = out
            self.data[key] = mod_dict

        mod_dict = self.data[key]
        c1_color = self.params.getcolor("con1_color")
        c2_color = self.params.getcolor("con2_color")
        neutral_color = self.params.getcolor("noncon_color")
        colors = (c1_color, neutral_color, c2_color)
        for i, ((mt, ov), (mod, hist)) in enumerate(mod_dict.items()):
            ax_s = axs_scatters[i]

            mv.plot_context_scatter(mod, ax=ax_s, colors=colors)
            ax_s.set_xlabel("C1 activity")
            ax_s.set_ylabel("C2 activity")


class FigureHiddenLayers(ModularizerFigure):
    def __init__(self, fig_key="hidden_layers_figure", colors=colors, **kwargs):
        fsize = (7, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        lb = 40

        task_schem = self.gs[:lb, 0:100]
        self.get_axs((task_schem,))

        n_layers = len(self.params.getlist("hidden_layer_inds"))
        layer_eg_grid = pu.make_mxn_gridspec(self.gs, 2, n_layers, lb, 100, 0, 60, 8, 8)
        gss["panel_eg"] = self.get_axs(
            layer_eg_grid, sharey="horizontal", sharex="horizontal"
        )

        quant_grid = pu.make_mxn_gridspec(self.gs, 3, 1, 20, 100, 70, 100, 8, 8)

        gss["panel_quant"] = self.get_axs(quant_grid, sharex="all")

        self.gss = gss

    def make_hidden_modularizers(self, retrain=False):
        if self.data.get("trained_hidden_models") is None or retrain:
            hiddens = self.params.getlist("test_hiddens", typefunc=int)

            m_hid, h_hid = self.train_modularizer(additional_hidden=hiddens)
            labels = ("hidden layers",)
            self.data["trained_hidden_models"] = ((m_hid,), (h_hid,), labels)
        return self.data["trained_hidden_models"]

    def panel_eg(self):
        key = "panel_eg"
        axs_hid = self.gss[key]

        inds = self.params.getlist("hidden_layer_inds", typefunc=int)

        nets, _, labels = self.make_hidden_modularizers()
        for i, ind in enumerate(inds):
            mv.plot_model_list_activity(nets, axs=axs_hid[:, i : i + 1], from_layer=ind)

    def panel_quant(self):
        key = "panel_quant"
        axs_all = self.gss[key]

        ri_list = self.params.getlist("ri_list")
        quant_keys = ("model_frac", "diff_act_ablation", "within_ccgp", "shattering")
        ri_list = self.params.getlist("ri_list")

        label_dict = {(3,): "D = 3", (5,): "D = 5", (8,): "D = 8"}
        nulls = (0, 0, 0.5, 0.5)
        self._quantification_panel(
            quant_keys, ri_list, axs_all, label_dict=label_dict, nulls=nulls
        )


class FigureImageModularity(ModularizerFigure):
    def __init__(self, fig_key="images", colors=colors, **kwargs):
        fsize = (7, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        ri_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 60, 100, 80, 100, 10, 10)
        ri_axs = self.get_axs(ri_grid, sharex="all", squeeze=True)
        gss["panel_ri"] = ri_axs

        n_plots = len(self.params.getlist("n_eg_tasks"))
        tc_grid = pu.make_mxn_gridspec(self.gs, 3, n_plots, 50, 100, 20, 70, 5, 5)
        tc_axs = self.get_axs(tc_grid, sharey="horizontal", sharex="horizontal")
        gss["panel_task_compare"] = tc_axs

        self.gss = gss

    def panel_ri(self):
        key = "panel_ri"
        axs_all = self.gss[key]
        quant_keys = ("model_frac", "diff_act_ablation")
        ri_list = self.params.getlist("ri_list")

        label_dict = {(3,): "2D shapes", (5,): "D = 5", (8,): "D = 8"}
        nulls = (0, 0, 0.5, 0.5)
        self._quantification_panel(
            quant_keys, ri_list, axs_all, label_dict=label_dict, nulls=nulls
        )

    def make_fdg(self, use_cache=True):
        fdg = self.data.get("trained_fdg")
        if fdg is None:
            fdg = ms.load_twod_dg(use_cache=use_cache)
        return fdg

    def panel_task_compare(self, refit_models=False, recompute_ablation=False):
        key = "panel_task_compare"
        axs_all = self.gss[key]

        # maybe also add high-dim visualization?
        n_tasks = self.params.getlist("n_eg_tasks", typefunc=int)
        act_cmap = self.params.get("activity_cmap")
        ablation_cmap = self.params.get("ablation_cmap")

        if self.data.get(key) is None or refit_models:
            models = []
            hists = []
            abls = []
            fdg = self.make_fdg()
            n_groups = fdg.n_cats
            for i, nt in enumerate(n_tasks):
                m_i, h_i = self.train_modularizer(tasks_per_group=nt, n_groups=n_groups)
                tc_i = ma.act_ablation(
                    m_i,
                    single_number=False,
                )

                models.append(m_i)
                hists.append(h_i)
                abls.append(tc_i)
            self.data[key] = (fdg, models, hists, abls)

        fdg, models, hists, abls = self.data[key]
        for i, nt in enumerate(n_tasks):
            m_i = models[i]
            tc_i = abls[i]
            if recompute_ablation:
                tc_i = ma.act_ablation(
                    m_i,
                    single_number=False,
                )

            axs_i = axs_all[:, i]
            mv.plot_context_clusters(m_i, ax=axs_i[0], cmap=act_cmap)
            mv.plot_context_scatter(m_i, ax=axs_i[1])

            tc_i[tc_i < 0] = 0
            m = axs_i[2].imshow(tc_i, cmap=ablation_cmap)
            self.f.colorbar(m, ax=axs_i[2], label="normalized\nperformance change")
            axs_i[2].set_xlabel("context")
            axs_i[2].set_ylabel("inferred cluster")
            axs_i[2].set_yticks([0, 1, 2])
