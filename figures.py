import itertools as it
import numpy as np
import matplotlib.pyplot as plt

import modularity.simple as ms
import modularity.analysis as ma
import modularity.visualization as mv
import modularity.auxiliary as maux
import disentangled.aux as da
import disentangled.disentanglers as dd
import disentangled.regularizer as dr
import disentangled.data_generation as dg

import general.utility as u
import general.plotting_styles as gps
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

            dg_epochs = self.params.getint("dg_epochs")
            dg_noise = self.params.getfloat("dg_noise")
            dg_regweight = self.params.getlist("dg_regweight", typefunc=float)
            dg_layers = self.params.get("dg_layers")
            dg_layers = self.params.getlist("dg_layers", typefunc=int)
            dg_train_egs = self.params.getint("dg_train_egs")
            dg_pr_reg = self.params.getboolean("dg_pr_reg")
            dg_bs = self.params.getint("dg_batch_size")

            source_distr = u.MultiBernoulli(0.5, inp_dim)
            fdg = dg.FunctionalDataGenerator(
                inp_dim,
                dg_layers,
                dg_dim,
                noise=dg_noise,
                use_pr_reg=dg_pr_reg,
                l2_weight=dg_regweight,
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

    def make_ident_modularizer(self, linear=False, **kwargs):
        if linear:
            m_type = ms.LinearIdentityModularizer
        else:
            m_type = ms.IdentityModularizer
        m_ident, h = self.train_modularizer(model_type=m_type, train_epochs=0, **kwargs)
        return m_ident

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

    def train_modularizer(self, verbose=False, **kwargs):
        fdg = self.make_fdg()
        return ms.train_modularizer(fdg, verbose=verbose, params=self.params, **kwargs)

    def load_run(self, run_ind):
        folder = self.params.get("sim_folder")
        out = maux.load_run(run_ind, folder=folder)
        return out

    
class FigureInput(ModularizerFigure):
    def __init__(self, fig_key="input_figure", colors=colors, **kwargs):
        fsize = (1.4, 4)
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

        dim_sparse_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 0, 20, 0, 100, 10, 30)
        gss["panel_dim_sparse"] = self.get_axs(
            dim_sparse_grid, squeeze=True, sharex="row"
        )

        act_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 40, 100, 10, 90, 10, 10)
        gss["panel_act"] = self.get_axs(act_grid, squeeze=True)

        tasks_grid = self.gs[85:100, 70:]
        gss["panel_tasks"] = self.get_axs((tasks_grid,), squeeze=False)[0, 0]

        self.gss = gss

    def panel_dim_sparse(self):
        key = "panel_dim_sparse"
        ax_sparse, ax_dims = self.gss[key]

        fdg = self.make_fdg()
        lv_dim = fdg.input_dim
        lvs, reps = fdg.sample_reps(10000)

        lv_sparse = u.quantify_sparseness(lvs)

        rep_sparse = u.quantify_sparseness(reps)
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

        modu = self.make_ident_modularizer()
        mv.plot_context_clusters(modu, ax=ax_clust, cmap=cmap)
        mv.plot_context_scatter(modu, ax=ax_scatt)
        ax_scatt.set_xlabel("activity in context 1")
        ax_scatt.set_ylabel("activity in context 2")
        ax_clust.set_xlabel("units")
        ax_clust.set_ylabel("")


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
                print(svs.shape)
                thr = 0.1
                print(hist[-5:, :])
                svs = svs[hist[-1, :] < thr]
                print(svs.shape)
                model_out = ms.make_linear_network(
                    stims, targs, use_relu=True, verbose=False
                )
                ws_all = model_out[0].weights[0].numpy().T
                ws = u.make_unit_vector(ws_all)
                stable_gates[nt] = (stims, targs, svs, ws, model_out)
            self.data[key] = stable_gates

        stable_key = self.data[key]
        for i, nt in enumerate(nts):
            stims, targs, gates, ws, model_out = stable_gates[nt]
            resp = model_out[1](stims)
            mask_c1 = stims[:, -1] == 1

            print(gates)
            print(ws[:10])
            resp_c1 = np.mean(resp[mask_c1], axis=0)
            resp_c2 = np.mean(resp[~mask_c1], axis=0)
            # axs_vis[i].plot(resp_c1, resp_c2, 'o')
            mv.visualize_stable_gates(stims, targs, gates, ws=ws, ax=axs_vis[i])
            axs_vis[i].view_init(0, 0)
            mv.visualize_gate_angle(stims, targs, gates, ws=ws, ax=axs_dirs[i])


class FigureDiscreteModularity(ModularizerFigure):
    def __init__(self, fig_key='modularity_discrete', colors=colors, **kwargs):
        fsize = (7, 5)
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

        ri_grid = pu.make_mxn_gridspec(self.gs, 4, 1,
                                       20, 100,
                                       80, 100,
                                       10, 10)
        ri_axs = self.get_axs(ri_grid, sharex="all", squeeze=True)
        gss['panel_ri'] = ri_axs

        n_plots = len(self.params.getlist('n_eg_tasks'))
        tc_grid = pu.make_mxn_gridspec(self.gs, 3, n_plots,
                                       50, 100,
                                       20, 70,
                                       5, 5)
        tc_axs = self.get_axs(tc_grid,
                              sharey="horizontal",
                              sharex="horizontal")
        gss['panel_task_compare'] = tc_axs

        self.gss = gss

    def panel_ri(self):
        key = 'panel_ri'
        axs_all = self.gss[key]
        quant_keys = ('model_frac', 'diff_act_ablation',
                      'within_ccgp', 'shattering')
        ri_list = self.params.getlist('ri_list')
        model_templ = self.params.get('model_template')
        label_dict = {(3,): 'D = 3', (5,): 'D = 5', (8,): 'D = 8'}
        nulls = (0, 0, .5, .5)

        for i, qk in enumerate(quant_keys):
            qk_ri = mv.accumulate_run_quants(
                ri_list,
                templ=model_templ,
                quant_key=qk,
                legend_keys=('group_size',),
            )
            for k, (xs, qs) in qk_ri.items():
                gpl.plot_trace_werr(xs, qs.T, ax=axs_all[i],
                                    label=label_dict[k], log_x=True)
            gpl.add_hlines(nulls[i], axs_all[i])

    def panel_task_compare(self, refit_models=False, recompute_ablation=False):
        key = 'panel_task_compare'
        axs_all = self.gss[key]

        # maybe also add high-dim visualization?
        n_tasks = self.params.getlist('n_eg_tasks', typefunc=int)
        act_cmap = self.params.get('activity_cmap')
        ablation_cmap = self.params.get('ablation_cmap')

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
        for i, nt in enumerate(n_tasks):
            m_i = models[i]
            tc_i = abls[i]
            if recompute_ablation:
                tc_i = ma.act_ablation(
                    m_i, single_number=False,
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


class FigureGeometryConsequences(ModularizerFigure):
    def __init__(self, fig_key="geometry_consequences", colors=colors, **kwargs):
        fsize = (7, 5)
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

        nt_grid = pu.make_mxn_gridspec(self.gs, 1, 2,
                                       70, 100,
                                       0, 100,
                                       10, 10)
        nt_axs = self.get_axs(nt_grid,)
        gss['panel_new_task'] = nt_axs[0, 0]
        gss['panel_new_context'] = nt_axs[0, 1]
        
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
        )
        labels = ("D = 3", "D = 5")
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
            axs[2, 1].set_ylabel("CCGP")
            axs[2, 1].set_xlabel("N tasks")
            gpl.add_hlines(0.5, axs[2, 1])

            shatter = run_data["shattering"].T
            gpl.plot_trace_werr(
                order, shatter, ax=axs[2, 2], log_x=True, color=colors[i]
            )
            axs[2, 2].set_ylabel("shattering\ndimensionality")
            axs[2, 2].set_xlabel("N tasks")
            gpl.add_hlines(0.5, axs[2, 2])

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
        for i in u_clusts:
            xs, ys = [], []
            mask = clusts == i
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
        key = 'panel_new_context'
        ax = self.gss[key]

        train_epochs = self.params.getint('context_train_epochs')
        train_samps = self.params.getint('context_train_samples')
        n_tasks = self.params.getint('n_tasks')
        if self.data.get(key) is None:
            out_two = self.train_modularizer(n_groups=3,
                                             only_groups=(0, 1),
                                             train_epochs=2*train_epochs,
                                             n_train=2*train_samps,
                                             tasks_per_group=n_tasks)
            h_next = out_two[0].fit(track_dimensionality=True,
                                    epochs=train_epochs,
                                    n_train=train_samps,
                                    verbose=False,
                                    val_only_groups=(2,))

            out_one = self.train_modularizer(n_groups=3,
                                             only_groups=(0,),
                                             train_epochs=train_epochs,
                                             n_train=train_samps,
                                             tasks_per_group=n_tasks)
            self.data[key] = (h_next, out_one[1])
        pretrain_history, naive_history = self.data[key]
        xs = np.arange(train_epochs)

        key = 'val_loss'
        gpl.plot_trace_werr(xs, pretrain_history.history[key], ax=ax, log_y=True,
                            label='pretrained network')
        gpl.plot_trace_werr(xs, naive_history.history[key], ax=ax, log_y=True,
                            label='naive network')

    def panel_new_task(self):
        key = 'panel_new_task'
        ax = self.gss[key]

        train_epochs = self.params.getint('task_train_epochs')
        train_samps = self.params.getint('task_train_samples')
        n_tasks = self.params.getint('n_tasks')

        all_tasks = set(np.arange(n_tasks))
        nov_task = set([0])
        pretrain_tasks = all_tasks.difference(nov_task)
        if self.data.get(key) is None:
            out_two = self.train_modularizer(
                train_epochs=len(pretrain_tasks)*train_epochs,
                n_train=len(pretrain_tasks)*train_samps,
                tasks_per_group=n_tasks,
                only_tasks=pretrain_tasks,
            )
            h_next = out_two[0].fit(track_dimensionality=True,
                                    epochs=train_epochs,
                                    n_train=train_samps,
                                    verbose=False,
                                    val_only_tasks=nov_task)

            out_one = self.train_modularizer(train_epochs=train_epochs,
                                             n_train=train_samps,
                                             tasks_per_group=n_tasks,
                                             only_tasks=nov_task)
            self.data[key] = (h_next, out_one[1])
        pretrain_history, naive_history = self.data[key]
        xs = np.arange(train_epochs)

        key = 'val_loss'
        gpl.plot_trace_werr(xs, pretrain_history.history[key], ax=ax, log_y=True,
                            label='pretrained network')
        gpl.plot_trace_werr(xs, naive_history.history[key], ax=ax, log_y=True,
                            label='naive network')


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
            ax_clustering.set_xlabel("N tasks")

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
            ax_ccgp.set_ylabel("CCGP")

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
            ax_shatter.set_xlabel("N tasks")
            gpl.add_hlines(0.5, ax_shatter)
            ax_ccgp.set_xlabel("N tasks")


def _get_last_dim(dims):
    """
    dims is T x R x E
    where we want to find the last element on the third dimension that is not nan
    """
    mask = np.isnan(dims)
    change_mask = np.diff(mask, axis=2)
    out = dims[..., :-1][change_mask].reshape((dims.shape[:2]))
    return out


class FigureExplanation(ModularizerFigure):
    def __init__(self, fig_key="explanation_figure", colors=colors, **kwargs):
        fsize = (7, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_nonlinear_single",
            "panel_hidden_layers",
            "panel_overlap_clusters",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        lb = 50

        task_schem = self.gs[:20, 0:50]
        self.get_axs((task_schem,))

        # nl_single_grid = pu.make_mxn_gridspec(self.gs, 2, 1,
        #                                       15, lb, 0, 30,
        #                                       10, 10)
        # nl_single_abl_grid = pu.make_mxn_gridspec(self.gs, 2, 1,
        #                                           15, lb, 35, 55,
        #                                           10, 10)
        # gss[self.panel_keys[0]] = (self.get_axs(nl_single_grid),
        #                            self.get_axs(nl_single_abl_grid))

        task_comp_grid = pu.make_mxn_gridspec(self.gs, 2, 3, 15, lb, 0, 55, 5, 8)

        gss[self.panel_keys[0]] = self.get_axs(task_comp_grid, sharey="vertical")

        hidden_grid = pu.make_mxn_gridspec(self.gs, 2, 3, 15, lb, 60, 100, 5, 10)

        gss[self.panel_keys[1]] = self.get_axs(hidden_grid)

        cs_grid = pu.make_mxn_gridspec(self.gs, 2, 4, lb + 10, 100, 0, 100, 5, 10)
        gss[self.panel_keys[2]] = self.get_axs(cs_grid).T

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

    def make_hidden_modularizers(self, retrain=False):
        if self.data.get("trained_hidden_models") is None or retrain:
            hiddens = self.params.getlist("test_hiddens", typefunc=int)

            m_hid, h_hid = self.train_modularizer(additional_hidden=hiddens)
            labels = ("hidden layers",)
            self.data["trained_hidden_models"] = ((m_hid,), (h_hid,), labels)
        return self.data["trained_hidden_models"]

    def panel_hidden_layers(self):
        key = self.panel_keys[1]
        axs_hid = self.gss[key]

        inds = self.params.getlist("hidden_layer_inds", typefunc=int)

        nets, _, labels = self.make_hidden_modularizers()
        for i, ind in enumerate(inds):
            mv.plot_model_list_activity(nets, axs=axs_hid[:, i : i + 1], from_layer=ind)

    def panel_overlap_clusters(self):
        key = self.panel_keys[2]
        axs_clusters = self.gss[key]

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
        for i, ((mt, ov), (mod, hist)) in enumerate(mod_dict.items()):
            ax_cluster, ax_scatter = axs_clusters[i]

            mv.plot_context_clusters(mod, ax=ax_cluster)
            mv.plot_context_scatter(mod, ax=ax_scatter)
            ax_scatter.set_xlabel("C1 activity")
            ax_scatter.set_ylabel("C2 activity")
