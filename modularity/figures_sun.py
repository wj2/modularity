import functools as ft
import pickle

import general.paper_utilities as pu
import general.plotting as gpl
import general.utility as u
import matplotlib.pyplot as plt
import numpy as np

import modularity.analysis as ma
import modularity.learning_analysis as mla

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
        self.fluor_cache = None
        super().__init__(*args, find_panel_keys=False, **kwargs)

    def get_fluor_maps(self):
        if self.data.get("fluor") is None and self.fluor_cache is None:
            self.data["fluor"] = mla.make_all_fluor_maps()
        elif self.fluor_cache is not None:
            with open(self.fluor_cache, "rb") as f:
                self.data["fluor"] = pickle.load(f)
        return self.data["fluor"]


class FigureSunCondensed(ModularizerFigure):
    def __init__(self, fwid=2.5, fig_key="sun_figure", colors=colors, **kwargs):
        fsize = (3 * fwid, fwid)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_networks",
            "panel_tradeoff_modularity",
            "panel_tuning",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        geom_ax = self.get_axs((self.gs[:50, :30],), squeeze=False, all_3d=True)[0, 0]
        decomp_ax = self.get_axs((self.gs[:40, 40:50],), squeeze=False)[0, 0]
        t_ax = self.get_axs((self.gs[:40, 57:60],), squeeze=False)[0, 0]
        gss["panel_geometry_eg"] = (geom_ax, decomp_ax, t_ax)

        eg_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 50, 100, 0, 60, 5, 5)
        gss["panel_tuning"] = self.get_axs(
            eg_grid, sharey="all", sharex="all", squeeze=True
        )

        tm_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 0, 100, 70, 100, 2, 5)
        tm_axs = self.get_axs(tm_grid, sharey="all", squeeze=True)
        gss["panel_tradeoff_modularity"] = tm_axs[-1]
        gss["panel_networks"] = tm_axs[0]

        self.gss = gss

    def panel_networks(self, retrain=False):
        key = "panel_networks"
        ax = self.gss[key]

        if self.data.get(key) is None or retrain:
            n_reps = self.params.getint("n_reps")
            n_ts = self.params.getint("n_tradeoffs")
            ts = np.linspace(0, 1, n_ts)
            specs = np.zeros((n_ts, n_reps))
            info = []
            for i, t in enumerate(ts):
                out = ma.train_n_activity_models(
                    t,
                    input_dim=1,
                    group_size=(1,),
                    n_reps=n_reps,
                    flip_tasks=True,
                    model_epochs=10,
                )
                specs[i] = [ma.compute_frac_contextual(m) for m in out[0]]
                info.append(out)
            self.data[key] = ts, specs, info
        ts, specs, _ = self.data[key]
        gpl.plot_trace_werr(specs.T, 1 - ts, ax=ax, confstd=True)
        ax.set_xlim([-0.05, 1])
        ax.set_ylim([-0.05, 1])
        gpl.make_yaxis_scale_bar(ax, 1, double=False, anchor=0)
        gpl.make_xaxis_scale_bar(ax, 0.2, double=False, anchor=0)

    def video_geometry(self, fwid=2):
        eg_animal = self.params.get("eg_animal")
        xs, (f_N, _), (f_F, _) = self.get_fluor_maps()[eg_animal]
        feat_color = self.params.getcolor("lv_talk")
        ctx_color = self.params.getcolor("ctx_talk")
        r1_color = self.params.getcolor("r1_color")
        r2_color = self.params.getcolor("r2_color")
        n_trls = self.params.getint("n_trls")

        f, ax = plt.subplots(
            1, 1, figsize=(fwid, fwid), subplot_kw={"projection": "3d"}
        )
        mla.plot_trial_points(
            xs,
            f_N,
            f_F,
            ax=ax,
            f_color=feat_color,
            ctx_color=ctx_color,
            r1_color=r1_color,
            r2_color=r2_color,
            n_trls=n_trls,
            ms=1,
            plot_avg=False,
        )
        dpi = 500
        ax.view_init(elev=10, azim=45)
        f.savefig(
            f"figs/geom_{eg_animal}_no-avg.png",
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
        )

        f, ax = plt.subplots(
            1, 1, figsize=(fwid, fwid), subplot_kw={"projection": "3d"}
        )
        mla.plot_trial_points(
            xs,
            f_N,
            f_F,
            ax=ax,
            f_color=feat_color,
            ctx_color=ctx_color,
            r1_color=r1_color,
            r2_color=r2_color,
            n_trls=n_trls,
            ms=1,
            ms_avg_mult=5,
        )

        gpl.rotate_3d_plot(
            f, ax, f"figs/geom_{eg_animal}.mp4", dpi=dpi, n_frames=300, format="png"
        )

    def panel_geometry_eg(self, recompute=False):
        key = "panel_geometry_eg"
        ax_geom, ax_dists, ax_t = self.gss[key]
        n_trls = self.params.getint("n_trls")
        eg_animal = self.params.get("eg_animal")
        if self.data.get(key) is None or recompute:
            xs, (f_N, _), (f_F, _) = self.get_fluor_maps()[eg_animal]
            dists, ps = mla.get_geometry_tradeoff(xs, f_N, f_F, n_trls=n_trls)
            out = ((dists, ps), (xs, f_N, f_F))
            self.data[key] = out
        (dists, ps), activity = self.data[key]

        feat_color = self.params.getcolor("lv_talk")
        ctx_color = self.params.getcolor("ctx_talk")
        r1_color = self.params.getcolor("r1_color")
        r2_color = self.params.getcolor("r2_color")
        mla.plot_trial_points(
            *activity,
            ax=ax_geom,
            f_color=feat_color,
            ctx_color=ctx_color,
            r1_color=r1_color,
            r2_color=r2_color,
            n_trls=n_trls,
        )

        use_dists = dists[0]
        gpl.plot_trace_werr(
            [0, 1, 2], use_dists.T, ax=ax_dists, fill=False, points=True, no_lines=True
        )
        cmap = plt.get_cmap("magma")
        gpl.plot_trace_werr(
            [0], ps[0][-1][:, None], ax=ax_t, points=True, fill=False, color=cmap(0.3)
        )
        ax_t.set_ylim([0, 1])
        gpl.make_yaxis_scale_bar(ax_t, 1, double=False, anchor=0)
        gpl.clean_plot_bottom(ax_t)

    def panel_tradeoff_modularity(self, recompute=False):
        key = "panel_tradeoff_modularity"
        ax = self.gss[key]

        n_trls = self.params.getint("n_trls")
        thr = self.params.getfloat("thr")
        if self.data.get(key) is None or recompute:
            fluor = self.get_fluor_maps()
            out = {}
            for animal, (xs, (f_N, _), (f_F, _)) in fluor.items():
                distances = mla.get_geometry_tradeoff(xs, f_N, f_F, n_trls=n_trls)[1]
                modularity = mla.get_specialization_frac(
                    xs, f_N, f_F, thr=thr, n_trls=n_trls
                )
                out[animal] = distances, modularity
            self.data[key] = out
        data = self.data[key]

        cmap = plt.get_cmap("magma")
        eg_animal = self.params.get("eg_animal")
        ms = 2
        for i, (animal, (dists, modularity)) in enumerate(data.items()):
            color = cmap(i / len(data))
            ts = dists[0][-1]
            spec = modularity[1] - modularity[0]
            if animal == eg_animal:
                gpl.plot_trace_werr(
                    np.mean(spec[:, None], axis=0),
                    np.mean(ts[:, None], axis=0),
                    points=True,
                    fill=False,
                    ax=ax,
                    color="g",
                    s=ms * 8,
                    no_lines=True,
                )

            gpl.plot_trace_werr(
                spec[:, None],
                ts[:, None],
                points=True,
                fill=False,
                ax=ax,
                color=color,
                err_x=gpl.std(spec[:, None], axis=0),
                no_lines=True,
            )

        ax.set_xlim([0, 0.5])
        ax.set_ylim([-0.05, 1])
        gpl.make_yaxis_scale_bar(ax, 1, double=False, anchor=0)
        gpl.make_xaxis_scale_bar(ax, 0.2, double=False, anchor=0)

    def video_tuning(self, fwid=2):
        key = "panel_tuning"
        if self.data.get(key) is None:
            self.panel_tuning()
        tuning = self.data[key]
        n_steps = 400

        def func(i, ax=None, ind=0, c1=None, c2=None):
            c1_ind = int(np.round((len(c1) - 1) * i))
            c2_ind = int(np.round((len(c2) - 1) * i))
            ax.scatter(
                c1[c1_ind],
                c2[c2_ind],
                s=1,
                lw=0,
                color=self.params.getcolor("noncon_color"),
            )
            gpl.clean_plot(ax)
            ax.set_aspect("equal")
            ax.set_xlabel("avg activity in \\textbf{ctx 1}", usetex=True)
            ax.set_ylabel("avg activity in \\textbf{ctx 2}", usetex=True)

        eg_animal = self.params.get("eg_animal")
        tuning = {eg_animal: tuning[eg_animal]}
        for animal, (c1, c2) in tuning.items():
            f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
            func_a = ft.partial(func, c1=c1, c2=c2)

            gpl.animate_plot(
                f,
                ax,
                f"figs/tuning-video_{animal}.mp4",
                np.linspace(0, 1, n_steps),
                func_a,
                dpi=500,
                format="png",
            )

    def panel_tuning(self, recompute=False, all_animals=False):
        key = "panel_tuning"
        axs = self.gss[key]

        n_trls = self.params.getint("n_trls")
        eg_animal = self.params.get("eg_animal")

        if self.data.get(key) is None or recompute:
            fluor = self.get_fluor_maps()
            tuning = {}
            for animal, (xs, (f_N, _), (f_F, _)) in fluor.items():
                out = mla.get_area_activity(xs, f_N, f_F, n_trls=n_trls)
                f1_c1, f1_c2 = out["r1"]
                f2_c1, f2_c2 = out["r2"]

                f_c1 = np.max(np.stack((f1_c1, f2_c1), axis=0), axis=0)
                f_c2 = np.max(np.stack((f1_c2, f2_c2), axis=0), axis=0)
                tuning[animal] = (f_c1, f_c2)
            self.data[key] = tuning
        tuning = self.data[key]
        tuning = {eg_animal: tuning[eg_animal]}

        begin = 0
        end = -1
        for i, (animal, (c1, c2)) in enumerate(tuning.items()):
            axs[0].scatter(c1[begin], c2[begin], s=1)
            axs[1].scatter(c1[end], c2[end], s=1)
            axs[0].set_aspect("equal")
            axs[1].set_aspect("equal")
            gpl.clean_plot(axs[0], 0)
            gpl.clean_plot(axs[1], 0)


class FigureSunData(ModularizerFigure):
    def __init__(self, fwid=2.5, fig_key="sun_figure", colors=colors, **kwargs):
        fsize = (3 * fwid, fwid * len(mla.ANIMALS))
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            "panel_decoding",
            "panel_distance",
            "panel_tuning",
        )
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        h_gap = 2
        v_gap = 2
        n_rows = len(mla.ANIMALS)
        dec_grid = pu.make_mxn_gridspec(self.gs, n_rows, 3, 0, 100, 0, 55, v_gap, h_gap)
        dec_axs = self.get_axs(dec_grid, squeeze=True, sharey="all")
        gss["panel_decoding"] = dec_axs

        geom_grid = pu.make_mxn_gridspec(
            self.gs, n_rows, 1, 0, 100, 60, 70, v_gap, h_gap
        )
        gss["panel_distance"] = self.get_axs(geom_grid, squeeze=True, sharey="all")

        tune_grid = pu.make_mxn_gridspec(
            self.gs, n_rows, 2, 0, 100, 75, 100, v_gap, h_gap
        )
        gss["panel_tuning"] = self.get_axs(
            tune_grid, squeeze=False, sharey="horizontal", sharex="horizontal"
        )

        self.gss = gss

    def get_fluor_maps(self):
        if self.data.get("fluor") is None:
            self.data["fluor"] = mla.make_all_fluor_maps()
        return self.data["fluor"]

    def panel_decoding(self):
        key = "panel_decoding"
        axs = self.gss[key]

        n_trls = self.params.getint("n_trls")
        if self.data.get(key) is None:
            fluor = self.get_fluor_maps()
            dec_out = mla.get_all_geometry_analysis(fluor, n_trls=n_trls)
            self.data[key] = dec_out
        dec_out = self.data[key]
        for i, (animal, decs) in enumerate(dec_out.items()):
            if i == 0:
                titles = ("action", "context", "zone")
            else:
                titles = ("",) * 3
            mla.plot_all_dec(
                *decs,
                chance_level=0,
                one_row=True,
                axs=axs[i : i + 1],
                titles=titles,
            )

    def panel_distance(self, recompute=False):
        key = "panel_distance"
        axs = self.gss[key]

        n_trls = self.params.getint("n_trls")
        if self.data.get(key) is None or recompute:
            fluor = self.get_fluor_maps()
            out = {}
            for animal, (xs, (f_N, _), (f_F, _)) in fluor.items():
                distances = mla.get_geometry_tradeoff(xs, f_N, f_F, n_trls=n_trls)[1]
                out[animal] = distances
            self.data[key] = out
        distances = self.data[key]

        for i, (animal, ((_, _, early), (_, _, late))) in enumerate(distances.items()):
            dists = np.stack((early, late), axis=1)
            gpl.plot_trace_werr([0, 1], dists, fill=False, points=True, ax=axs[i])
            gpl.clean_plot(axs[i], 0)
            gpl.clean_plot_bottom(axs[i])

    def video_tuning(self, fwid=2):
        key = "panel_tuning"
        if self.data.get(key) is None:
            self.panel_tuning()
        tuning = self.data[key]
        n_steps = 400

        def func(i, ax=None, ind=0, c1=None, c2=None):
            c1_ind = int(np.round((len(c1) - 1) * i))
            c2_ind = int(np.round((len(c2) - 1) * i))
            ax.scatter(
                c1[c1_ind],
                c2[c2_ind],
                s=0.3,
                lw=0,
                color=self.params.getcolor("noncon_color"),
            )
            gpl.clean_plot(ax)
            ax.set_aspect("equal")
            ax.set_xlabel("ctx 1")
            ax.set_ylabel("ctx 2")

        tuning = {"A5": tuning["A5"]}
        for animal, (c1, c2) in tuning.items():
            f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
            func_a = ft.partial(func, c1=c1, c2=c2)

            gpl.animate_plot(
                f,
                ax,
                f"figs/tuning-video_{animal}.mp4",
                np.linspace(0, 1, n_steps),
                func_a,
                dpi=500,
            )

    def panel_tuning(self, recompute=False):
        key = "panel_tuning"
        axs = self.gss[key]

        n_trls = self.params.getint("n_trls")

        if self.data.get(key) is None or recompute:
            fluor = self.get_fluor_maps()
            tuning = {}
            for animal, (xs, (f_N, _), (f_F, _)) in fluor.items():
                out = mla.get_area_activity(xs, f_N, f_F, n_trls=n_trls)
                f1_c1, f1_c2 = out["r1"]
                f2_c1, f2_c2 = out["r2"]

                f_c1 = np.max(np.stack((f1_c1, f2_c1), axis=0), axis=0)
                f_c2 = np.max(np.stack((f1_c2, f2_c2), axis=0), axis=0)
                tuning[animal] = (f_c1, f_c2)
            self.data[key] = tuning
        tuning = self.data[key]

        begin = 0
        end = -1
        for i, (animal, (c1, c2)) in enumerate(tuning.items()):
            axs[i, 0].scatter(c1[begin], c2[begin], s=1)
            axs[i, 1].scatter(c1[end], c2[end], s=1)
            axs[i, 0].set_aspect("equal")
            axs[i, 1].set_aspect("equal")
            gpl.clean_plot(axs[i, 0], 0)
            gpl.clean_plot(axs[i, 1], 0)
