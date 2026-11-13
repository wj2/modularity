import itertools as it
import os

import general.neural_analysis as na
import general.plotting as gpl
import general.utility as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rsatoolbox as rsa
import scipy.optimize as sopt
import scipy.signal as sig
from scipy.stats import rankdata

import vr2p
import vr2p.signal

CUE_AREA = (60, 100)
R1_AREA = (130, 150)
R2_AREA = (180, 200)
POST_CUE = (100, 500)

DATAPATH = "/Volumes/dt-ext/data-ext/context_learning/set-a/Set A/"
PATTERN = os.path.join(DATAPATH, "Tyche-{}-SetA.zarr")
ANIMALS = ("A2", "A4", "A5", "A7", "B2", "B3", "B4", "B5", "C1", "C3", "D1", "F1", "F5")


def get_specialization_frac(xs, f_N, f_F, begin=0, end=-1, n_trls=100, thr=0.1):
    out = get_area_activity(xs, f_N, f_F, n_trls=n_trls)
    f1_c1, f1_c2 = out["r1"]
    f2_c1, f2_c2 = out["r2"]

    f_c1 = np.max(np.stack((f1_c1, f2_c1), axis=0), axis=0)
    f_c2 = np.max(np.stack((f1_c2, f2_c2), axis=0), axis=0)
    early_specialization = np.logical_or(
        np.logical_and(f_c1[begin] < thr, f_c2[begin] > thr),
        np.logical_and(f_c1[begin] > thr, f_c2[begin] < thr),
    )
    early_mask = np.logical_or(f_c1[begin] > thr, f_c2[begin] > thr)
    early_specialization = early_specialization[early_mask]
    late_specialization = np.logical_or(
        np.logical_and(f_c1[end] < thr, f_c2[end] > thr),
        np.logical_and(f_c1[end] > thr, f_c2[end] < thr),
    )
    late_mask = np.logical_or(f_c1[end] > thr, f_c2[end] > thr)
    late_specialization = late_specialization[late_mask]
    out = (
        u.bootstrap_list(early_specialization, np.nanmean),
        u.bootstrap_list(late_specialization, np.nanmean),
    )
    return out


@gpl.ax_adder(three_dim=True)
def plot_trial_points(
    xs,
    f_N,
    f_F,
    ms=2,
    ms_avg_mult=2,
    n_trls=100,
    ax=None,
    f_color=None,
    ctx_color=None,
    r1_color=None,
    r2_color=None,
    plot_avg=True,
):
    out = get_area_activity(xs, f_N, f_F, smooth=False)
    f1_c1, f1_c2 = out["r1"]
    f2_c1, f2_c2 = out["r2"]

    f_coll = (
        f1_c1,
        f2_c2,
        f1_c2,
        f2_c1,
    )
    f_coll = [x[:n_trls] for x in f_coll]
    colors = (r1_color, r1_color, r2_color, r2_color)
    _, p = gpl.plot_highdim_points(
        *f_coll,
        ax=ax,
        colors=colors,
        ms=ms,
    )
    
    if plot_avg:
        f_coll_mus = [np.mean(x, axis=0, keepdims=True) for x in f_coll]
        f1_c1_mu, f2_c2_mu, f1_c2_mu, f2_c1_mu = f_coll_mus
        gpl.plot_highdim_trace(
            np.concatenate((f1_c1_mu, f2_c1_mu), axis=0),
            np.concatenate((f1_c2_mu, f2_c2_mu), axis=0),
            color=f_color,
            p=p,
            ax=ax,
        )
        gpl.plot_highdim_trace(
            np.concatenate((f1_c1_mu, f1_c2_mu), axis=0),
            np.concatenate((f2_c1_mu, f2_c2_mu), axis=0),
            color=ctx_color,
            p=p,
            ax=ax,
        )
        gpl.plot_highdim_points(
            *f_coll_mus,
            colors=colors,
            ax=ax,
            p=p,
            ms=ms * ms_avg_mult,
            mec="k",
            mew=1,
        )
    gpl.clean_3d_plot(ax)
    gpl.make_3d_bars(ax, center=(-10, -10, -5), bar_len=5)


def get_all_geometry_analysis(data, proj=True, n_trls=100):
    out = {}
    for animal, (xs, (f_N, _), (f_F, _)) in data.items():
        x_a = get_area_activity(xs, f_N, f_F, smooth=False)
        f_11, f_21 = x_a["r1"]
        f_12, f_22 = x_a["r2"]
        early = [x[:n_trls] for x in (f_11, f_21, f_12, f_22)]
        late = [x[-n_trls:] for x in (f_11, f_21, f_12, f_22)]

        dec_early = geometry_analysis(*early, proj=proj)
        dec_late = geometry_analysis(*late, proj=proj)
        out[animal] = (dec_early, dec_late)
    return out


def get_geometry_tradeoff(xs, f_N, f_F, n_trls=100, n_reps=5):
    out = get_area_activity(xs, f_N, f_F, smooth=False)
    f_11, f_21 = out["r1"]
    f_12, f_22 = out["r2"]

    early = [x[:n_trls] for x in (f_11, f_21, f_12, f_22)]
    late = [x[-n_trls:] for x in (f_11, f_21, f_12, f_22)]

    early_ps = []
    late_ps = []
    early_ds = []
    late_ds = []
    for i in range(n_reps):
        ds, ps = distance_analysis(*[x[i::n_reps] for x in early])
        early_ds.append(ds)
        early_ps.append(ps)

        ds, ps = distance_analysis(*[x[i::n_reps] for x in late])
        late_ps.append(ps)
        late_ds.append(ds)
    return (
        (
            np.stack(early_ds, axis=-1),
            np.stack(late_ds, axis=-1),
        ),
        (
            np.stack(early_ps, axis=-1),
            np.stack(late_ps, axis=-1),
        ),
    )


def make_all_fluor_maps(animals=ANIMALS):
    out = {}
    for animal in animals:
        data = load_data(animal)
        out[animal] = make_cell_fluor_maps(data)
    return out


def make_all_lick_maps(animals=ANIMALS):
    out = {}
    for animal in animals:
        data = load_data(animal)
        out[animal] = make_lick_maps(data)
    return out


def get_lick_specialization(data, n_trls=100, begin=0, end=-1):
    licks_begin = np.zeros((len(data), 2))
    licks_end = np.zeros_like(licks_begin)
    for i, (animal, licks_a) in enumerate(data.items()):
        out_a = group_licks(licks_a, n_trls=n_trls, smooth=True)
        lick_diff_near = np.squeeze(-np.diff(out_a["near"][1], axis=1)) / np.sum(
            out_a["near"][1], axis=1
        )
        lick_diff_far = np.squeeze(np.diff(out_a["far"][1], axis=1)) / np.sum(
            out_a["far"][1], axis=1
        )
        licks_begin[i] = (lick_diff_near[begin], lick_diff_far[begin])
        licks_end[i] = (lick_diff_near[end], lick_diff_far[end])
    return np.array((licks_begin, licks_end))


def make_geometry_specialization(data, thr=0.1, n_trls=100):
    begin_tradeoff = []
    end_tradeoff = []
    begin_specialization = []
    end_specialization = []
    for xs, (f_N, _), (f_F, _) in data.values():
        specialization = get_specialization_frac(xs, f_N, f_F, thr=thr, n_trls=n_trls)
        begin_specialization.append(np.mean(specialization[0]))
        end_specialization.append(np.mean(specialization[-1]))

        t_early, t_late = get_geometry_tradeoff(xs, f_N, f_F, n_trls=n_trls)[1]
        begin_tradeoff.append(t_early)
        end_tradeoff.append(t_late)
    out = (
        np.array((begin_specialization, end_specialization)),
        np.array((begin_tradeoff, end_tradeoff)),
    )
    return out


def load_data(animal, pattern=PATTERN):
    data = vr2p.ExperimentData(pattern.format(animal))
    return data


def make_lick_maps(
    data,
    remove_licks_on_place=False,
    remove_licks_on_time=False,
    remove_licks_after_reward_sec=4,
    remove_licks_after_reward_cm=10,
):
    results = pd.DataFrame()
    for i_session, vr in enumerate(data.vr):
        # get licking information
        lick = vr.lick.copy()
        trial = vr.trial.copy()
        # select needed info
        lick = lick[["time", "position", "trial_number"]]
        # remove licks outside of trials.
        lick = lick[lick["trial_number"].notnull()]
        lick["trial_number"] = lick["trial_number"].astype("int")
        # remove licks.
        # remove based on proximity to reward
        reward = vr.reward.copy()
        if remove_licks_on_place:
            # Go through reward.
            filter_ind = []  # stores lick indices to remove
            for index, row in reward.iterrows():
                is_trial = lick.trial_number == row.trial_number
                is_in_pos = lick.position.between(
                    row.position, row.position + remove_licks_after_reward_cm
                )
                is_in_time = lick.time.between(
                    row.time,
                    row.time + pd.to_timedelta(remove_licks_after_reward_sec, unit="s"),
                )
                if remove_licks_on_time:
                    filter_ind.append(
                        lick.loc[is_trial & is_in_pos & is_in_time].index.to_numpy()
                    )
                else:
                    filter_ind.append(lick.loc[is_trial & is_in_pos].index.to_numpy())
            lick = lick.drop(np.hstack(filter_ind))
        # remove based on time
        else:
            for index, row in reward.iterrows():
                lick = lick.loc[
                    ~lick.time.between(
                        row.time,
                        row.time
                        + pd.to_timedelta(remove_licks_after_reward_sec, unit="s"),
                    )
                ]
        lick = lick[["position", "trial_number"]]
        #  select data.
        trial = trial[["trial_number", "reward_id", "is_guided"]]
        # adjust trial type (near / far)
        trial["reward_id"] = trial["reward_id"].replace({1: "near", 2: "far"})
        trial = trial.rename(columns={"reward_id": "trial_type"})
        # merge trial information
        lick = pd.merge(lick, trial, on="trial_number", how="inner")
        # filter on position < 230
        lick = lick[lick["position"] < 230]
        # add session info for store
        lick["session"] = i_session
        lick = lick.set_index(["trial_type"])
        lick = lick.sort_index()
        # make trial number cumulative per session type.
        for trial_type in ["near", "far"]:
            # make trial numbers sequential for this trial.
            trial_ranks = rankdata(
                lick.loc[trial_type].trial_number.to_numpy(), method="dense"
            )
            # check what the max trial number is for this set (if present)
            has_info = any(results.index.isin([(trial_type)]))
            cum_trial = results.loc[trial_type]["trial_number"].max() if has_info else 0
            # make cumulative.
            lick.loc[(trial_type), "trial_number"] = trial_ranks + cum_trial
        results = pd.concat((results, lick)).sort_index()
    results = results.sort_values(["trial_type", "trial_number", "position"])
    return results


def _group_licks(
    licks,
    r1_area=R1_AREA,
    r2_area=R2_AREA,
    pos_key="position",
    trial_key="trial_number",
    smooth=False,
    n_trls=50,
):
    u_trls = np.unique(licks[trial_key])
    out = np.zeros((len(u_trls), 2))
    for i, ut in enumerate(u_trls):
        pos = licks[licks[trial_key] == ut][pos_key]
        out[i, 0] = np.sum(np.logical_and(pos >= r1_area[0], pos < r1_area[1]))
        out[i, 1] = np.sum(np.logical_and(pos >= r2_area[0], pos < r2_area[1]))
    if smooth:
        filt = np.ones((n_trls, 1)) / n_trls
        out = sig.convolve(out, filt, mode="valid")
        u_trls = sig.convolve(u_trls, filt[:, 0], mode="valid")
    return u_trls, out


def group_licks(lick_data, guide_key="is_guided", no_guide=False, **kwargs):
    if no_guide:
        lick_data = lick_data[~lick_data[guide_key].astype(bool)]
    out = {}
    near_licks = lick_data[lick_data.index == "near"]
    out["near"] = _group_licks(near_licks, **kwargs)

    far_licks = lick_data[lick_data.index == "far"]
    out["far"] = _group_licks(far_licks, **kwargs)
    return out


def make_cell_fluor_maps(data, bin_size=5, min_speed=5, track_size=230):
    spatial_edges = np.arange(0, track_size + bin_size, bin_size)
    mid_points = spatial_edges[1:] - (bin_size / 2)  # mid points of spatial bins.

    all_fields_N = []
    all_fields_F = []

    day_ind_N = []
    day_ind_F = []

    for session in range(len(data.vr)):
        print(f"On Session {session}")
        vr = data.vr[session]
        trial = vr.trial.copy()
        position = vr.path.frame.copy().reset_index()
        # merge reward_id info
        position = position.merge(
            trial[["trial_number", "reward_id"]], on="trial_number"
        )
        F = data.signals.multi_session.Fdemix[int(session)]
        F = F[:]
        F = F - np.min(F, axis=1)[..., np.newaxis]
        # calculate df (Based on all frames.)
        dF_F0, _ = vr2p.signal.df_over_f0(
            F, "maximin", subtract_min=True, sigma_baseline=20, window_size=200
        )

        for reward_id in [1, 2]:
            # print(f'  On reward_id {reward_id}')

            selected_trials = trial.loc[(trial.reward_id == reward_id), "trial_number"]
            selected_frames = position.loc[
                position["trial_number"].isin(selected_trials), "frame"
            ]
            position["speed"] = position.vr2p.rolling_speed(
                window_size=100, ignore_threshold=7.5
            )
            # filter frames with min speed and in 'selected_frames'
            filtered_frames = position.loc[
                (position["speed"] >= min_speed)
                & position["frame"].isin(selected_frames),
                "frame",
            ]
            filtered_pos_data = position.loc[
                position["frame"].isin(filtered_frames),
                ["frame", "position", "trial_number"],
            ]

            for trial_n in selected_trials:
                # print(f'On trial {trial_n}')

                single_pos_data = filtered_pos_data[
                    filtered_pos_data.trial_number == trial_n
                ]

                if len(single_pos_data.position) == 0 or (
                    len(single_pos_data.position) > 0
                    and np.nanmax(single_pos_data.position) < 226
                ):
                    # print(f'    skipping incomplete trial {trial_n}')
                    continue
                bin_pos = single_pos_data.copy()
                # change trials to rank order (for filling out matrix).
                bin_pos["trial_number"] = (
                    bin_pos["trial_number"].rank(method="dense") - 1
                ).astype(int)

                # Assign bin position.
                bin_pos["bin"] = (
                    pd.cut(
                        bin_pos.position,
                        spatial_edges,
                        include_lowest=True,
                        labels=False,
                    )
                    .to_numpy()
                    .astype(int)
                )
                bin_pos = bin_pos.loc[bin_pos.bin >= 0]  # valid bins only.
                # prepare binF matrix.
                num_trials = bin_pos.trial_number.max() + 1
                num_bins = spatial_edges.size - 1
                num_cells = F.shape[0]
                if np.isnan(num_trials):
                    num_trials = 0
                binF_mat = np.full(
                    (num_trials, num_bins, num_cells), np.nan
                )  # holds mean fluorescence data.
                # Get frames per trial and bin.
                bin_pos = bin_pos.groupby(["trial_number", "bin"]).agg({"frame": list})
                # get mean dF value for each bin, trial frame set.
                for index, row in bin_pos.iterrows():
                    binF_mat[index[0], index[1], :] = np.mean(
                        dF_F0[:, row["frame"]], axis=1
                    )

                binF_mat = np.swapaxes(binF_mat, 1, 2)

                if reward_id == 1:
                    all_fields_N.append(binF_mat)
                    day_ind_N.append(session)
                if reward_id == 2:
                    all_fields_F.append(binF_mat)
                    day_ind_F.append(session)
    out = (
        mid_points,
        (np.concatenate(all_fields_N, axis=0), day_ind_N),
        (np.concatenate(all_fields_F, axis=0), day_ind_F),
    )
    # bin 12 - 19 is cue
    # bin 26 - 29 is R1
    # bin 36 - 39 is R2
    return out


def get_area_activity(
    xs,
    f_N,
    f_F,
    cue_area=CUE_AREA,
    r1_area=R1_AREA,
    r2_area=R2_AREA,
    post_cue=POST_CUE,
    smooth=True,
    n_trls=100,
):
    cue = np.logical_and(cue_area[0] <= xs, cue_area[1] > xs)
    r1 = np.logical_and(r1_area[0] <= xs, r1_area[1] > xs)
    r2 = np.logical_and(r2_area[0] <= xs, r2_area[1] > xs)
    post_cue = np.logical_and(post_cue[0] <= xs, post_cue[1] > xs)
    r1r2 = np.logical_or(r1, r2)

    masks = {"cue": cue, "r1": r1, "r2": r2, "r1r2": r1r2, "post_cue": post_cue}
    filt = np.ones((n_trls, 1)) / n_trls
    out = {}
    for k, mask in masks.items():
        f_N_k = np.nanmean(f_N[..., mask], axis=-1)
        f_F_k = np.nanmean(f_F[..., mask], axis=-1)

        if smooth:
            f_N_k = sig.convolve(f_N_k, filt, mode="valid", method="direct")
            f_F_k = sig.convolve(f_F_k, filt, mode="valid", method="direct")
        out[k] = (f_N_k, f_F_k)
    return out


def decompose_distance_matrix_nnls(mat, schema, n_neurs, indiv_dists=False):
    mat = mat * n_neurs
    mat[mat < 0] = 0

    design = []
    targ = []
    for i, j in it.combinations(range(len(schema)), 2):
        if indiv_dists:
            add_d = np.zeros(len(schema))
            add_d[i] = 1
            add_d[j] = 1
        else:
            add_d = (2,)
        design_ij = np.concatenate((np.abs(schema[i] - schema[j]), add_d))
        design.append(design_ij)
        targ.append(mat[i, j])
    out = sopt.nnls(np.array(design), np.array(targ))
    # out = np.linalg.lstsq(np.array(design), np.array(targ), rcond=None)
    x = out[0]

    d_lins = x[: schema.shape[1]]
    d_nl = x[schema.shape[1] :]

    d_nl = np.mean(d_nl)
    return d_lins, d_nl, None


def distance_analysis(*data, schema=None):
    if schema is None:
        schema = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    X, y = na.make_data_labels(*data)
    data = rsa.data.Dataset(X, obs_descriptors={"stimulus": y})
    rdm = rsa.rdm.calc_rdm(data, descriptor="stimulus", noise=None, method="crossnobis")
    rdm_arr = rdm.get_matrices()[0]

    n_neurs = X.shape[1]
    d_lin, d_nl, _ = decompose_distance_matrix_nnls(rdm_arr, schema, n_neurs)
    p_nl = (1 / 2) * d_nl
    p_l = (1 / 2) * np.mean(d_lin)
    t = p_l / (p_l + p_nl)

    return (np.sqrt(np.concatenate((d_lin, d_nl[None]))), (p_l, p_nl, t))


def geometry_analysis(*data, test_prop=0.1, n_folds=100, proj=False, **kwargs):
    f_11, f_21, f_12, f_22 = [x[..., None] for x in data]
    tr_X_con, tr_l_con = na.make_data_labels(f_11, f_21)
    te_X_con, te_l_con = na.make_data_labels(f_12, f_22)
    context1 = na.fold_skl_shape(
        tr_X_con,
        tr_l_con,
        n_folds,
        test_prop=test_prop,
        mean=False,
        c_gen=te_X_con,
        l_gen=te_l_con,
        ret_projections=proj,
        **kwargs,
    )
    context2 = na.fold_skl_shape(
        te_X_con,
        te_l_con,
        n_folds,
        test_prop=test_prop,
        mean=False,
        c_gen=tr_X_con,
        l_gen=tr_l_con,
        ret_projections=proj,
        **kwargs,
    )

    tr_X_stop, tr_l_stop = na.make_data_labels(
        np.concatenate((f_11, f_22), axis=0), np.concatenate((f_12, f_21), axis=0)
    )
    stop = na.fold_skl_shape(
        tr_X_stop,
        tr_l_stop,
        n_folds,
        test_prop=test_prop,
        mean=False,
        ret_projections=proj,
        **kwargs,
    )

    tr_X_pat, tr_l_pat = na.make_data_labels(f_11, f_12)
    te_X_pat, te_l_pat = na.make_data_labels(f_21, f_22)
    pattern1 = na.fold_skl_shape(
        tr_X_pat,
        tr_l_pat,
        n_folds,
        test_prop=test_prop,
        mean=False,
        c_gen=te_X_pat,
        l_gen=te_l_pat,
        ret_projections=proj,
        **kwargs,
    )
    pattern2 = na.fold_skl_shape(
        te_X_pat,
        te_l_pat,
        n_folds,
        test_prop=test_prop,
        mean=False,
        c_gen=tr_X_pat,
        l_gen=tr_l_pat,
        ret_projections=proj,
        **kwargs,
    )
    out = ((stop,), (context1, context2), (pattern1, pattern2))
    return out


def plot_all_proj(*decs, axs=None, **kwargs):
    plot_all_dec(
        *decs, axs=axs, pkeys=("projection", "projection_gen"), chance_level=0, **kwargs
    )


def plot_all_dec(
    *decs,
    colors=None,
    one_row=False,
    axs=None,
    fwid=2,
    c_bounds=(0.2, 0.99),
    cm="Blues",
    titles=None,
    **kwargs,
):
    if titles is None:
        titles = ("",) * len(decs[0])
    if colors is None:
        cmap = plt.get_cmap(cm)
        colors = cmap(np.linspace(*c_bounds, len(decs)))
    if axs is None:
        if one_row:
            n_rows = 1
        else:
            n_rows = len(decs)
        n_cols = len(decs[0])
        _, axs = plt.subplots(
            n_rows,
            n_cols,
            figsize=(fwid * n_cols, fwid * n_rows),
            sharey="all",
            squeeze=False,
        )
    if one_row:
        axs = np.concatenate((axs,) * len(decs), axis=0)
    for i, dec_i in enumerate(decs):
        for j, dec_ij in enumerate(dec_i):
            plot_dec(dec_ij, ax=axs[i, j], color=colors[i], **kwargs)
            gpl.clean_plot(axs[i, j], j)
            if i == 0:
                axs[i, j].set_title(titles[j])


@gpl.ax_adder()
def plot_dec(dec, pkeys=("score", "score_gen"), color=None, chance_level=0.5, ax=None):
    xlabels = []
    for i, pk in enumerate(pkeys):
        for j, dec_j in enumerate(dec):
            dec_ij = dec_j.get(pk)
            if dec_ij is not None:
                gpl.violinplot([dec_ij], [i + j / 10], color=[color], ax=ax)
        xlabels.append(pk)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels)
    gpl.add_hlines(chance_level, ax)
