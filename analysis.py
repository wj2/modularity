import functools as ft
import numpy as np
import sklearn.preprocessing as skp
import sklearn.linear_model as sklm
import sklearn.cluster as skcl
import sklearn.mixture as skmx
import sklearn.svm as skm
import sklearn.model_selection as skms
import sklearn.pipeline as sklpipe
import sklearn.feature_selection as skfs
import itertools as it
import scipy.stats as sts
import scipy.special as ss
import scipy.optimize as spo
import tensorflow as tf

import general.utility as u
import general.stan_utility as su
import general.neural_analysis as na
import modularity.simple as ms
import modularity.auxiliary as maux
import disentangled.data_generation as dg
import composite_tangling.code_creation as cc
import sklearn.metrics.pairwise as skmp

tfk = tf.keras


class ModularizerCode(cc.Code):
    def __init__(
        self,
        model,
        group_ind=None,
        dg_model=None,
        source_distr=None,
        n_values=2,
        noise_cov=0.1**2,
        use_layer=None,
    ):
        self.use_layer = use_layer
        context_signal = model.integrate_context
        if dg_model is not None and source_distr is None:
            n_feats_all = dg_model.input_dim
            if context_signal:
                n_feats_all = n_feats_all - model.n_groups
            self.n_feats_all = n_feats_all
        elif dg_model is None and source_distr is not None:
            self.n_feats_all = source_distr.rvs(1).shape[1]
        else:
            raise IOError("one of dg_model or source_distr must be provided")
        self.group_ind = group_ind
        if self.group_ind is not None:
            self.group = model.groups[group_ind]
        else:
            self.group = np.arange(self.n_feats_all, dtype=int)
        self.context_signal = context_signal
        self.n_feats = len(self.group)
        self.n_values = n_values
        if use_layer is not None:
            n_units = model.model.layers[use_layer].output.shape[1]
        else:
            n_units = model.hidden_dims
        self.n_neurs = n_units
        self.n_stimuli = self.n_values**self.n_feats
        self.noise_distr = sts.multivariate_normal(np.zeros(self.n_neurs), noise_cov)
        self.model = model
        self.dg_model = dg_model
        self.power = self._get_avg_power()
        self.stim = self._get_all_stim()

    def _get_avg_power(self, n_avg=10000):
        samps, inp_reps = self.sample_dg_reps(n_avg)
        reps = self.model.get_representation(inp_reps, layer=self.use_layer)
        pwr = np.mean(np.sum(reps**2, axis=1), axis=0)
        return pwr

    def _get_all_stim(self):
        stim = list(it.product(range(self.n_values), repeat=self.n_feats))
        return stim

    def get_random_full_stim(self, **kwargs):
        return self._get_random_stim(self.n_feats_all, **kwargs)

    def get_random_stim(self, **kwargs):
        return self._get_random_stim(self.n_feats, **kwargs)

    def _get_random_stim(self, n_feats, non_nan=None):
        s, _ = self.sample_dg_reps(1)
        s = s[0].astype(float)
        if non_nan is not None:
            if n_feats < self.n_feats_all:
                mask = np.logical_not(np.isin(self.group, non_nan))
            else:
                mask = np.logical_not(np.isin(np.arange(n_feats), non_nan))
            s[: self.n_feats_all][mask] = np.nan
        return s

    def get_full_stim(self, stim):
        f_stim, _ = self.sample_dg_reps(stim.shape[0])
        f_stim[:, self.group] = stim
        return f_stim

    def sample_dg_reps(self, n_samps=1000):
        stim, reps = self.dg_model.sample_reps(n_samps)
        if self.context_signal:
            stim[:, self.n_feats_all :] = 0
            stim[:, self.n_feats_all + self.group_ind] = 1
            reps = self.dg_model.get_representation(stim)
        return stim, reps

    def get_nan_stim(self, stim, ref_stim=None):
        if ref_stim is None:
            n_stim, _ = self.sample_dg_reps(stim.shape[0])
        else:
            n_stim = ref_stim
        mask = np.logical_not(np.isnan(stim))
        n_stim[mask] = stim[mask]
        return n_stim

    def get_representation(
        self, stim, noise=False, ret_stim=False, ref_stim=None, layer=None
    ):
        if layer is not None:
            print("the layer parameter is ignored")
        stim = np.array(stim)
        if len(stim.shape) == 1:
            stim = np.expand_dims(stim, 0)
        if np.any(np.isnan(stim)):
            stim = self.get_nan_stim(stim, ref_stim=ref_stim)
        if stim.shape[1] == len(self.group):
            stim = self.get_full_stim(stim)
        stim_rep = self.dg_model.get_representation(stim)
        reps = self.model.get_representation(stim_rep, layer=self.use_layer)
        if noise:
            reps = self._add_noise(reps)
        if ret_stim:
            out = (reps, stim)
        else:
            out = reps
        return out

    def _sample_noisy_reps(self, cs, n, add_noise=True, ret_stim=False, ref_stim=None):
        cs = np.array(cs)
        if len(cs.shape) == 1:
            cs = np.expand_dims(cs, 0)
        r_inds = np.random.choice(cs.shape[0], int(n))
        out = self.get_representation(
            cs[r_inds], noise=add_noise, ret_stim=ret_stim, ref_stim=ref_stim
        )
        return out

    def compute_shattering(self, n_reps=5, thresh=0.6, max_parts=100, **dec_args):
        partitions = self._get_partitions(random_thr=max_parts)
        n_parts = len(partitions)
        pcorrs = np.zeros((n_parts, n_reps))
        stim_arr = np.array(self.stim)
        for i, ps1 in enumerate(partitions):
            ps2 = list(set(range(self.n_stimuli)).difference(ps1))
            c1 = stim_arr[ps1]
            c2 = stim_arr[ps2]
            pcorrs[i] = self.decode_rep_classes(c1, c2, n_reps=n_reps, **dec_args)
        n_c = np.sum(np.mean(pcorrs, axis=1) > thresh)
        n_dim = np.log2(2 * n_c)
        n_dim_poss = np.log2(2 * n_parts)
        return n_dim, n_dim_poss, pcorrs

    def compute_within_group_ccgp(
        self, n_reps=10, max_combos=20, fix_features=1, **kwargs
    ):
        if fix_features < 0:
            fix_features = len(self.group) + fix_features
        combos = list(it.combinations(self.group, fix_features))
        n_possible_combos = int(ss.comb(len(self.group), fix_features))
        if n_possible_combos > max_combos:
            comb_inds = np.random.choice(
                range(n_possible_combos), max_combos, replace=False
            )
            combos = np.array(list(combos))[comb_inds]
            n_possible_combos = max_combos
        out = np.zeros((len(combos), n_reps))
        for i, combo in enumerate(combos):
            options = list(set(self.group).difference(combo))
            td = np.random.choice(options, 1)[0]
            gd = combo
            out[i] = self.compute_specific_ccgp(td, gd, n_reps=n_reps, **kwargs)
        return out

    def compute_across_group_ccgp(
        self, n_reps=10, max_combos=20, fix_features=1, **kwargs
    ):
        if fix_features < 0:
            fix_features = len(self.group) + fix_features
        all_inds = np.arange(self.n_feats_all, dtype=int)
        non_group_inds = set(all_inds).difference(self.group)

        ngi_iter = it.combinations(non_group_inds, fix_features)
        combos = list(it.product(self.group, ngi_iter))
        n_possible_combos = len(self.group) * len(non_group_inds)
        if n_possible_combos > max_combos:
            comb_inds = np.random.choice(
                range(n_possible_combos), max_combos, replace=False
            )
            combos = np.array(list(combos), dtype=object)[comb_inds]
            n_possible_combos = max_combos
        out = np.zeros((len(combos), n_reps))
        for i, (td, gd) in enumerate(combos):
            out[i] = self.compute_specific_ccgp(td, gd, n_reps=n_reps, **kwargs)
        return out

    def compute_specific_ccgp(
        self,
        train_dim,
        gen_dim,
        train_dist=1,
        gen_dist=1,
        n_reps=10,
        ref_stim=None,
        train_noise=False,
        n_train=10,
        balance_training=False,
        **dec_kwargs,
    ):
        if not u.check_list(gen_dim):
            gen_dim = (gen_dim,)
        gen_dim = np.array(gen_dim)
        all_dim = np.concatenate(((train_dim,), gen_dim))
        cond = (
            ref_stim is None
            and train_dim in self.group
            and np.all(np.isin(gen_dim, self.group))
        )
        if cond:
            ref_stim = self.get_random_full_stim(non_nan=all_dim)
        elif ref_stim is None:
            ref_stim = self.get_random_full_stim(non_nan=all_dim)
        tr_stim = np.mod(
            np.array(
                tuple(
                    rs + train_dist * (i == train_dim) for i, rs in enumerate(ref_stim)
                )
            ),
            self.n_values,
        )
        gen_stim1 = np.mod(
            np.array(
                tuple(
                    rs + gen_dist * np.isin(i, gen_dim) for i, rs in enumerate(ref_stim)
                )
            ),
            self.n_values,
        )
        gen_stim2 = np.mod(
            np.array(
                tuple(
                    rs + gen_dist * np.isin(i, gen_dim) for i, rs in enumerate(tr_stim)
                )
            ),
            self.n_values,
        )

        pcorr = self.decode_rep_classes(
            ref_stim,
            tr_stim,
            c1_test=gen_stim1,
            c2_test=gen_stim2,
            n_reps=n_reps,
            train_noise=train_noise,
            n_train=n_train,
            balance_training=balance_training,
            **dec_kwargs,
        )

        return pcorr

    def _get_ccgp_stim_sets(self):
        f_combs = list(it.combinations(range(self.n_values), 2))
        train_sets = []
        test_sets = []
        for i in range(self.n_feats):
            for j, comb in enumerate(f_combs):
                train_stim_ind = np.random.choice(self.n_stimuli, 1)[0]
                train_stim = self.stim[train_stim_ind]

                c1_eg_stim = list(train_stim)
                c1_eg_stim[i] = comb[0]
                c1_eg_stim = tuple(c1_eg_stim)

                c2_eg_stim = list(train_stim)
                c2_eg_stim[i] = comb[1]
                c2_eg_stim = tuple(c2_eg_stim)

                stim_arr = np.array(self.stim)
                c1_test_stim = stim_arr[stim_arr[:, i] == comb[0]]
                c1_exclusion = np.any(c1_test_stim != c1_eg_stim, axis=1)
                c1_test_stim = c1_test_stim[c1_exclusion]

                c2_test_stim = stim_arr[stim_arr[:, i] == comb[1]]
                c2_exclusion = np.any(c2_test_stim != c2_eg_stim, axis=1)
                c2_test_stim = c2_test_stim[c2_exclusion]

                train_sets.append((c1_eg_stim, c2_eg_stim))
                test_sets.append((c1_test_stim, c2_test_stim))
        return train_sets, test_sets


def expected_pn(modules, tasks, inp_dim, acc=0.95, sigma=0.1, n_values=2):
    s = sts.norm(0, 1).ppf(acc)
    return modules * (s**2) * (sigma**2) * (n_values**inp_dim)


@u.arg_list_decorator
def train_variable_models(
    group_size,
    tasks_per_group,
    group_maker,
    model_type,
    n_reps=2,
    n_overlap=(0,),
    **kwargs,
):
    out_ms = np.zeros(
        (
            len(group_size),
            len(tasks_per_group),
            len(group_maker),
            len(model_type),
            len(n_overlap),
            n_reps,
        ),
        dtype=object,
    )
    out_hs = np.zeros_like(out_ms)
    for i, j, k, l_, m in u.make_array_ind_iterator(out_ms.shape[:-1]):
        out = train_n_models(
            group_size[i],
            tasks_per_group[j],
            group_maker=group_maker[k],
            model_type=model_type[l_],
            n_overlap=n_overlap[m],
            n_reps=n_reps,
            **kwargs,
        )
        out_ms[i, j, k, l_, m], out_hs[i, j, k, l_, m] = out
    return out_ms, out_hs


def avg_corr(k, n=2):
    n_s = 0.5 * ss.binom(n**k, n ** (k - 1))
    f = np.arange(1, n ** (k - 2) + 1)
    n_f = k * 2 * ss.binom(n ** (k - 1), f)
    r = 1 - 4 * f / (n**k)
    corr = (1 / n_s) * (k + np.sum(n_f * r))
    return corr


def cluster_graph(m, n_clusters=None, layer=None, **kwargs):
    if n_clusters is None:
        n_clusters = m.n_groups
    if layer is None:
        layer = -1
    ws = m.model.weights
    # ws_mats = ws[::2]

    w_ih = ws[0]
    w_ho = ws[2]
    n_hidden_neurs = w_ih.shape[1]
    n_out_neurs = w_ho.shape[1]

    w_ih_abs = np.abs(w_ih)
    w_ho_abs = np.abs(w_ho)
    z_i = np.zeros((w_ih.shape[0], w_ih.shape[0]))
    z_io = np.zeros((w_ih.shape[0], w_ho.shape[1]))

    z_h = np.zeros((w_ih.shape[1], w_ih.shape[1]))
    z_o = np.zeros((w_ho.shape[1], w_ho.shape[1]))

    wm_full = np.block(
        [[z_i, w_ih_abs, z_io], [w_ih_abs.T, z_h, w_ho_abs], [z_io.T, w_ho_abs.T, z_o]]
    )

    cl = skcl.SpectralClustering(n_clusters, affinity="precomputed")
    clusters = cl.fit_predict(wm_full)
    h_clusters = clusters[-(n_hidden_neurs + n_out_neurs) : -n_out_neurs]
    return h_clusters


def cluster_max_corr(m, n_clusters=None, n_samps=5000, ret_overlap=False, **kwargs):
    rep_act, out_act = sample_all_contexts(m, n_samps=n_samps, ret_out=True)
    masks = []
    for i, ra_i in enumerate(rep_act):
        oa_i = out_act[i]
        task_masks = []
        for j in range(oa_i.shape[1]):
            m = sklm.LogisticRegression(penalty="l1", solver="liblinear")
            m.fit(ra_i.numpy(), oa_i[:, j].numpy() > 0.5)
            task_masks.append(np.abs(m.coef_) > 0)
        c_mask = np.sum(task_masks, axis=0) > 0
        masks.append(c_mask * (i + 1))
    out = np.concatenate(masks)
    multi_mask = np.sum(out > 0, axis=0) > 1
    any_mask = np.sum(out > 0, axis=0) > 0
    out[:, multi_mask] = 0
    out_clusters = np.sum(out, axis=0)
    if ret_overlap:
        overlap = np.sum(multi_mask) / np.sum(any_mask)
        out = (out_clusters, overlap)
    else:
        out = out_clusters
    return out


exp_fields = [
    "group_size",
    "tasks_per_group",
    "group_method",
    "model_type",
    "group_overlap",
    "n_groups",
    "args_kernel_init_std",
    "args_group_width",
    "args_activity_reg",
]
target_fields = ["gm", "shattering", "within_ccgp", "across_ccgp"]


def explain_clustering(df, target_fields=target_fields, explainer_fields=exp_fields):
    targ = df[target_fields]
    targ = targ  # - np.mean(targ, axis=0)
    ohe = skp.OneHotEncoder()
    preds = ohe.fit_transform(df[explainer_fields])
    fnames = ohe.get_feature_names_out(explainer_fields)
    r = sklm.Ridge()
    r.fit(preds, targ)

    coefs_all = r.coef_
    inter_all = r.intercept_
    out_coefs = {}
    for fn in explainer_fields:
        mask = np.array(list((fn in el) for el in fnames))
        vals = list(el.split("_")[-1] for i, el in enumerate(fnames) if mask[i])
        out_coefs[fn] = (vals, coefs_all[:, mask])
    return out_coefs, inter_all, target_fields


def task_dim_analytic(d, n_tasks, n_cons=2):
    a = 1 / (n_cons * d)
    dim = n_tasks / (1 + (n_tasks - 1) * a)
    return dim


def prob_eliminate(d, n_tasks, n_cons=2):
    dm1d = (d - 1) / d
    p = 1 - dm1d**2 + dm1d * (1 / (2 * d))
    1 - p  # is prob of elimination

    elim = 1 - p**n_tasks
    return elim


def estimate_prob_eliminate(targs, lvs):
    perf = np.zeros((lvs.shape[1], targs.shape[1]))
    for i in range(lvs.shape[1]):
        m1 = lvs[:, i] == 1
        m2 = lvs[:, i] == -1
        for j in range(targs.shape[1]):
            m = skm.LinearSVC(dual="auto")
            lv_m1 = lvs[m1]
            tj1 = targs[m1, j]
            if np.var(tj1) == 0:
                p1 = 1
            else:
                m_j1 = m.fit(lv_m1, tj1)
                p1 = m_j1.score(lv_m1, tj1)

            m = skm.LinearSVC(dual="auto")
            lv_m2 = lvs[m2]
            tj2 = targs[m2, j]
            if np.var(tj2) == 0:
                p2 = 1
            else:
                m_j2 = m.fit(lv_m2, tj2)
                p2 = m_j2.score(lv_m2, tj2)
            perf[i, j] = np.mean([p1, p2])
    return perf


def _task_dim(
    m, group_dim=False, noncon_dim=False, use_group=0, n_samps=1000, diff_break=False
):
    if group_dim:
        _, _, targ = m.get_x_true(group_inds=use_group, n_train=n_samps)
    elif noncon_dim:
        gs = m.groups
        g0 = gs[0][0]
        _, true, targ = m.get_x_true(n_train=2 * n_samps)
        mask = true[:, g0] == 1
        targ = targ[mask]
    else:
        _, _, targ = m.get_x_true(n_train=n_samps)

    pr, pv = u.participation_ratio(targ, ret_pv=True)
    if diff_break:
        dim = np.argmax(np.abs(np.diff(pv))) + 1
    else:
        dim = pr
    return dim, pv


@u.arg_list_decorator
def task_dimensionality(
    group_size,
    n_overlap,
    tasks_per_group,
    n_groups,
    group_maker=ms.overlap_groups,
    n_reps=3,
    inp_dim=20,
    models={"linear": ms.LinearModularizer, "coloring": ms.ColoringModularizer},
):
    out_dims_dict = {}
    out_pv_dict = {}
    for key, model in models.items():
        dims_tot = np.zeros(
            (
                len(group_size),
                len(n_overlap),
                len(tasks_per_group),
                len(n_groups),
                n_reps,
            )
        )
        dims_wi = np.zeros_like(dims_tot)
        dims_nc = np.zeros_like(dims_tot)
        pv_tot = np.zeros(
            (
                len(group_size),
                len(n_overlap),
                len(tasks_per_group),
                len(n_groups),
                n_reps,
                max(tasks_per_group),
            )
        )
        pv_wi = np.zeros_like(pv_tot)
        pv_nc = np.zeros_like(pv_tot)

        for ind in u.make_array_ind_iterator(dims_tot.shape):
            (i, j, k, l_, m) = ind
            m_null = model(
                inp_dim,
                group_size=group_size[i],
                n_groups=n_groups[l_],
                group_maker=group_maker,
                tasks_per_group=tasks_per_group[k],
                n_overlap=n_overlap[j],
                single_output=True,
                integrate_context=True,
            )
            n_tasks = tasks_per_group[k]
            pv_ind = ind + (slice(0, n_tasks),)
            dims_tot[ind], pv_tot[pv_ind] = _task_dim(m_null)
            dims_wi[ind], pv_wi[pv_ind] = _task_dim(m_null, group_dim=True)
            dims_nc[ind], pv_nc[pv_ind] = _task_dim(m_null, noncon_dim=True)
        out_dims_dict[key] = (dims_tot, dims_wi, dims_nc)
        out_pv_dict[key] = (pv_tot, pv_wi, pv_nc)
    return out_dims_dict, out_pv_dict


def contrast_rich_lazy(
    inp_dim, rep_dim, init_bounds=(0.01, 3), n_inits=20, train_epochs=0, **kwargs
):
    weight_vars = np.linspace(*(init_bounds + (n_inits,)))
    source_distr = u.MultiBernoulli(0.5, inp_dim)
    for i, wv in enumerate(weight_vars):
        kernel_init = tfk.initializers.RandomNormal(stddev=wv)
        fdg_ut = dg.FunctionalDataGenerator(
            inp_dim,
            (),
            rep_dim,
            source_distribution=source_distr,
            use_pr_reg=True,
            kernel_init=kernel_init,
            **kwargs,
        )
        fdg_ut.fit(epochs=train_epochs, verbose=False)
        dim_i = fdg_ut.representation_dimensionality(participation_ratio=True)

        ident = ms.IdentityModularizer(inp_dim, n_groups=1)
        out_i = apply_geometry_model_list([ident], fdg_ut, group_ind=None)
        shatt_i = out_i[0][0]
        ccgp_i = out_i[1][0]
        if i == 0:
            dims = np.zeros(n_inits)
            shatts = np.zeros((n_inits,) + shatt_i.shape)
            ccgps = np.zeros((n_inits,) + ccgp_i.shape)
        shatts[i] = shatt_i
        ccgps[i] = ccgp_i
        dims[i] = dim_i
    return weight_vars, dims, shatts, ccgps


def train_n_models(
    group_size,
    tasks_per_group,
    group_width=200,
    fdg=None,
    n_reps=2,
    n_groups=5,
    group_maker=ms.random_groups,
    model_type=ms.ColoringModularizer,
    epochs=5,
    verbose=False,
    act_reg_weight=0,
    noise=0.1,
    inp_noise=0.01,
    n_overlap=0,
    constant_init=None,
    single_output=False,
    integrate_context=False,
    remove_last_inp=False,
    kernel_init=None,
    out_kernel_init=None,
    additional_hidden=(),
    use_early_stopping=True,
    **training_kwargs,
):
    if fdg is None:
        use_mixer = False
    else:
        use_mixer = True
    inp_dim = fdg.input_dim
    out_ms = []
    out_hs = []
    for i in range(n_reps):
        m_i = model_type(
            inp_dim,
            group_size=group_size,
            n_groups=n_groups,
            group_maker=group_maker,
            use_dg=fdg,
            group_width=group_width,
            use_mixer=use_mixer,
            tasks_per_group=tasks_per_group,
            act_reg_weight=act_reg_weight,
            noise=noise,
            inp_noise=inp_noise,
            constant_init=constant_init,
            n_overlap=n_overlap,
            single_output=single_output,
            integrate_context=integrate_context,
            remove_last_inp=remove_last_inp,
            kernel_init=kernel_init,
            out_kernel_init=out_kernel_init,
            additional_hidden=additional_hidden,
            use_early_stopping=use_early_stopping,
        )
        h_i = m_i.fit(epochs=epochs, verbose=verbose, **training_kwargs)
        out_ms.append(m_i)
        out_hs.append(h_i)
    return out_ms, out_hs


def contextual_performance(
    mod,
    context_ind=-1,
    c1_ind=0,
    c2_ind=1,
    n_samps=200,
    model=skm.LinearSVC,
    n_folds=20,
    rep_kwargs=None,
    **kwargs,
):
    if rep_kwargs is None:
        rep_kwargs = {}
    stim, reps = mod.sample_reps(n_samps, **rep_kwargs)
    targ = np.zeros(n_samps)
    mask = stim[:, context_ind] == 1
    targ[mask] = stim[mask, c1_ind]
    targ[~mask] = stim[~mask, c2_ind]

    pipe = na.make_model_pipeline(model, dual="auto", **kwargs)
    out = skms.cross_validate(pipe, reps, targ, cv=n_folds)
    return out["test_score"]


# def task_dimensionality(n_tasks, n_g, contexts, m_type, n_overlap=0,
#                         group_maker=ms.overlap_groups, group_inds=None):
#     inp_dim = n_g*contexts + contexts
#     source_distr = u.MultiBernoulli(.5, inp_dim)

#     fdg = dg.FunctionalDataGenerator(inp_dim, (300,), 400,
#                                      source_distribution=source_distr,
#                                      use_pr_reg=True)
#     m = m_type(inp_dim, group_size=n_g, n_groups=contexts, use_dg=fdg,
#                group_maker=group_maker,
#                use_mixer=True, tasks_per_group=n_tasks, n_overlap=n_overlap,
#                single_output=True, integrate_context=True)
#     x, true, targ = m.get_x_true(group_inds=group_inds)

#     return true, targ


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


def threshold_clusters(groups, w_matrix, cumu_weight=0.9):
    w_abs = np.abs(np.array(w_matrix).T)
    w_abs = w_abs / np.sum(w_abs, axis=1, keepdims=True)

    w_sim = np.zeros((w_abs.shape[0], w_abs.shape[0]))
    for i, j in it.product(range(w_abs.shape[0]), repeat=2):
        inds_i = np.argsort(w_abs[i])[::-1]
        wi_sum = np.cumsum(w_abs[i, inds_i])
        ind_set_i = set(inds_i[wi_sum < cumu_weight])
        inds_j = np.argsort(w_abs[j])[::-1]
        wj_sum = np.cumsum(w_abs[j, inds_j])
        ind_set_j = set(inds_j[wj_sum < cumu_weight])
        avg_len = np.mean([len(ind_set_i), len(ind_set_j)])
        w_sim[i, j] = len(ind_set_i.intersection(ind_set_j)) / avg_len
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


def _reorg_matrix(wm, groups):
    u_groups = np.unique(groups)
    n_groups = len(u_groups)
    n_tasks = int(len(groups) / n_groups)
    new_wm = np.zeros((wm.shape[0], n_groups, n_tasks))
    for i, g in enumerate(u_groups):
        new_wm[:, i] = wm[:, groups == g]
    return new_wm


def standardize_wm(wm, flat=True):
    if flat:
        axis = None
    else:
        axis = 0
    mu_wm = np.mean(wm, axis=axis, keepdims=True)
    std_wm = np.std(wm, axis=axis, keepdims=True)
    new_wm = (wm - mu_wm) / std_wm
    return new_wm


def compute_prob_cluster(
    wt, std_prior=1, model_path="modularity/wm_mixture.pkl", **kwargs
):
    n_units, n_groups, n_tasks = wt.shape
    stan_dict = dict(N=n_units, M=n_groups, T=n_tasks, std_prior=std_prior, W=wt)
    out = su.fit_model(stan_dict, model_path, arviz_convert=False, **kwargs)
    fit, fit_az, diag = out
    return out


def likely_clusters(groups, w_matrix, std_prior=1, **kwargs):
    w_reorg = _reorg_matrix(w_matrix, groups)
    n_units, n_groups, n_tasks = w_reorg.shape
    w_reorg = standardize_wm(w_reorg, flat=True)
    out = compute_prob_cluster(w_reorg, std_prior=std_prior, **kwargs)
    # overlap = np.identity(n_groups)*out
    # return overlap, out
    return out


def make_group_matrix(groups):
    u_groups = np.unique(groups)
    g_mat = np.zeros((len(groups), len(u_groups)))
    for i, g in enumerate(u_groups):
        g_mat[:, i] = groups == g
    return g_mat


def simple_brim(groups, w_matrix, threshold=0.1):
    wm = standardize_wm(w_matrix)
    conn_m = np.abs(wm) > threshold
    null_m = np.ones_like(conn_m) * np.mean(conn_m, axis=1, keepdims=True)
    b_tilde = conn_m - null_m
    t_mat = make_group_matrix(groups)
    t_tilde = np.dot(b_tilde, t_mat)
    inf_groups = np.argmax(t_tilde, axis=1)
    r_mat = make_group_matrix(inf_groups)
    prod_mat = np.dot(r_mat.T, t_tilde)
    out_mod = np.trace(prod_mat) / np.sum(conn_m)
    out_mat = np.identity(len(np.unique(groups))) * out_mod
    return out_mat, out_mod


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


def sample_all_contexts(
    m,
    n_samps=1000,
    use_mean=False,
    ret_out=False,
    from_layer=None,
    cluster_funcs=None,
    layer=None,
):
    if from_layer is None and layer is not None:
        from_layer = layer
    if cluster_funcs is None:
        n_g = m.n_groups
    else:
        n_g = len(cluster_funcs)
    activity = []
    out_act = []
    for i in range(n_g):
        if cluster_funcs is None:
            _, samps_i, reps_i = m.sample_reps(n_samps, context=i, layer=from_layer)
        else:
            true_i, samps_i, reps_i = m.sample_reps(n_samps * n_g, layer=from_layer)
            rel_dim = maux.get_relevant_dims(true_i, m)
            mask = cluster_funcs[i](rel_dim)
            reps_i = reps_i[mask]
            samps_i = samps_i[mask]
        if use_mean:
            reps_i = np.mean(reps_i, axis=0, keepdims=True)
        out_act_i = m.model(samps_i)

        activity.append(reps_i)
        out_act.append(out_act_i)
    if ret_out:
        out = (activity, out_act)
    else:
        out = activity
    return out


def context_separable_only_prob(T, L):
    ts = np.arange(2, T + 1)
    prob_list = (
        ss.binom(T, ts) * ((1 / (2 * L)) ** ts) * ((1 - (1 / (2 * L))) ** (T - ts))
    )
    diff_list = 1 - 1 / L ** (ts - 1)
    p = np.sum(prob_list * diff_list)
    return p


def context_separable_only_prob_est(T, L, fdg, C=2, n_samps=100):
    n_splits = np.zeros(n_samps)
    for i in range(n_samps):
        out_m, out_h = ms.train_modularizer(
            fdg,
            train_epochs=0,
            tasks_per_group=T,
            group_size=L,
            n_groups=C,
            single_output=True,
            integrate_context=True,
            n_overlap=L,
        )

        out_funcs = decompose_model_tasks(out_m)
        n = len(list(filter(lambda x: len(x) == 1, out_funcs.keys())))
        n_splits[i] = n - C
    return n_splits


def _fit_optimal_clusters(
    act,
    max_components,
    model=skmx.GaussianMixture,
    use_init=False,
    demean=True,
    ret_scores=False,
):
    if demean:
        act = act - np.mean(act, axis=0, keepdims=True)
    n_components = range(1, max_components + 1)
    scores = np.zeros(len(n_components))
    models = np.zeros_like(scores, dtype=object)
    labels = np.zeros((len(n_components), act.shape[1]))
    for i, n_comp in enumerate(n_components):
        if use_init and n_comp > 1:
            means_init = np.identity(n_comp)[:, : n_comp - 1]
        else:
            means_init = None
        m = model(n_comp, means_init=means_init)
        labels[i] = m.fit_predict(act.T)
        models[i] = m
        scores[i] = m.bic(act.T)
    ind = np.argmin(scores)
    m = models[ind]
    labels = labels[ind]
    out = (m, labels)
    if ret_scores:
        out = (m, labels, scores)
    return out


def _fit_clusters(
    act, n_components, model=skmx.GaussianMixture, use_init=False, demean=True
):
    if use_init and n_components > 1:
        means_init = np.identity(n_components)[:, : n_components - 1]
    else:
        means_init = None
    if demean:
        act = act - np.mean(act, axis=0, keepdims=True)
    m = model(n_components, means_init=means_init)
    labels = m.fit_predict(act.T)
    return m, labels


def _sort_ablate_inds(losses):
    if losses.shape[0] > losses.shape[1]:
        losses = np.concatenate((losses, np.zeros((losses.shape[0], 1))), axis=1)
    losses[np.isnan(losses)] = 0
    _, sort_inds = spo.linear_sum_assignment(losses.T, maximize=True)
    return sort_inds


def _kfunc(x):
    x[x > 1] = 1
    x[x < -1] = -1
    main = np.sqrt(1 - x**2) + (np.pi - np.arccos(x)) * x
    return main / (2 * np.pi)


def _compute_kernel(x):
    """x is N x D array of stimulus representations"""
    xw = np.sqrt(np.sum(x**2, axis=1, keepdims=True))
    kernel = (xw**2) * _kfunc((x / xw) @ (x / xw).T)
    return kernel


def kernel_theory(fdg, tasks=None, n_cons=2, weight_var=1):
    if tasks is None:
        tasks = ms.make_non_overlapping_contextual_task_func(1, 1, n_cons)
    stim, inp_rep = fdg.get_all_stim(con_dims=np.arange(-n_cons, 0))
    targ = 2 * (tasks(stim) - 0.5)
    kernel = weight_var * _compute_kernel(inp_rep)
    v = kernel @ targ
    return v.T @ v


def ablate_label_sets(
    m,
    unit_labels,
    n_samps=5000,
    separate_contexts=True,
    n_shuffs=10,
    ret_null=False,
    eps=0.01,
    layer=None,
    cluster_funcs=None,
):
    rng = np.random.default_rng()
    clusters = np.unique(unit_labels)
    if separate_contexts:
        n_contexts = m.n_groups
        con_list = np.arange(n_contexts)
    else:
        n_contexts = 1
        con_list = (None,)
    if len(clusters) < n_contexts:
        diff = n_contexts - len(clusters)
        clusters = np.concatenate((clusters, np.arange(-diff, 0)))
    c_losses = np.zeros((len(clusters), n_contexts))
    n_losses = np.zeros((len(clusters), n_contexts, n_shuffs))
    for i, label in enumerate(clusters):
        for j, cl in enumerate(con_list):
            base_loss = m.get_ablated_loss(
                np.zeros_like(unit_labels, dtype=bool),
                n_samps=n_samps,
                group_ind=cl,
                layer=layer,
            )
            c_mask = unit_labels == label
            c_loss = m.get_ablated_loss(
                c_mask, n_samps=n_samps, group_ind=cl, layer=layer
            )
            for k in range(n_shuffs):
                rng.shuffle(c_mask)
                nl = m.get_ablated_loss(
                    c_mask, n_samps=n_samps, group_ind=cl, layer=layer
                )
                n_losses[i, j, k] = nl
            null_loss = np.mean(n_losses[i, j])
            null_loss_range = max(null_loss - base_loss, eps)
            manip_loss_range = max(c_loss - base_loss, eps)
            c_losses[i, j] = np.log(manip_loss_range / null_loss_range)
    sort_inds = _sort_ablate_inds(c_losses)
    c_losses = c_losses[sort_inds]
    n_losses = n_losses[sort_inds]
    if ret_null:
        out = c_losses, n_losses
    else:
        out = c_losses
    return out


def act_cluster(
    m, n_clusters=None, n_samps=1000, use_init=False, cluster_funcs=None, layer=None
):
    activity = sample_all_contexts(
        m, n_samps=n_samps, use_mean=True, from_layer=layer, cluster_funcs=cluster_funcs
    )
    a_full = np.concatenate(activity, axis=0)
    if n_clusters is None:
        if cluster_funcs is not None:
            n_clusters = len(cluster_funcs) + 1
        else:
            n_clusters = len(activity) + 1

    _, hidden_clusters = _fit_clusters(a_full, n_clusters, use_init=use_init)
    return hidden_clusters


def graph_ablation(m, **kwargs):
    return ablation_experiment(m, cluster_method=cluster_graph, **kwargs)


def act_ablation(m, **kwargs):
    return ablation_experiment(m, cluster_method=act_cluster, **kwargs)


def ablation_experiment(
    m,
    n_clusters=None,
    n_samps=1000,
    cluster_method=act_cluster,
    use_init=False,
    single_number=True,
    layer=None,
    cluster_funcs=None,
    **kwargs,
):
    hidden_clusters = cluster_method(
        m,
        n_clusters=n_clusters,
        n_samps=n_samps,
        cluster_funcs=cluster_funcs,
        layer=layer,
    )

    out = ablate_label_sets(
        m,
        hidden_clusters,
        n_samps=n_samps,
        layer=layer,
        cluster_funcs=cluster_funcs,
        **kwargs,
    )
    return out


def across_ablation_experiment(*args, **kwargs):
    cl = ablation_experiment(*args, n_shuffs=20, **kwargs)

    cols = cl.shape[1]
    diff = cl.shape[0] - cl.shape[1]
    mask_off = ~np.identity(cl.shape[1], dtype=bool)
    add_row = np.zeros((diff, cols), dtype=bool)
    mask_off = np.concatenate((mask_off, add_row), axis=0)
    out = np.mean(cl[mask_off])
    return out


def within_ablation_experiment(*args, **kwargs):
    cl = ablation_experiment(*args, **kwargs)
    cols = cl.shape[1]
    diff = cl.shape[0] - cl.shape[1]
    mask = np.identity(cl.shape[1], dtype=bool)
    mask = np.concatenate((mask, np.zeros((diff, cols), dtype=bool)), axis=0)

    out = np.mean(cl[mask])
    return out


def diff_ablation_experiment(*args, **kwargs):
    cl = ablation_experiment(*args, **kwargs)
    cols = cl.shape[1]
    diff = cl.shape[0] - cl.shape[1]
    mask_on = np.identity(cl.shape[1], dtype=bool)
    mask_off = ~mask_on
    mask_on = np.concatenate((mask_on, np.zeros((diff, cols), dtype=bool)), axis=0)
    mask_off = np.concatenate((mask_off, np.zeros((diff, cols), dtype=bool)), axis=0)

    out = np.mean(cl[mask_on]) - np.mean(cl[mask_off])
    return out


def within_max_corr_ablation(*args, **kwargs):
    return within_ablation_experiment(*args, cluster_method=cluster_max_corr, **kwargs)


def across_max_corr_ablation(*args, **kwargs):
    return across_ablation_experiment(*args, cluster_method=cluster_max_corr, **kwargs)


def diff_max_corr_ablation(*args, **kwargs):
    return diff_ablation_experiment(*args, cluster_method=cluster_max_corr, **kwargs)


def within_graph_ablation(*args, **kwargs):
    return within_ablation_experiment(*args, cluster_method=cluster_graph, **kwargs)


def within_act_ablation(*args, **kwargs):
    return within_ablation_experiment(*args, cluster_method=act_cluster, **kwargs)


def diff_act_ablation(*args, **kwargs):
    return diff_ablation_experiment(*args, cluster_method=act_cluster, **kwargs)


def across_graph_ablation(*args, **kwargs):
    return across_ablation_experiment(*args, cluster_method=cluster_graph, **kwargs)


def diff_graph_ablation(*args, **kwargs):
    return diff_ablation_experiment(*args, cluster_method=cluster_graph, **kwargs)


def across_act_ablation(*args, **kwargs):
    return across_ablation_experiment(*args, cluster_method=act_cluster, **kwargs)


def _quantify_model_ln(m, n, n_samps=1000, **kwargs):
    activity = sample_all_contexts(m, n_samps=n_samps, **kwargs)
    act_all = np.concatenate(activity, axis=0)
    norm = np.mean(np.sum(np.abs(act_all) ** n, axis=1))
    return norm


def quantify_model_l1(m, **kwargs):
    return _quantify_model_ln(m, 1, **kwargs)


def quantify_model_l2(m, **kwargs):
    return _quantify_model_ln(m, 2, **kwargs)


def quantify_activity_clusters(
    m, n_samps=1000, use_mean=True, layer=None, model=skmx.GaussianMixture
):
    activity = sample_all_contexts(m, n_samps=n_samps, use_mean=use_mean, layer=layer)
    a_full = np.concatenate(activity, axis=0)

    m_full_p1, _ = _fit_clusters(a_full, len(activity) + 1, model=model)
    m_full, _ = _fit_clusters(a_full, len(activity), model=model)
    m_one, _ = _fit_clusters(a_full, 1, model=model)
    fp1_score = m_full_p1.score(a_full.T)
    f_score = m_full.score(a_full.T)
    o_score = m_one.score(a_full.T)
    return max(f_score, fp1_score) - o_score


def task_performance_learned(
    model,
    use_model_rep=False,
    n_samps=10000,
    n_folds=10,
    classifier_model=skm.LinearSVC,
    pca=0.95,
    post_norm=False,
    **kwargs,
):
    pipe = na.make_model_pipeline(
        classifier_model, pca=pca, post_norm=post_norm, dual="auto", **kwargs
    )
    stim_rep, stim, targ = model.get_x_true(n_train=n_samps)
    if use_model_rep:
        stim_rep = model.get_representation(stim_rep)
    stim_rep = np.array(stim_rep)

    perf = np.zeros((targ.shape[1], n_folds))
    for i in range(targ.shape[1]):
        out = skms.cross_validate(pipe, stim_rep, targ[:, i], cv=n_folds)
        perf[i] = out["test_score"]
    return perf


def quantify_max_corr_clusters(m, n_samps=5000):
    clusters, overlap = cluster_max_corr(m, n_samps=n_samps, ret_overlap=True)
    return 1 - overlap


def apply_act_clusters_list(
    models, func=quantify_activity_clusters, eval_layers=False, **kwargs
):
    if eval_layers:
        n_layers = (len(models.flatten()[0].model.layers) - 1,)
    else:
        n_layers = ()
    clust = np.zeros(models.shape + n_layers)
    for ind in u.make_array_ind_iterator(clust.shape):
        if eval_layers:
            layer = ind[-1]
            model_ind = ind[:-1]
        else:
            model_ind = ind
            layer = None
        clust[ind] = func(models[model_ind], layer=layer, **kwargs)
    return clust


def new_task_training(
    fdg,
    params=None,
    verbose=False,
    n_tasks=10,
    novel_tasks=1,
    train_epochs=10,
    train_samps=5000,
    new_samps=1000,
    **kwargs,
):
    all_tasks = set(range(n_tasks))
    nov_task = set(range(novel_tasks))
    pretrain_tasks = all_tasks.difference(nov_task)
    kwargs["single_output"] = True

    out_two = ms.train_modularizer(
        fdg,
        params=params,
        verbose=verbose,
        train_epochs=len(pretrain_tasks) * train_epochs,
        n_train=len(pretrain_tasks) * train_samps,
        tasks_per_group=n_tasks,
        only_tasks=pretrain_tasks,
        **kwargs,
    )
    out_dict = {}
    out_dict["initial_hist"] = out_two[1]
    out_dict["initial_modularity"] = compute_frac_contextual(out_two[0])
    out_dict["initial_subspace"] = compute_alignment_index(out_two[0])

    h_next = out_two[0].fit(
        track_dimensionality=True,
        epochs=train_epochs,
        n_train=new_samps * len(nov_task),
        verbose=False,
        val_only_tasks=nov_task,
        track_mean_tasks=False,
    )
    out_dict["trained_hist"] = h_next
    out_dict["trained_modularity"] = compute_frac_contextual(out_two[0])
    out_dict["trained_subspace"] = compute_alignment_index(out_two[0])

    print("single task model")
    out_one = ms.train_modularizer(
        fdg,
        params=params,
        verbose=verbose,
        train_epochs=train_epochs,
        n_train=new_samps * len(nov_task),
        tasks_per_group=n_tasks,
        only_tasks=nov_task,
        track_mean_tasks=False,
        **kwargs,
    )
    out_dict["null_hist"] = out_one[1]
    out_dict["null_modularity"] = compute_frac_contextual(out_one[0])
    out_dict["null_subspace"] = compute_alignment_index(out_one[0])
    return out_dict


def new_related_context_training(
    *args,
    total_groups=3,
    novel_groups=1,
    **kwargs,
):
    ng = range(total_groups - novel_groups, total_groups)
    tg = range(total_groups - novel_groups)
    share_pairs = list(zip(ng, tg))
    return new_context_training(
        *args,
        total_groups=total_groups,
        novel_groups=novel_groups,
        share_pairs=share_pairs,
        **kwargs,
    )


def train_separate_models(
    mixing_strength,
    n_group=3,
    dg_pwr=1,
    n_train=300,
    **kwargs,
):
    _, model_null, hist_null = train_controlled_model(
        n_group,
        mixing_strength,
        total_power=dg_pwr,
        n_train=n_train,
        **kwargs,
    )
    _, model_lin, hist_lin = train_controlled_model(
        n_group,
        0,
        total_power=1 - mixing_strength,
        n_train=n_train,
        **kwargs,
    )
    _, model_nl, hist_nl = train_controlled_model(
        n_group,
        1,
        total_power=mixing_strength,
        n_train=n_train,
        **kwargs,
    )
    out = {
        "null": (model_null, hist_null),
        "lin": (model_lin, hist_lin),
        "nonlin": (model_nl, hist_nl),
    }
    return out


def train_controlled_model(
    n_group,
    mixing_strength,
    n_cons=2,
    n_units=500,
    n_feats=None,
    train_epochs=20,
    n_train=5000,
    tasks_per_group=20,
    n_overlap=None,
    group_width=50,
    total_power=None,
    irrel_vars=2,
    **kwargs,
):
    if n_overlap is None:
        n_overlap = n_group
    if n_feats is None:
        n_feats = n_group * n_cons - n_overlap * (n_cons - 1) + irrel_vars + n_cons
    mddg = dg.MixedDiscreteDataGenerator(
        n_feats,
        mix_strength=mixing_strength,
        n_units=n_units,
        total_power=total_power,
    )
    model, hist = ms.train_modularizer(
        mddg,
        group_size=n_group,
        n_groups=n_cons,
        n_overlap=n_overlap,
        train_epochs=train_epochs,
        n_train=n_train,
        group_width=group_width,
        single_output=True,
        integrate_context=True,
        tasks_per_group=tasks_per_group,
        **kwargs,
    )
    return mddg, model, hist


def analyze_model_orders(model, **kwargs):
    _, stim, _ = model.get_x_true(n_train=2)
    orders = np.arange(1, stim.shape[1] + 1)

    for i, order in enumerate(orders):
        out = analyze_model_order(model, order, **kwargs)
        out_models, out_scores = out
        if i == 0:
            score_dict = {k: np.zeros(len(orders)) for k in out_scores.keys()}
            model_dict = {
                k: np.zeros(len(orders), dtype=object) for k in out_scores.keys()
            }
        for k in out_models.keys():
            model_dict[k][i] = out_models[k]
        for k in out_scores.keys():
            score_dict[k][i] = out_scores[k]
    return orders, model_dict, score_dict


def order_dim_cij(k, o, n=2):
    all_avgs = np.zeros(o - 1)
    no = n**o
    n2o = n ** (2 * o)
    for r in range(1, o):
        a = ss.binom(o, r) * ss.binom(k - o, o - r)
        nr = n**r
        r_out = a * (nr - 1) / (n2o * no)

        all_avgs[r - 1] = r_out
    out = sum(all_avgs) + (no - 1) * (1 / n2o) ** 2
    return out


def order_dim(k, o, n=2):
    no = n**o
    big_n = ss.binom(k, o) * no

    c2_ij_sum = order_dim_cij(k, o, n=n)
    c2_ij_mu = c2_ij_sum / (big_n - 1)

    c2_ii = (1 / (no**2)) * (1 - 1 / no) ** 2
    pr = big_n * c2_ii / (c2_ii + (big_n - 1) * c2_ij_mu)
    return pr


def order2_dim(k, n=2, o=2):
    no = n**o
    nop = n ** (o + 1)
    no2p = n ** (2 * o + 1)
    big_n = ss.binom(k, o) * no
    o_terms = 2 * (ss.binom(k - 1, o - 1) - 1)
    print(o_terms)
    o_terms = ss.binom(o, 1) * ss.binom(k - o, o - 1)
    print(o_terms)

    a = (o_terms * (no - 2) + no - 1) * (1 / no) ** 4
    b = n * o_terms * ((1 / nop) - 4 / no2p + (1 / no) ** 2) ** 2
    c2_ij_mu = (a + b) / (big_n - 1)
    c2_ii = (1 / no * (1 - 1 / no) ** 2 + (1 - 1 / no) * 1 / no**2) ** 2
    pr = big_n * c2_ii / (c2_ii + (big_n - 1) * c2_ij_mu)
    return pr


def order_regression(order, n_feats, n_vals=2, min_order=1, single_order=False):
    if single_order:
        min_order = order
    vals = np.array(list(it.product(np.arange(n_vals), repeat=n_feats)))

    pf = skp.PolynomialFeatures(
        degree=(min_order, order), interaction_only=True, include_bias=False
    )
    steps = (skp.OneHotEncoder(), pf, skfs.VarianceThreshold())
    pipe = sklpipe.make_pipeline(*steps)
    pipe.fit(vals)
    betas = pipe.transform(vals).todense()
    return pipe, betas


def explain_order_regression(
    samps,
    targs,
    order,
    lm_model=sklm.Ridge,
    **kwargs,
):
    n_feats = samps.shape[1]
    pipe, _ = order_regression(order, n_feats, **kwargs)

    betas = np.asarray(pipe.transform(samps).todense())
    m = lm_model()
    m.fit(betas, targs)
    score = m.score(betas, targs)
    return m, score


def compute_order_spectrum(
    stim,
    rep,
    max_order,
    lm_model=sklm.Ridge,
    **kwargs,
):
    n_feats = stim.shape[1]
    spectrum = np.zeros(max_order)
    for i in range(1, max_order + 1):
        if i > 1:
            lo_pipe, _ = order_regression(i - 1, n_feats, min_order=1)
            lo_coeffs = lo_pipe.transform(stim)
            m_low = lm_model()
            m_low.fit(lo_coeffs, rep)
            o_guess = m_low.predict(lo_coeffs)
            remaining = 1 - m_low.score(lo_coeffs, rep)
            residual = rep - o_guess
        else:
            residual = rep
            remaining = 1
        targ_pipe, _ = order_regression(i, n_feats, single_order=True)
        targ_coeffs = targ_pipe.transform(stim)
        m_targ = lm_model()
        m_targ.fit(targ_coeffs, residual)
        spectrum[i - 1] = m_targ.score(targ_coeffs, residual) * remaining
    return spectrum


def compute_model_spectrum(
    model,
    max_order=None,
    subset=True,
    layer=None,
    n_samps=1000,
    history=None,
    **kwargs,
):
    inp_rep, stim, targ = model.get_x_true(n_train=n_samps)
    rep = model.get_representation(inp_rep, layer=layer)
    if subset:
        gs = model.groups
        group_inds = np.unique(np.concatenate(gs))
        n_cons = np.arange(-len(gs), 0)
        inds = np.concatenate((group_inds, n_cons))
        stim = stim[:, inds]
    if max_order is None:
        max_order = stim.shape[1]
    ir_spec = compute_order_spectrum(stim, inp_rep, max_order, **kwargs)
    rep_spec = compute_order_spectrum(stim, rep, max_order, **kwargs)
    task_spec = compute_order_spectrum(stim, targ, max_order, **kwargs)
    out_dict = {
        "input": ir_spec,
        "model_rep": rep_spec,
        "task": task_spec,
    }
    if history is not None and history.history.get("tracked_activity") is not None:
        stim, _, rep_dynamic = history.history.get("tracked_activity")
        dynamic_spectrum = np.zeros((len(rep_dynamic), max_order))
        for i, rd_i in enumerate(rep_dynamic):
            dynamic_spectrum[i] = compute_order_spectrum(
                stim,
                rd_i,
                max_order,
                **kwargs,
            )
        out_dict["model_rep_dynamic"] = dynamic_spectrum
    return out_dict


def prob_elim(d, n_reps=10000):
    # ds = np.arange(1, d + 1)
    # ds_all = np.concatenate((ds, -ds))
    # rng = np.random.default_rng()
    # ds_chosen = rng.choice(ds_all, size=(n_reps, 2))
    # chosen = np.any(np.abs(ds_chosen) == 1, axis=1)
    # other_twice = ds_chosen[:, 0] == ds_chosen[:, 1]

    # safe = np.mean(np.logical_or(chosen, other_twice))
    safe_theor = 1 - ((d - 1) / d) ** 2 + (1 / 2) * (d - 1) * (1 / d**2)
    return 1 - safe_theor


def prob_all_elim(d, t):
    p = prob_elim(d)
    return 1 - sts.binom.cdf(d - 1, t, p)


def analyze_model_order(model, order, subset=True, layer=None, n_samps=1000, **kwargs):
    inp_rep, stim, targ = model.get_x_true(n_train=n_samps)
    rep = model.get_representation(inp_rep, layer=layer)
    if subset:
        gs = model.groups
        group_inds = np.unique(np.concatenate(gs))
        n_cons = np.arange(-len(gs), 0)
        inds = np.concatenate((group_inds, n_cons))
        stim = stim[:, inds]
    ir_model, ir_score = explain_order_regression(stim, inp_rep, order, **kwargs)
    rep_model, rep_score = explain_order_regression(stim, rep, order, **kwargs)
    task_model, task_score = explain_order_regression(stim, targ, order, **kwargs)
    out_scores = {
        "input": ir_score,
        "model_rep": rep_score,
        "task": task_score,
    }
    out_models = {
        "input": ir_model,
        "model_rep": rep_model,
        "task": task_model,
    }
    return out_models, out_scores


def task_order_regression(
    order,
    n_tasks,
    group_size,
    n_cons=2,
    n_samples=1000,
    axis_tasks=False,
    **kwargs,
):
    n_feats = group_size + n_cons

    source = u.MultiBernoulli(0.5, n_feats - 1)
    samps = source.rvs(n_samples)
    samps = np.concatenate((samps, np.expand_dims(1 - samps[:, -1], 1)), axis=1)

    task_func = ms.make_contextual_task_func(
        group_size, n_tasks=n_tasks, renorm=False, axis_tasks=axis_tasks
    )
    targs = task_func(samps)

    out = explain_order_regression(
        samps,
        targs,
        order,
        **kwargs,
    )
    return out


def rep_order_regression(model, order, n_samples=1000, **kwargs):
    samps, reps = model.sample_reps(n_samples)
    return explain_order_regression(samps, reps, order, **kwargs)


def zero_shot_training(
    fdg,
    params=None,
    verbose=True,
    total_groups=2,
    n_tasks=10,
    train_epochs=10,
    train_samps=5000,
    test_samps=1000,
    fix_n_irrel_vars=1,
    new_samps=0,
    **kwargs,
):
    kwargs["single_output"] = True
    out = ms.train_modularizer(
        fdg,
        verbose=verbose,
        params=params,
        n_groups=total_groups,
        train_epochs=train_epochs,
        n_train=train_samps,
        tasks_per_group=n_tasks,
        track_mean_tasks=False,
        fix_n_irrel_vars=fix_n_irrel_vars,
        fix_irrel_value=0,
        **kwargs,
    )
    model, hist = out

    irrel = model.irrel_vars[:fix_n_irrel_vars]
    irep, true, targ = model.get_x_true(n_train=test_samps, fix_vars=irrel, fix_value=1)
    resps = model.out_model(model.get_representation(irep))
    errs_ood = (resps > 0.5) == (targ > 0.5)

    irrel = model.irrel_vars[:fix_n_irrel_vars]
    irep, true, targ = model.get_x_true(n_train=test_samps, fix_vars=irrel, fix_value=0)
    resps = model.out_model(model.get_representation(irep))
    errs_ind = (resps > 0.5) == (targ > 0.5)

    return errs_ood, errs_ind, (model, hist)


def new_context_training(
    fdg,
    params=None,
    verbose=True,
    total_groups=3,
    novel_groups=1,
    n_tasks=10,
    train_epochs=10,
    train_samps=5000,
    new_samps=1000,
    untrained_tasks=0,
    separate_untrained=False,
    track_reps=False,
    **kwargs,
):
    all_tasks = set(range(n_tasks))
    untrained_task = set(range(untrained_tasks))
    train_tasks = all_tasks.difference(untrained_task)
    if separate_untrained:
        separate_tasks = untrained_task
    else:
        separate_tasks = None

    all_groups = list(range(total_groups))
    kwargs["single_output"] = True
    # print(
    #     "arguments",
    #     fdg,
    #     verbose,
    #     params,
    #     total_groups,
    #     all_groups[:-novel_groups],
    #     (total_groups - novel_groups) * train_epochs,
    #     (total_groups - novel_groups) * train_samps,
    #     n_tasks,
    #     separate_tasks,
    # )
    # print(kwargs)
    out_two = ms.train_modularizer(
        fdg,
        verbose=verbose,
        params=params,
        n_groups=total_groups,
        only_groups=all_groups[:-novel_groups],
        train_epochs=(total_groups - novel_groups) * train_epochs,
        n_train=(total_groups - novel_groups) * train_samps,
        tasks_per_group=n_tasks,
        separate_tasks=separate_tasks,
        track_mean_tasks=False,
        track_reps=track_reps,
        **kwargs,
    )
    con_inds = all_groups[:-novel_groups]
    out_dict = {}
    out_dict["initial_hist"] = out_two[1]
    out_dict["initial_modularity"] = compute_frac_contextual(
        out_two[0],
        con_inds=con_inds,
    )
    out_dict["initial_subspace"] = compute_alignment_index(
        out_two[0],
        con_inds=con_inds,
    )
    if untrained_tasks > 0:
        only_tasks = train_tasks
        val_only_tasks = untrained_task
    else:
        only_tasks = None
        val_only_tasks = None
    h_next = out_two[0].fit(
        track_dimensionality=True,
        epochs=train_epochs,
        n_train=new_samps * novel_groups,
        verbose=False,
        only_groups=all_groups[-novel_groups:],
        val_only_groups=all_groups[-novel_groups:],
        only_tasks=only_tasks,
        track_mean_tasks=False,
        val_only_tasks=val_only_tasks,
        track_reps=track_reps,
    )
    out_dict["trained_hist"] = h_next
    out_dict["trained_modularity"] = compute_frac_contextual(out_two[0])
    out_dict["trained_subspace"] = compute_alignment_index(out_two[0])

    out_one = ms.train_modularizer(
        fdg,
        verbose=verbose,
        params=params,
        n_groups=total_groups,
        only_groups=all_groups[:novel_groups],
        train_epochs=train_epochs,
        n_train=new_samps * novel_groups,
        tasks_per_group=n_tasks,
        only_tasks=only_tasks,
        val_only_tasks=val_only_tasks,
        track_mean_tasks=False,
        **kwargs,
    )
    out_dict["seq_model"] = out_two[0]
    out_dict["null_model"] = out_one[0]
    out_dict["null_hist"] = out_one[1]
    out_dict["null_modularity"] = compute_frac_contextual(out_one[0])
    out_dict["null_subspace"] = compute_alignment_index(out_one[0])
    return out_dict


def infer_optimal_activity_clusters(
    m,
    n_samps=1000,
    use_mean=True,
    ret_act=False,
    model=skmx.GaussianMixture,
    from_layer=None,
    order=True,
):
    activity = sample_all_contexts(
        m, n_samps=n_samps, use_mean=use_mean, from_layer=from_layer
    )
    act_full = np.concatenate(activity, axis=0)
    out = infer_optimal_clusters_from_mean_activity(act_full, order=order)
    if ret_act:
        out = (out, act_full.T)
    return out


def _make_mean_activity(stim, activity, n_contexts=2):
    con_ind = np.argmax(stim[:, -n_contexts:], axis=1)
    u_cons = np.unique(con_ind)
    out_mu = []
    for uc in u_cons:
        mu_uc = np.mean(activity[con_ind == uc], axis=0, keepdims=True)
        out_mu.append(mu_uc)
    out_mu = np.concatenate(out_mu, axis=0)
    return out_mu


def infer_optimal_clusters_from_activity(
    stim, activity, n_contexts=2, order=True, ret_act=False
):
    act_full = _make_mean_activity(stim, activity, n_contexts=n_contexts)
    out = infer_optimal_clusters_from_mean_activity(act_full, order=order)
    if ret_act:
        out = (out, act_full.T)
    return out


def infer_optimal_clusters_from_mean_activity(activity, order=True):
    _, out = _fit_optimal_clusters(activity, len(activity) + 1)
    if order:
        u_clust = np.unique(out)
        m_diff = activity[0] - activity[1]
        diffs = np.zeros(len(u_clust))
        for i, uc in enumerate(u_clust):
            diffs[i] = np.mean(m_diff[uc == out])
        inds = np.argsort(diffs)
        ranks = np.argsort(inds)
        new_out = np.zeros_like(out)
        for i, uc in enumerate(u_clust):
            new_out[out == uc] = ranks[i]
        out = new_out
    return out


def infer_activity_clusters(
    m,
    n_samps=1000,
    use_mean=True,
    ret_act=False,
    model=skmx.GaussianMixture,
    from_layer=None,
    order=True,
):
    activity = sample_all_contexts(
        m, n_samps=n_samps, use_mean=use_mean, from_layer=from_layer
    )
    act_full = np.concatenate(activity, axis=0)
    _, out = _fit_clusters(act_full, len(activity) + 1)
    if order:
        u_clust = np.unique(out)
        m_diff = np.mean(activity[0] - activity[1], axis=0)
        diffs = np.zeros(len(u_clust))
        for i, uc in enumerate(u_clust):
            diffs[i] = np.mean(m_diff[uc == out])
        inds = np.argsort(diffs)
        ranks = np.argsort(inds)
        new_out = np.zeros_like(out)
        for i, uc in enumerate(u_clust):
            new_out[out == uc] = ranks[i]
        out = new_out
    if ret_act:
        out = (out, act_full.T)
    return out


def apply_geometry_model_list(
    ml,
    fdg,
    group_ind=0,
    n_train=4,
    fix_features=2,
    noise_cov=0.01**2,
    eval_layers=False,
    **kwargs,
):
    ml = np.array(ml)
    if not u.check_list(noise_cov):
        noise_cov = (noise_cov,)
    if eval_layers:
        n_layers = (len(ml.flatten()[0].model.layers) - 1,)
    else:
        n_layers = ()
    shattering = np.zeros(ml.shape + (len(noise_cov),) + n_layers, dtype=object)
    within_ccgp = np.zeros_like(shattering)
    across_ccgp = np.zeros_like(shattering)
    for ind in u.make_array_ind_iterator(shattering.shape):
        if eval_layers:
            model_ind = ind[:-2]
            layer = ind[-1]
            noise_ind = ind[-2]
        else:
            model_ind = ind[:-1]
            noise_ind = ind[-1]
            layer = None
        m = ml[model_ind]
        m_code = ModularizerCode(
            m,
            dg_model=fdg,
            group_ind=group_ind,
            noise_cov=noise_cov[noise_ind],
            use_layer=layer,
        )
        within_ccgp[ind] = m_code.compute_within_group_ccgp(
            n_train=n_train, fix_features=fix_features, **kwargs
        )
        across_ccgp[ind] = m_code.compute_across_group_ccgp(
            fix_features=fix_features, n_train=n_train, **kwargs
        )
        shattering[ind] = m_code.compute_shattering(**kwargs)[-1]
    return shattering, within_ccgp, across_ccgp


def apply_clusters_model_list(ml, func=quantify_clusters, **kwargs):
    ml = np.array(ml)
    mats = np.zeros_like(ml, dtype=object)
    diffs = np.zeros_like(mats)
    for ind in u.make_array_ind_iterator(ml.shape):
        m = ml[ind]
        mat, diff = func(m.out_group_labels, m.model.weights[-2], **kwargs)
        mats[ind] = mat
        diffs[ind] = diff
    return mats, diffs


def process_histories(
    hs, n_epochs, keep_keys=("loss", "val_loss", "dimensionality", "corr_rate")
):
    hs = np.array(hs)
    ind = (0,) * len(hs.shape)
    # n_epochs = hs[ind].params['epochs']
    out_dict = {}
    for key in keep_keys:
        out_dict[key] = np.zeros(hs.shape + (n_epochs + 1,))
        out_dict[key][:] = np.nan
    for ind in u.make_array_ind_iterator(hs.shape):
        for i, key in enumerate(keep_keys):
            quant = hs[ind].history[key]
            ind_epochs = len(quant)
            out_dict[key][ind][:ind_epochs] = quant
    return out_dict


def linearly_separable_condition(stim, targs, mask_dims, thresh=0.90, **kwargs):
    scores = np.zeros(targs.shape[1])
    for j in range(targs.shape[1]):
        if len(np.unique(targs[:, j])) > 1 and sum(mask_dims) > 0:
            m = skm.LinearSVC(dual="auto", **kwargs)
            m.fit(stim[:, mask_dims], targs[:, j])
            scores[j] = m.score(stim[:, mask_dims], targs[:, j])
        else:
            scores[j] = 1
    condition = np.all(scores > thresh)
    return condition


def single_class_condition(stim, targs, mask_dims):
    lens = np.zeros(targs.shape[1])
    for j in range(targs.shape[1]):
        lens[j] = len(np.unique(targs[:, j]))
    condition = np.all(lens == 1)
    return condition


def _dichotomy_generator(stim, mask, thr=0.99):
    stim = stim[:, mask]
    dichots = it.product((True, False), repeat=len(stim))
    for i, dichot in enumerate(dichots):
        d_mask = np.array(dichot)
        if np.all(d_mask) or np.all(~d_mask):
            pass
        else:
            m = skm.LinearSVC(dual="auto")
            m.fit(stim, d_mask)
            sep = m.score(stim, d_mask)
            uv = u.make_unit_vector(m.coef_)
            inter = m.intercept_
            if sep > thr:
                yield ((tuple(uv), inter[0]), d_mask)


def _ind_generator(stim, mask, ref=None):
    stim = stim[:, mask]
    inds = np.where(mask)[0]
    if ref is None:
        ref = np.unique(stim)[0]
    for i in range(stim.shape[1]):
        yield (inds[i], stim[:, i] == ref)


def _decompose_task_object_helper(
    stim,
    targs,
    mask_dims,
    condition_func=linearly_separable_condition,
    depth=0,
    max_depth=None,
    mask_generator=_ind_generator,
    **kwargs,
):
    condition = condition_func(stim, targs, mask_dims)
    if condition:
        out = None
    else:
        out = {}
        gen = mask_generator(stim, mask_dims)
        for i, z_mask in gen:
            o_mask = np.logical_not(z_mask)

            md_z = np.var(stim[z_mask], axis=0) > 0
            md_o = np.var(stim[o_mask], axis=0) > 0

            if max_depth is not None and depth > max_depth:
                out0 = False
                out1 = False
            else:
                out0 = _decompose_task_object_helper(
                    stim[z_mask],
                    targs[z_mask],
                    md_z,
                    depth=depth + 1,
                    max_depth=max_depth,
                    condition_func=condition_func,
                    **kwargs,
                )

                out1 = _decompose_task_object_helper(
                    stim[o_mask],
                    targs[o_mask],
                    md_o,
                    depth=depth + 1,
                    max_depth=max_depth,
                    condition_func=condition_func,
                    **kwargs,
                )
            out[i] = (out0, out1)
    return out


def _mask_func(x, k=None, i=None, comp_func=None):
    mask = x[:, k] == i
    if comp_func is not None:
        mask = np.logical_and(mask, comp_func(x))
    return mask


@u.arg_list_decorator
def estimate_split_probability(n_latents, n_tasks, n_reps=100, **kwargs):
    outs = np.zeros((n_reps, len(n_latents), len(n_tasks)))
    for r, i, j in u.make_array_ind_iterator(outs.shape):
        nl, nt = n_latents[i], n_tasks[j]
        splits = compute_split(nl, nt, **kwargs)
        outs[r, i, j] = len(splits) > 0
    return outs


@u.arg_list_decorator
def split_prob(l_, p, rand_sample=True, n_samps=1000):
    l_ = np.array(l_)
    p = np.array(p)
    out = np.zeros((len(l_), len(p)))
    rng = np.random.default_rng()
    for i, j in u.make_array_ind_iterator(out.shape):
        l_ij = l_[i]
        p_ij = p[j]
        if rand_sample:
            elims = [0, l_ij - 1, l_ij - 2]
            ps = [1 / (2 * l_ij), 1 / (2 * l_ij), 1 - 1 / l_ij]
            s = rng.choice(elims, size=(n_samps, p_ij), p=ps)
            s = np.sum(s, axis=1, keepdims=True)
        else:
            s = p_ij * ((l_ij - 1) / (2 * l_ij) + (l_ij - 2) * (1 - 1 / l_ij))
            # s = np.round(s).astype(int)

        k = np.expand_dims(np.arange(l_ij + 1), 0)
        prob = ((-1) ** k) * ss.binom(l_ij, k) * (1 - k / l_ij) ** s
        if j == 0:
            print(l_ij, p_ij, np.mean(s))
            print(np.mean(np.sum(prob, axis=1)))
        prob = np.mean(np.sum(prob, axis=1))
        out[i, j] = 1 - prob
    out[out > 1] = 1
    out[out < 0] = 0
    return out


def split_prob2(l_, p):
    l_ = np.array(l_)
    p = np.array(p)
    out = np.zeros((len(l_), len(p)))
    for i, j in u.make_array_ind_iterator(out.shape):
        l_ij = l_[i]
        p_ij = p[j]

        p_l2 = 1 - 1 / l_ij
        p_l1 = 1 / (2 * l_ij)

        k = np.arange(0, p_ij + 1)
        choice = ss.binom(p_ij, k)
        prob_choice = choice * (p_l1) ** (p_ij - k) * (p_l1 + p_l2) ** (k)
        comb = (
            p_l1 * p_l1 * binom_func(l_ij, 1, 1)
            + 2 * p_l1 * p_l2 * binom_func(l_ij, 1, 2)
            + p_l2 * p_l2 * binom_func(l_ij, 2, 2)
        )
        n_pair_tasks = ss.comb(k, 2)
        prob = np.sum(prob_choice * (1 - (1 - comb) ** n_pair_tasks))
        out[i, j] = 1 - prob

    out[out > 1] = 1
    out[out < 0] = 0
    return out


def make_target_funcs(
    n_latents,
    n_tasks,
    n_vals=2,
    n_contexts=2,
    renorm_targets=False,
    renorm_stim=False,
    **kwargs,
):
    stim = np.array(list(it.product(range(n_vals), repeat=n_latents)))
    task_funcs = []
    targs = []
    stim_all = []
    for i in range(n_contexts):
        tf = ms.make_linear_task_func(n_latents, n_tasks=n_tasks, **kwargs)
        targs.append(tf(stim))
        task_funcs.append(tf)
        stim_all.append(np.concatenate((stim, np.ones((len(stim), 1)) * i), axis=1))

    stim_all = np.concatenate(stim_all, axis=0)
    targs_all = np.concatenate(targs, axis=0)
    if renorm_targets:
        targs_all = 2 * (targs_all - 0.5)
    if renorm_stim:
        stim_all = 2 * (stim_all - 0.5)
    return stim_all, targs_all, task_funcs


def compute_split(
    n_latents, n_tasks, n_vals=2, n_contexts=2, print_tasks=False, **kwargs
):
    stim_all, targs_all, task_funcs = make_target_funcs(
        n_latents, n_tasks, n_vals=n_vals, n_contexts=n_contexts
    )

    if print_tasks:
        print(task_funcs[0].keywords["task"])
        print(task_funcs[1].keywords["task"])
    return find_top_splits(stim_all, targs_all, **kwargs)


def _ind_extractor(out_tree, dim):
    splits = []
    if out_tree is None:
        splits.append("all")
    else:
        out_tree.pop(dim - 1)
        for k, (v1, v2) in out_tree.items():
            if (v1 is None) and (v2 is None):
                splits.append(k)
    return splits


def _dichotomy_extractor(out_tree, dim):
    splits = []
    if out_tree is None:
        splits.append("all")
    else:
        for (uv, inter), (v1, v2) in out_tree.items():
            non_con = np.argmax(uv) != uv[dim - 1]
            if non_con and (v1 is None) and (v2 is None):
                splits.append((uv, inter))
    return splits


def find_top_splits(stim_all, targs_all, use_all_dichotomies=False, **kwargs):
    if use_all_dichotomies:
        mask_generator = _dichotomy_generator
        extractor = _dichotomy_extractor
    else:
        mask_generator = _ind_generator
        extractor = _ind_extractor
    out_tree = decompose_task_object(
        stim_all, targs_all, mask_generator=mask_generator, **kwargs
    )

    splits = extractor(out_tree, stim_all.shape[1])
    return splits


def decompose_model_tasks(model, n_samps=1000, ret_tree=False, **kwargs):
    _, stim, targs = model.get_x_true(n_train=n_samps)
    rel_stim = maux.get_relevant_dims(stim, model)

    out_tree = decompose_task_object(rel_stim, targs, **kwargs)
    if out_tree is None:
        out_funcs = {(): lambda x: np.ones(x.shape[0], dtype=bool)}
    else:
        out_funcs = flatten_splits(out_tree)
    if ret_tree:
        out = (out_funcs, out_tree)
    else:
        out = out_funcs
    return out


def decompose_task_object(stim, targs, **kwargs):
    """
    This function will find linearly separable decompositions of the task
    object.

    Notes:
    1. If a given partition is linearly separable, then all subpartitions will
    also be linearly separable.
    2. The maximal split is always linearly separable.

    Parameters
    ----------
    stim : (M, N) array_like
        An array with M trials and N-dimensional stimuli, only dimensions
        that are relevant to at least one context should be included
    targs : (M, T) array_like
        An array with the same M trials and the desired output on each trial

    Returns
    -------
    splits : dict
        A dictionary that describes all the linear decompositions of the
        object.
    """

    original_md = np.ones(stim.shape[1], dtype=bool)
    out = _decompose_task_object_helper(stim, targs, original_md, depth=0, **kwargs)
    return out


def binom_func(l_, k, n):
    num = ss.binom(l_, k) * ss.binom(l_ - k, n)
    denom = ss.binom(l_, k) * ss.binom(l_, n)
    return num / denom


@u.arg_list_decorator
def noncontext_split_probability(l_, t):
    outs = np.zeros((len(l_), len(t)))
    for i, j in u.make_array_ind_iterator(outs.shape):
        l_ij, t_ij = l_[i], t[j]
        p1 = 1 / (2 * l_ij)
        p2 = 1 - 1 / l_ij

        possible_ts = np.arange(0, t_ij + 1)
        n_pairs = ss.binom(t_ij, 2)

        n_pairs = ss.binom(t_ij - possible_ts, 2)

        t_ij_probs = sts.binom(t_ij, p1).pmf(possible_ts)

        combs = (
            p1 * p1 * (binom_func(l_ij, 1, 1))
            + 2 * p2 * p1 * (binom_func(l_ij, 1, 2))
            + p2 * p2 * (binom_func(l_ij, 2, 2))
        )
        print(1 - combs)
        # outs[i, j] = 1 - np.sum(t_ij_probs*combs*n_pairs)
        outs[i, j] = np.sum(n_pairs * t_ij_probs * (1 - combs))
    outs[outs < 0] = 0
    outs[outs > 1] = 1
    return np.squeeze(outs)


def _make_metric_mat(
    stim,
    groups,
    con_inds=(-2, -1),
    flip_groups=None,
    merger=ft.partial(np.concatenate, axis=1),
    metric=skmp.euclidean_distances,
    collapse_dim=None,
):
    c_masks = list(stim[:, ci] == 1 for ci in con_inds)

    group_size = len(groups[0])
    group_stim = np.zeros((len(groups), len(stim), group_size))
    for i, cm in enumerate(c_masks):
        g = stim[cm][:, groups[i]]
        group_stim[i, cm] = g

    if flip_groups is not None:
        for fg in flip_groups:
            group_stim[fg, c_masks[fg], 0] = 1 - group_stim[fg, c_masks[fg], 0]

    if collapse_dim is not None:
        for i in range(len(c_masks)):
            cd = np.reshape(collapse_dim, (group_size, 1))
            group_stim[i] = group_stim[i] @ cd
    group_stim = merger(group_stim)

    dists = metric(group_stim)
    return dists


def make_shared_metric(
    *args,
    **kwargs,
):
    merger = ft.partial(np.sum, axis=0)
    return _make_metric_mat(*args, merger=merger, **kwargs)


def make_orthog_metric(
    *args,
    **kwargs,
):
    merger = ft.partial(np.concatenate, axis=1)
    return _make_metric_mat(*args, merger=merger, **kwargs)


default_hypoth_makers = {
    "shared_full": make_shared_metric,
    "orthog_full": make_orthog_metric,
    "shared_collapse": ft.partial(make_shared_metric, collapse_dim=(1, -1)),
    "orthog_collapse": ft.partial(make_orthog_metric, collapse_dim=(1, -1)),
}


def rsa_correlation(
    model,
    time_reps=None,
    hypoth_makers=default_hypoth_makers,
    flip_groups=None,
    metric=skmp.linear_kernel,
    **kwargs,
):
    stim, inp_rep, targs = model.get_all_stim()
    con_inds = np.arange(-model.n_groups, 0)
    s_inds = np.argsort(stim[:, con_inds[-1]], axis=0)
    stim = stim[s_inds]
    inp_rep = inp_rep[s_inds]
    targs = targs[s_inds]
    model_rep = model.get_representation(inp_rep)
    groups = model.groups
    model_mat = metric(model_rep)
    corrs = {}
    for k, h_func in hypoth_makers.items():
        hypoth_mat = h_func(stim, groups, con_inds=con_inds, metric=metric, **kwargs)
        hm_flat = hypoth_mat.flatten()
        m_flat = model_mat.flatten()
        end_corr = np.corrcoef(hm_flat, m_flat)
        if time_reps is not None:
            t_corr = np.zeros((len(time_reps),) + end_corr.shape)
            for i, tr in enumerate(time_reps):
                tr_flat = metric(tr).flatten()
                t_corr[i] = np.corrcoef(tr_flat, hm_flat)
        else:
            t_corr = None

        corrs[k] = (end_corr, t_corr, hypoth_mat)
    return corrs


def task_masks(splits, n_vals=2):
    val_list = range(n_vals)
    mask_list = []
    for s in splits:
        s_masks = []
        for v in val_list:
            s_masks.append((s, v))
        mask_list.append(s_masks)
    prods = it.product(*mask_list)
    mask_funcs = {}
    for prod in prods:
        mask = None
        str_ident = []
        for s, v in prod:
            mask = ft.partial(_mask_func, k=s, i=v, comp_func=mask)
            str_ident.append("F{} = {}".format(s, v))
        full_ident = " and ".join(str_ident)
        mask_funcs[full_ident] = mask
    return mask_funcs


# this seems to work, but it needs to be merged a little further
# the longest key sequences should contain all of the shorter key
# sequences
def _flatten_splits_helper(
    split_dict,
    key,
    current_list,
    full_dict,
):
    # k is the feature
    # v_list is a list of nodes
    full_dict = {}
    for k, v_list in split_dict.items():
        k_dicts = []
        leaf_dict = {}
        comb_key = key + (k,)
        for i, v in enumerate(v_list):
            # v is a node in the tree
            # i is the index of that node

            # this node is a leaf, we want to add this function no matter
            # what
            if v is None:
                # a function saying if feature k is equal to i, include
                mf = ft.partial(_mask_func, k=k, i=i)
                for feat, val in current_list:
                    mf = ft.partial(_mask_func, k=feat, i=val, comp_func=mf)
                fl = leaf_dict.get(comb_key, [])
                fl.append(mf)
                leaf_dict[comb_key] = fl
            else:
                k_i_dict = _flatten_splits_helper(
                    v,
                    comb_key,
                    current_list + ((k, i),),
                    full_dict,
                )
                k_dicts.append(k_i_dict)

        if len(k_dicts) > 0:
            key_pairs = it.product(*list(kd_i.keys() for kd_i in k_dicts))
            for key_g in key_pairs:
                kg_list = list(k_dicts[i][key_g[i]] for i in range(len(k_dicts)))
                if len(leaf_dict) > 0:
                    kg_list = kg_list + [leaf_dict[comb_key]]

                full_dict[tuple(key_g)] = np.concatenate(kg_list)
        else:
            full_dict[comb_key] = leaf_dict[comb_key]

    return full_dict


# def _flatten_splits_helper(split_dict, level=0):
#     """
#     Returns a dictionary where the keys are the list of dimensions
#     that were conditioned on and the values are a list of functions
#     that split the stimuli into linearly separable groups
#     """
#     func_dict = {}

#     for k, v_list in split_dict.items():
#         leaf_dict = {}
#         partial_dict = {}
#         for i, v in enumerate(v_list):
#             if v is None:
#                 mf = ft.partial(_mask_func, k=k, i=i)
#                 ld_v = leaf_dict.get((k,), [])
#                 ld_v.append(mf)
#                 leaf_dict[(k,)] = ld_v
#             else:
#                 mf_sub = _flatten_splits_helper(v, level=level+1)
#                 if k == 3 and level == 0:
#                     print(mf_sub.keys())
#                     print(len(mf_sub.keys()))

#                 for sub_k, sub_v in mf_sub.items():
#                     comb_mfs = list(ft.partial(_mask_func,
#                                                k=k,
#                                                i=i,
#                                                comp_func=mf)
#                                     for mf in sub_v)
#                     comb_key = (k,) + sub_k
#                     pd_sk = partial_dict.get(comb_key, [])
#                     pd_sk.extend(comb_mfs)
#                     partial_dict[comb_key] = pd_sk
#             if k == 3 and level == 0:
#                 print('')
#                 print(i, 'pd', len(partial_dict))

#                 # print(partial_dict)
#                 print(i, 'ld', len(leaf_dict))
#         if k == 3 and level == 0:
#             print('')
#             print('pd', len(partial_dict)) # , partial_dict)
#             # print(partial_dict)
#             print('ld', len(leaf_dict)) # , leaf_dict)
#             print('----')

#         if k == 3 and level == 0:
#             print(len(partial_dict))

#         if len(partial_dict) > 0:
#             for pd_k, pd_v in partial_dict.items():
#                 if k == 3 and level == 0:
#                     print(pd_k, pd_v)
#                 if pd_k in func_dict.keys():
#                     print(pd_k, func_dict.keys())

#                 func_dict[pd_k] = pd_v

#                 for ld_k, ld_v in leaf_dict.items():
#                     func_dict[pd_k].extend(ld_v)
#         else:
#             intersect = set(func_dict.keys()).intersection(leaf_dict.keys())
#             if len(intersect) > 0:
#                 print(intersect)
#             func_dict.update(leaf_dict)
#         if k == 3 and level == 0:
#             # print('fd', func_dict)
#             print('fd', len(func_dict[(3,0)]))
#             print('---------')
#     if level == 0:
#         print(func_dict[(3, 0)])
#     return func_dict


def compute_svd(
    m, xs, ys, ind=0, thr=0.0001, unit=False, renorm_targs=True, final_weights=None
):
    if renorm_targs:
        ys = 2 * (ys - 0.5)
    reps = np.array(m(xs))
    x_mask = reps[:, ind] > thr

    xs_masked, ys_masked = xs[x_mask], ys[x_mask]
    comb_v = compute_weight_vector(
        xs_masked, ys_masked, unit=unit, renorm_targs=renorm_targs
    )
    if final_weights is not None:
        comb_v = comb_v * np.sign(final_weights)
    return comb_v


def compute_weight_vectors(xs, ys, gate_vectors):
    xs = np.array(xs)
    ys = np.array(ys)
    gate_vectors = np.array(gate_vectors)

    # gate_vectors has shape C x L
    # xs has shape N x L
    # ys has shape N x P
    n_orig = xs.shape[0]
    c = gate_vectors.shape[0]

    # shape C x N
    masks = (xs @ gate_vectors.T > 0).T

    xy_mats = []
    xx_mats = []
    for i in range(c):
        m_i = masks[i]
        sub_xx_mats = []
        for j in range(c):
            m_j = masks[j]
            m_comb = np.logical_and(m_i, m_j)
            sub_xx_mats.append(xs[m_comb].T @ xs[m_comb] / n_orig)
        xx_mats.append(np.concatenate(sub_xx_mats, axis=1))
        xy_mats.append(ys[m_i].T @ xs[m_i] / n_orig)

    xy_comb = np.concatenate(xy_mats, axis=1).T
    xx_comb = np.concatenate(xx_mats, axis=0)

    w, res, rank, s = np.linalg.lstsq(xx_comb, xy_comb, rcond=None)
    split_w = np.reshape(w, (c, xs.shape[1], ys.shape[1]))
    gvs = np.expand_dims(gate_vectors, 1)
    out_uvs = np.zeros((c, xs.shape[1]))
    for i in range(c):
        signs = np.sign(np.sum(gvs[i] * split_w[i].T, axis=1, keepdims=True))
        out_uvs[i] = u.make_unit_vector(np.sum(split_w[i].T * signs, axis=0))
        # out_uvs[i] = u.make_unit_vector(np.sum(split_w[i].T, axis=0))
    # print(gvs.shape, split_w.shape)
    # print(out_uvs.shape)
    # print(signs.shape)
    return out_uvs


def compute_stable_wvs(xs, ys, gate_vectors=None, n_samps=1000, iters=1000):
    if gate_vectors is None:
        gates, gate_vectors = sample_gates(xs, ys, n_samples=n_samps, ret_uvs=True)
    use_vecs = gate_vectors
    delts = np.zeros((iters, len(gate_vectors)))
    for i in range(iters):
        new_vecs = compute_weight_vectors(xs, ys, use_vecs)
        delts[i] = np.sum((use_vecs - new_vecs) ** 2, axis=1)
        use_vecs = new_vecs
    return use_vecs, delts


def compute_weight_vector_simple(xs, ys, ref_v, n_orig=None, use_alternates=True):
    xs = np.array(xs)
    ys = np.array(ys)
    if n_orig is None:
        n_orig = 1

    a = ys.T @ xs / n_orig
    x_m = xs.T @ xs / n_orig
    w = np.linalg.solve(x_m, a.T).T

    w_uv = u.make_unit_vector(w)
    if len(ref_v.shape) == 1:
        ref_v = np.expand_dims(ref_v, 0)
    signs = np.sign(np.sum(w_uv * ref_v, axis=1, keepdims=True))
    out_v = u.make_unit_vector(np.sum(w_uv * signs, axis=0))

    return out_v


def compute_weight_vector(
    xs,
    ys,
    n_orig=None,
    unit=True,
    renorm_targs=False,
    collapse=True,
    return_a=False,
    include_x=True,
    repoint_tasks=False,
    eps=1e-6,
    unique=False,
):
    xs = np.array(xs)
    ys = np.array(ys)
    if renorm_targs:
        ys = 2 * (ys - 0.5)
    if repoint_tasks:
        mult_mask = np.mean(ys < 0, axis=0) > 0.5
        mult = np.ones((1, ys.shape[1]))
        mult[:, mult_mask] = -1
        ys = ys * mult
    if n_orig is None:
        n_orig = 1
    a = (ys.T @ xs) / n_orig
    u_, s, vh = np.linalg.svd(a)
    # print('a\n', a)

    # there are multiple pathways going through each node
    # so need to subtract some multiple of x probably? not totally sure
    if include_x:
        x_corr = xs.T @ xs / n_orig
        # print('xc', x_corr)

        x_full = vh @ x_corr @ vh.T
        # x_full = vh.T @ x_corr @ vh
        # print('vh', vh)
        # print('xf', x_full)

        x_diag = np.diagonal(x_full)
        use_x = np.copy(x_diag[: s.shape[0]])
        use_x[use_x < eps] = 1
    else:
        use_x = np.ones(s.shape[0])

    # print('ma', np.mean(np.sign(a[:, 0:1])*a, axis=0))
    # print(use_x)
    # print(x_full)
    # print('vh', vh)
    # print(s)
    # print(use_x)
    # print(s/use_x)
    d_vals = np.sqrt(s / use_x)
    d_vals[np.isnan(d_vals)] = 0
    d_vals[np.isinf(d_vals)] = 0
    mat = np.diag(d_vals)
    # print('s', s)
    # print('x', use_x)
    # print('dv', d_vals)
    n_cols = vh.shape[0] - mat.shape[1]
    mat = np.concatenate((mat, np.zeros((mat.shape[0], n_cols))), axis=1)
    v = mat @ vh
    comb_v = np.array(u_[:, : v.shape[0]] @ v)
    # print(u_[:, :v.shape[0]])
    # print(comb_v)
    # print(v)
    # comb_v = np.sign(comb_v[:, 0:1])*comb_v
    print("vh\n", vh)

    print("ucv\n", u.make_unit_vector(np.unique(comb_v, axis=0)))
    # print(comb_v)
    # print('mean', np.mean(comb_v, axis=0))
    if unique:
        comb_v = np.unique(comb_v, axis=0)
    if collapse:
        comb_v = np.mean(np.abs(comb_v), axis=0, keepdims=True)

    # print('abs mean', comb_v)
    if unit:
        comb_v = u.make_unit_vector(comb_v, squeeze=False)
    out = comb_v
    if return_a:
        out = (comb_v, a, v)
    return out


def check_stability_masks(xs, masks, ys, **kwargs):
    stability = np.zeros(len(masks))
    for i, mask in enumerate(masks):
        stability[i] = check_stability(xs, mask, ys, ret_max=True, **kwargs)
    return stability


def sample_stable_gates(stims, targs, stable_only=True, **kwargs):
    gatings, uvs = sample_gates(stims, targs, ret_uvs=True)

    stability = np.zeros(len(gatings))
    svs = np.zeros_like(uvs)
    for i, mask in enumerate(gatings):
        stability[i], svs[i] = check_stability(
            stims, mask, targs, return_inferred_vec=True, **kwargs
        )
        # svs[i] = u.make_unit_vector(np.mean(stims[mask], axis=0)
        #                             - np.mean(stims[~mask], axis=0))
    print(stability)
    mask = stability == 1
    if stable_only:
        out = (gatings[mask], svs[mask])
    else:
        out = (gatings, svs, stability)
    return out


def existing_gates(mod, stims, targs, determine_stability=True):
    gates_all = np.array(mod.get_representation(stims) > 0)
    gates_u, gate_counts = np.unique(gates_all.T, axis=0, return_counts=True)

    if determine_stability:
        stability = check_stability_masks(stims, gates_u, targs)
    else:
        stability = None
    return gates_u, gate_counts, stability


def sample_gates(stims, targs, n_samples=10000, ret_uvs=False):
    us = u.make_unit_vector(sts.norm(0, 1).rvs((n_samples, stims.shape[1])))
    outs = np.array(tf.nn.relu(us @ stims.T))

    gatings, inds = np.unique(np.array(outs > 0), axis=0, return_index=True)
    out = gatings
    if ret_uvs:
        u_uvs = us[inds]
        out = (gatings, u_uvs)
    return out


def _corr_sim(x, y):
    sim = np.corrcoef(x, y)[1, 0]
    return sim


def _sel_sim(x, y):
    sim = np.mean(np.abs((x - y) / (x + y)))
    return sim


def compute_weight_alignment_difference(
    weights, norm_weights=False, use_corr=False, sim_func=_corr_sim
):
    if use_corr:
        corr_func = _corr_sim
    else:
        corr_func = _sel_sim

    n_cons, n_latents = weights.shape[:2]
    acrosses = it.product(range(n_latents), repeat=n_cons)
    if norm_weights:
        mu = np.mean(weights, axis=2, keepdims=True)
        std = np.std(weights, axis=2, keepdims=True)
        weights = mu / std  # (weights - mu)/std

    weights = np.abs(weights)
    across_contexts = []
    within_contexts = []
    for i in range(n_cons):
        withins = it.combinations(range(n_latents), n_cons)
        for wi_inds in withins:
            use_weights = weights[i][np.array(wi_inds)]
            for j in range(use_weights.shape[1]):
                within_contexts.append(corr_func(*use_weights[:, j]))
    for conj in acrosses:
        correl_vecs = []
        for i in range(n_cons):
            correl_vecs.append(weights[i, conj[i]])
        for g1, g2 in it.combinations(correl_vecs, 2):
            for j in range(g1.shape[0]):
                across_contexts.append(corr_func(g1[j], g2[j]))
    return within_contexts, across_contexts


def compute_model_alignment(model, n_samples=10000, **kwargs):
    stim_reps, samps, _ = model.get_x_true(n_train=n_samples)
    reps = model.get_representation(stim_reps)
    con_inds = np.arange(-model.n_groups, 0)
    u_inds = np.concatenate([np.unique(model.groups), con_inds])
    samps_use = samps[:, u_inds]

    cluster_labels = act_cluster(model)
    out = compute_alignment(reps, samps_use, n_contexts=model.n_groups, **kwargs)
    return out, u_inds, cluster_labels


def compute_fdg_frac_contextual(mod, **kwargs):
    return compute_frac_contextual(mod, use_fdg=True, **kwargs)


def compute_frac_contextual(mod, **kwargs):
    active_units = compute_silences(mod, **kwargs)
    singles = np.sum(active_units, axis=0) == 1
    frac = np.mean(singles)
    return frac


def _get_unit_std(
    mod,
    n_samps=1000,
    use_abs=True,
    layer=None,
    rescale=True,
    use_fdg=False,
    use_rotation=False,
):
    n_g = mod.n_groups
    if use_fdg:
        use_ind = 1
    else:
        use_ind = 2

    mod_reps = list(
        mod.sample_reps(n_samps, context=i, layer=layer)[use_ind] for i in range(n_g)
    )
    mod_rep = np.stack(mod_reps, axis=0)
    if use_rotation:
        trs = sts.ortho_group.rvs(mod_rep.shape[2])
        mod_rep = mod_rep @ trs
    if use_abs:
        mod_rep = np.abs(mod_rep)
    if rescale:
        unit_norms = np.max(mod_rep, axis=(0, 1), keepdims=True)
        mod_rep = mod_rep / unit_norms
    unit_std = np.zeros((n_g, mod_rep.shape[2]))
    for i in range(n_g):
        mr_i = mod_rep[i]
        unit_std[i] = np.std(mr_i, axis=0)
    return unit_std


def compute_variance_specialization(mod, **kwargs):
    unit_stds = _get_unit_std(mod, **kwargs)
    n_cons = unit_stds.shape[0]
    f = np.zeros((n_cons, n_cons, unit_stds.shape[1]))
    for i, j in it.combinations(range(n_cons), 2):
        denom = unit_stds[i] + unit_stds[j]
        f[i, j] = (unit_stds[i] - unit_stds[j]) / denom
        f[j, i] = (unit_stds[j] - unit_stds[i]) / denom
    return f


def compute_null_variance_specialization(mod, n_reps=100, **kwargs):
    n_groups = mod.n_groups
    n_units = mod.hidden_dims
    null_matrix = np.zeros((n_reps, n_groups, n_groups, n_units))
    for i in range(n_reps):
        null_matrix[i] = compute_variance_specialization(
            mod,
            use_rotation=True,
            **kwargs,
        )
    return null_matrix


def compute_variance_specialization_signif(mod, n_reps=100, p_thr=0.01, **kwargs):
    tv = compute_variance_specialization(mod, **kwargs)
    null = compute_null_variance_specialization(mod, n_reps=n_reps, **kwargs)
    tv_exp = np.expand_dims(tv, 0)
    cond_high = (tv_exp - null) > 0
    cond_low = (null - tv_exp) > 0
    sig_high = np.mean(cond_high[:, 0, 1], axis=0) > (1 - p_thr / 2)
    sig_low = np.mean(cond_low[:, 0, 1], axis=0) > (1 - p_thr / 2)
    frac = np.sum(np.logical_or(sig_high, sig_low)) / cond_low.shape[3]
    return frac


def compute_variance_threshold_frac(
    mod,
    thr=1e-2,
    **kwargs,
):
    active_units = _get_unit_std(mod, **kwargs)
    active_units = active_units > thr
    singles = np.sum(active_units, axis=0) == 1
    frac = np.mean(singles)
    return frac


def average_target_specialization(
    group_size, n_tasks, n_samps=1000, n_reps=30, n_groups=2, **kwargs
):
    out_ss = np.zeros(n_reps)
    for i in range(n_reps):
        m = ms.LinearModularizer(
            group_size + n_groups,
            tasks_per_group=n_tasks,
            group_size=group_size,
            group_maker=ms.overlap_groups,
            n_overlap=group_size,
            n_groups=n_groups,
            single_output=True,
            integrate_context=True,
            **kwargs,
        )

        lvs, inp, _, targs = m.sample_reps(return_targ=n_samps)
        subspace_spec = compute_alignment_index(m, use_targets=True)
        out_ss[i] = subspace_spec
    return out_ss


def compute_alignment_index_samples(stim, rep, cons, rel_vars, eps=1e-5):
    con_ais = []
    con_inds = np.unique(cons)
    for i, j in it.combinations(con_inds, 2):
        m1 = cons == i
        m2 = cons == j
        try:
            ai_ij = u.alignment_index(rep[m1], rep[m2])
        except np.linalg.LinAlgError:
            ai_ij = np.nan
        con_ais.append(ai_ij)
    con_ais = np.array(con_ais)

    rv_ais = np.zeros(len(rel_vars))
    for i, rv in enumerate(rel_vars):
        m1 = stim[:, rv] == 0
        m2 = stim[:, rv] == 1
        try:
            rv_ais_i = u.alignment_index(rep[m1], rep[m2])
        except np.linalg.LinAlgError:
            rv_ais_i = np.nan
        rv_ais[i] = rv_ais_i
    ss = max(np.log((np.mean(rv_ais) + eps) / (np.mean(con_ais) + eps)), 0)
    return ss


def compute_alignment_index_targets(*args, **kwargs):
    return compute_alignment_index(*args, **kwargs, use_targets=True)


def compute_alignment_index(
    mod, n_samps=1000, con_inds=None, use_targets=False, **kwargs
):
    if con_inds is None:
        con_inds = np.arange(mod.n_groups)
    rel_vars = np.unique(mod.groups)
    stim_all = []
    rep_all = []
    cons_all = []
    for ci in con_inds:
        stim, _, rep, targ = mod.sample_reps(
            n_samps=n_samps, context=ci, return_targ=True
        )
        if use_targets:
            rep = targ
        stim_all.append(stim)
        rep_all.append(rep)
        cons_all.append((ci,) * n_samps)
    stim = np.concatenate(stim_all, axis=0)
    rep = np.concatenate(rep_all, axis=0)
    cons = np.concatenate(cons_all, axis=0)
    return compute_alignment_index_samples(stim, rep, cons, rel_vars)


def compute_silences(
    mod,
    use_fdg=False,
    thr=1e-2,
    rescale=True,
    n_samps=1000,
    use_abs=True,
    layer=None,
    con_inds=None,
):
    if con_inds is None:
        con_inds = range(mod.n_groups)

    _, inp_rep, mod_rep = mod.sample_reps(n_samps * len(con_inds), layer=layer)
    if use_fdg:
        mod_rep = inp_rep
    if use_abs:
        mod_rep = np.abs(mod_rep)
    if rescale:
        unit_norms = np.max(mod_rep, axis=0, keepdims=True)
    else:
        unit_norms = np.ones((1, mod_rep.shape[1]))

    active_units = np.zeros((len(con_inds), mod_rep.shape[1]))
    for i in con_inds:
        _, inp_rep, mod_rep = mod.sample_reps(n_samps, context=i, layer=layer)
        if use_fdg:
            mod_rep = inp_rep
        if use_abs:
            mod_rep = np.abs(mod_rep)
        active_units[i] = np.mean(mod_rep / unit_norms, axis=0) > thr
    return active_units


def compute_alignment(reps, samps, n_contexts=2, n_folds=10):
    weights = np.zeros(
        (n_contexts, samps.shape[1] - n_contexts, n_folds, reps.shape[1])
    )
    corr = np.zeros((n_contexts, samps.shape[1] - n_contexts, n_folds))
    for i in range(n_contexts):
        con_mask = samps[:, -(i + 1)] == 1
        samps_con = np.array(samps[con_mask])
        reps_con = np.array(reps[con_mask])
        for j in range(samps.shape[1] - n_contexts):
            pipe = na.make_model_pipeline(skm.LinearSVC, dual="auto")
            out = skms.cross_validate(
                pipe, reps_con, samps_con[:, j], return_estimator=True, cv=n_folds
            )
            corr[i, j] = out["test_score"]
            weights[i, j] = np.concatenate(
                list(oe["linearsvc"].coef_ for oe in out["estimator"]), axis=0
            )
    return weights, corr


def _optimize_signs(vec, target_vec):
    c = -np.squeeze(vec * target_vec)
    integr = np.ones(len(c))
    bounds = spo.Bounds(-1, 1)
    if np.any(np.abs(c) > 0):
        res = spo.milp(c, integrality=integr, bounds=bounds)
        out = res.x
        if np.any(out == 0):
            print("zero is used in the integer programming solution")
    else:
        out = np.nan
    return out


def check_stability(
    xs,
    x_mask,
    ys,
    renorm_targs=False,
    thr=0,  # 1e-5,
    marginal=True,
    ret_max=True,
    eps=1e-10,
    return_new_mask=False,
    return_inferred_vec=False,
):
    x1 = np.mean(xs[x_mask], axis=0)
    x2 = np.mean(xs[np.logical_not(x_mask)], axis=0)
    corr_vec = u.make_unit_vector(x1 - x2, squeeze=False)

    inferred_vec = compute_weight_vector_simple(
        xs[x_mask], ys[x_mask], corr_vec, n_orig=xs.shape[0]
    )
    # print(inferred_vec.shape)

    # vec = compute_weight_vector(xs[x_mask], ys[x_mask],
    #                             renorm_targs=renorm_targs,
    #                             collapse=False,
    #                             unique=False,
    #                             repoint_tasks=False)

    # print('cv', corr_vec)
    # print('vec', vec)

    # THIS IS CLOSER THAN OTHER THINGS TO WORKING
    # BUT NOT AS BALANCED AS DEFAULT THING
    # sv = np.sign(np.sum(corr_vec * vec, axis=1, keepdims=True))
    # inferred_vec = u.make_unit_vector(np.mean(sv*vec, axis=0))

    # BEST VECTOR WAY
    # ind = np.argmax(np.sum(vec * corr_vec, axis=1))
    # inferred_vec = vec[ind]
    # print(inferred_vec)

    # ORIGINAL WAY
    # signs = _optimize_signs(vec, corr_vec)
    # inferred_vec = vec*np.array(signs)
    new_mask = (inferred_vec @ xs.T) > thr
    print("iv\n", inferred_vec)
    print("cv\n", corr_vec)
    print(x_mask)
    print(new_mask)
    print(np.all(new_mask == x_mask))
    print("---")
    # print(((vec*np.array(signs)) @ xs.T))
    c = new_mask == x_mask
    if marginal:
        corresps = np.mean(c)
    else:
        corresps = np.all(c)
    if return_new_mask:
        corresps = (corresps, new_mask)
    if return_inferred_vec:
        corresps = (
            corresps,
            inferred_vec,
        )
    return corresps


def compute_gated_io_svd(stim, stim_rep, targ, func):
    mask = np.expand_dims(func(stim), 1)
    out = ((mask * targ).T @ stim_rep) / mask.shape[0]
    u, s, vh = np.linalg.svd(out, full_matrices=False)
    return out, (u, s, vh)


def computed_gated_input_covariance(stim, stim_rep, func1, func2):
    mask1 = np.expand_dims(func1(stim), 1)
    mask2 = np.expand_dims(func2(stim), 1)
    out = mask1 * stim_rep @ (mask2 * stim_rep).T
    vals, vecs = np.linalg.eig(out)
    return out, (vals, vecs)


def _compute_cov_spectrum(rel_samps, reps, targs, func_list, svd_func=np.mean):
    outs = []
    fracs = []
    for i, f_use in enumerate(func_list):
        out, (u, s, vh) = compute_gated_io_svd(
            rel_samps,
            reps,
            targs,
            f_use,
        )
        mask = np.expand_dims(f_use(rel_samps), 1)
        fracs.append(np.abs(np.mean(mask * targs, axis=0)))
        outs.append(svd_func(s))
    return outs, fracs


def compute_combined_svd(fdg, m, func_list, n_samps=10000, norm_targs=True):
    _, latent_samps, targs = m.get_x_true(n_train=n_samps)
    if norm_targs:
        targs = skp.StandardScaler().fit_transform(targs)
    reps = np.array(fdg.get_representation(latent_samps))
    rel_samps = maux.get_relevant_dims(latent_samps, m)

    full_corr = (targs.T @ reps) / targs.shape[0]
    u, s, vh = np.linalg.svd(full_corr, full_matrices=True)

    sub_s = []
    for func in func_list:
        mask = np.expand_dims(func(rel_samps), axis=1)
        sub_corr = ((mask * targs).T @ reps) / targs.shape[0]
        new_diag = u.T @ sub_corr @ vh.T
        sub_s.append(np.diag(new_diag))
    return sub_s


def covariance_spectrum(*args, **kwargs):
    return _func_dict_quantifier(*args, use_func=_compute_cov_spectrum, **kwargs)


def compute_gate_vector(rel_samps, reps, func, **kwargs):
    mask = func(rel_samps)
    m = skm.LinearSVC(dual="auto", **kwargs)
    m.fit(reps, mask)
    return u.make_unit_vector(m.coef_)


def _compute_gate_alignment(rel_samps, reps, targs, func_list):
    outs = []
    for i, f_use in enumerate(func_list):
        out, (u_, s, vh) = compute_gated_io_svd(
            rel_samps,
            reps,
            targs,
            f_use,
        )
        gating_vec = compute_gate_vector(rel_samps, reps, f_use)
        weighted_svd = u.make_unit_vector(np.expand_dims(s, 1) * vh)
        dot_prod = np.sum(np.expand_dims(gating_vec, 0) * weighted_svd, axis=1)
        outs.append(dot_prod)
    return outs


def gating_alignment(*args, **kwargs):
    return _func_dict_quantifier(*args, use_func=_compute_gate_alignment, **kwargs)


def _func_dict_quantifier(
    fdg,
    m,
    func_dict,
    n_samps=1000,
    svd_func=np.mean,
    norm_targs=True,
    use_func=_compute_cov_spectrum,
):
    _, latent_samps, targs = m.get_x_true(n_train=n_samps)
    if norm_targs:
        targs = skp.StandardScaler().fit_transform(targs)
    reps = fdg.get_representation(latent_samps)
    rel_samps = maux.get_relevant_dims(latent_samps, m)
    outs_dict = {}
    for k, func_list in func_dict.items():
        outs = use_func(rel_samps, reps, targs, func_list)
        outs_dict[k] = outs
    return outs_dict


def flatten_splits(split_dict):
    """
    The split dict has keys, which correspond to dimensions, and values,
    which give the fixed value of that dimension

    This function constructs a list of lists of gating functions that take in
    stimuli and return a gating output (zero or one).

    Each element of the output list consists of gating functions that would
    provide a linearly separable output of the task object.
    """
    out = _flatten_splits_helper(split_dict, (), (), {})

    # maybe add merging function here
    return out
