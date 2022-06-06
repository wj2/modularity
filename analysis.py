
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition as skd
import itertools as it
import scipy.stats as sts
import scipy.special as ss
import tensorflow as tf

tfk = tf.keras

import general.utility as u
import general.stan_utility as su
import modularity.simple as ms
import disentangled.data_generation as dg
import composite_tangling.code_creation as cc

class ModularizerCode(cc.Code):

    def __init__(self, model, group_ind=None, dg_model=None, source_distr=None,
                 n_values=2, noise_cov=.1**2):
        if dg_model is not None and source_distr is None:
            self.n_feats_all = dg_model.input_dim
        elif dg_model is None and source_distr is not None:
            self.n_feats_all = source_distr.rvs(1).shape[1]
        else:
            raise IOError('one of dg_model or source_distr must be provided')
        self.group_ind = group_ind
        if self.group_ind is not None:
            self.group = model.groups[group_ind]
        else:
            self.group = np.arange(self.n_feats_all, dtype=int)
        self.n_feats = len(self.group)
        self.n_values = n_values
        self.n_neurs = model.hidden_dims        
        self.n_stimuli = self.n_values**self.n_feats
        self.noise_distr = sts.multivariate_normal(np.zeros(self.n_neurs),
                                                   noise_cov)
        self.model = model
        self.dg_model = dg_model
        self.power = self._get_avg_power()
        self.stim = self._get_all_stim()

    def _get_avg_power(self, n_avg=10000):
        samps, reps = self.dg_model.sample_reps(n_avg)
        pwr = np.mean(np.sum(self.model.get_representation(reps)**2, axis=1),
                      axis=0)
        return pwr
    
    def _get_all_stim(self):
        stim = list(it.product(range(self.n_values), repeat=self.n_feats))
        return stim

    def get_random_full_stim(self, **kwargs):
        return self._get_random_stim(self.n_feats_all, **kwargs)

    def get_random_stim(self):
        return self._get_random_stim(self.n_feats, **kwargs)

    def _get_random_stim(self, n_feats, non_nan=None):
        s = np.random.choice(np.arange(self.n_values, dtype=float),
                             size=n_feats)
        if non_nan is not None:
            if n_feats < self.n_feats_all:
                mask = np.logical_not(np.isin(self.group, non_nan))
            else:
                mask = np.logical_not(np.isin(np.arange(n_feats), non_nan))
            s[mask] = np.nan
        return s
            
        
    def get_full_stim(self, stim):
        f_stim, _ = self.dg_model.sample_reps(stim.shape[0])
        f_stim[:, self.group] = stim
        return f_stim

    def get_nan_stim(self, stim, ref_stim=None):
        if ref_stim is None:
            n_stim, _ = self.dg_model.sample_reps(stim.shape[0])
        else:
            n_stim = ref_stim
        mask = np.logical_not(np.isnan(stim))
        n_stim[mask] = stim[mask]
        return n_stim
    
    def get_representation(self, stim, noise=False, ret_stim=False,
                           ref_stim=None):
        stim = np.array(stim)
        if len(stim.shape) == 1:
            stim = np.expand_dims(stim, 0)
        if np.any(np.isnan(stim)):
            stim = self.get_nan_stim(stim, ref_stim=ref_stim)
        if stim.shape[1] == len(self.group):
            stim = self.get_full_stim(stim)
        stim_rep = self.dg_model.get_representation(stim)
        reps = self.model.get_representation(stim_rep)
        if noise:
            reps = self._add_noise(reps)
        if ret_stim:
            out = (reps, stim)
        else:
            out = reps
        return out

    def _sample_noisy_reps(self, cs, n, add_noise=True, ret_stim=False,
                           ref_stim=None):
        cs = np.array(cs)
        if len(cs.shape) == 1:
            cs = np.expand_dims(cs, 0)
        r_inds = np.random.choice(cs.shape[0], int(n))
        out = self.get_representation(cs[r_inds], noise=add_noise,
                                      ret_stim=ret_stim,
                                      ref_stim=ref_stim)
        return out
    
    def compute_shattering(self, n_reps=5, thresh=.6, **dec_args):
        partitions = self._get_partitions()
        n_parts = len(partitions)
        pcorrs = np.zeros((n_parts, n_reps))
        stim_arr = np.array(self.stim)
        for i, ps1 in enumerate(partitions):
            ps2 = list(set(range(self.n_stimuli)).difference(ps1))
            c1 = stim_arr[ps1]
            c2 = stim_arr[ps2]
            pcorrs[i] = self.decode_rep_classes(c1, c2, n_reps=n_reps,
                                                **dec_args)
        n_c = np.sum(np.mean(pcorrs, axis=1) > thresh)
        n_dim = np.log2(2*n_c)
        n_dim_poss = np.log2(2*n_parts)
        return n_dim, n_dim_poss, pcorrs

    def compute_within_group_ccgp(self, n_reps=10, max_combos=20,
                                  **kwargs):
        combos = it.combinations(self.group, 2)
        n_possible_combos = int(ss.comb(len(self.group), 2))
        if n_possible_combos > max_combos:
            comb_inds = np.random.choice(range(n_possible_combos), max_combos,
                                         replace=False)
            combos = np.array(list(combos))[comb_inds]
            n_possible_combos = max_combos
        out = np.zeros((n_possible_combos, n_reps))
        for i, (td, gd) in enumerate(combos):
            out[i] = self.compute_specific_ccgp(td, gd, n_reps=n_reps,
                                                **kwargs)
        return out

    def compute_across_group_ccgp(self, n_reps=10, max_combos=20,
                                  **kwargs):
        all_inds = np.arange(self.n_feats_all, dtype=int)
        non_group_inds = set(all_inds).difference(self.group)
        combos = it.product(self.group, non_group_inds)
        n_possible_combos = len(self.group)*len(non_group_inds)
        if n_possible_combos > max_combos:
            comb_inds = np.random.choice(range(n_possible_combos), max_combos,
                                         replace=False)
            combos = np.array(list(combos))[comb_inds]
            n_possible_combos = max_combos
        out = np.zeros((n_possible_combos, n_reps))
        for i, (td, gd) in enumerate(combos):
            out[i] = self.compute_specific_ccgp(td, gd, n_reps=n_reps,
                                                **kwargs)
        return out
    
    def compute_specific_ccgp(self, train_dim, gen_dim, train_dist=1,
                              gen_dist=1, n_reps=10, ref_stim=None,
                              train_noise=False, n_train=10,
                              balance_training=False, **dec_kwargs):
        if (ref_stim is None and train_dim in self.group
            and gen_dim in self.group):
            ref_stim = self.get_random_full_stim(non_nan=(train_dim, gen_dim))
        elif ref_stim is None:
            ref_stim = self.get_random_full_stim(non_nan=(train_dim, gen_dim))
        tr_stim = np.mod(np.array(tuple(rs + train_dist*(i == train_dim)
                                        for i, rs in enumerate(ref_stim))),
                         self.n_values)
        gen_stim1 = np.mod(np.array(tuple(rs + gen_dist*(i == gen_dim)
                                          for i, rs in enumerate(ref_stim))),
                           self.n_values)
        gen_stim2 = np.mod(np.array(tuple(rs + gen_dist*(i == gen_dim)
                                          for i, rs in enumerate(tr_stim))),
                           self.n_values)
        pcorr = self.decode_rep_classes(ref_stim, tr_stim,
                                        c1_test=gen_stim1,
                                        c2_test=gen_stim2,
                                        n_reps=n_reps,
                                        train_noise=train_noise,
                                        n_train=n_train,
                                        balance_training=balance_training,
                                        **dec_kwargs)

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

@u.arg_list_decorator
def train_variable_models(group_size, tasks_per_group, group_maker, model_type,
                          n_reps=2, **kwargs):
    out_ms = np.zeros((len(group_size), len(tasks_per_group), len(group_maker),
                       len(model_type), n_reps), dtype=object)
    out_hs = np.zeros_like(out_ms)
    for (i, j, k, l) in u.make_array_ind_iterator(out_ms.shape[:-1]):
        out = train_n_models(group_size[i], tasks_per_group[j],
                             group_maker=group_maker[k],
                             model_type=model_type[l],
                             n_reps=n_reps, **kwargs)
        out_ms[i, j, k, l], out_hs[i, j, k, l] = out
    return out_ms, out_hs

def contrast_rich_lazy(inp_dim, rep_dim, init_bounds=(.01, 3), n_inits=20,
                        train_epochs=0, **kwargs):
    weight_vars = np.linspace(*(init_bounds + (n_inits,)))
    source_distr = u.MultiBernoulli(.5, inp_dim)
    for i, wv in enumerate(weight_vars):
    
        kernel_init = tfk.initializers.RandomNormal(stddev=wv)
        fdg_ut = dg.FunctionalDataGenerator(inp_dim, (), rep_dim,
                                            source_distribution=source_distr, 
                                            use_pr_reg=True,
                                            kernel_init=kernel_init,
                                            **kwargs)
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


def train_n_models(group_size, tasks_per_group, group_width=200, fdg=None,
                   n_reps=2, n_groups=5, group_maker=ms.random_groups,
                   model_type=ms.ColoringModularizer, epochs=5, verbose=False,
                   **training_kwargs):
    if fdg is None:
        use_mixer = False
    else:
        use_mixer = True
    inp_dim = fdg.input_dim
    out_ms = []
    out_hs = []
    for i in range(n_reps):
         m_i = model_type(inp_dim, group_size=group_size, n_groups=n_groups,
                          group_maker=group_maker, use_dg=fdg,
                          group_width=group_width, use_mixer=use_mixer,
                          tasks_per_group=tasks_per_group)
         h_i = m_i.fit(epochs=epochs, verbose=verbose, **training_kwargs)
         out_ms.append(m_i)
         out_hs.append(h_i)
    return out_ms, out_hs

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

def _reorg_matrix(wm, groups):
    u_groups = np.unique(groups)
    n_groups = len(u_groups)
    n_tasks = int(len(groups)/n_groups)
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
    new_wm = (wm - mu_wm)/std_wm
    return new_wm 

def compute_prob_cluster(wt, std_prior=1, model_path='modularity/wm_mixture.pkl',
                         **kwargs):
    n_units, n_groups, n_tasks = wt.shape
    stan_dict = dict(N=n_units, M=n_groups, T=n_tasks,
                     std_prior=std_prior, W=wt)
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

def simple_brim(groups, w_matrix, threshold=.1):
    wm = standardize_wm(w_matrix)
    conn_m = np.abs(wm) > threshold
    null_m = np.ones_like(conn_m)*np.mean(conn_m, axis=1, keepdims=True)
    b_tilde = conn_m - null_m
    t_mat = make_group_matrix(groups)
    t_tilde = np.dot(b_tilde, t_mat)
    inf_groups = np.argmax(t_tilde, axis=1)
    r_mat = make_group_matrix(inf_groups)
    prod_mat = np.dot(r_mat.T, t_tilde)
    out_mod = np.trace(prod_mat)/np.sum(conn_m)
    out_mat = np.identity(len(np.unique(groups)))*out_mod
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

def apply_geometry_model_list(ml, fdg, group_ind=0, **kwargs):
    ml = np.array(ml)
    shattering = np.zeros_like(ml, dtype=object)
    within_ccgp = np.zeros_like(shattering)
    across_ccgp = np.zeros_like(shattering)
    for ind in u.make_array_ind_iterator(ml.shape):
        m = ml[ind]
        m_code = ModularizerCode(m, dg_model=fdg, group_ind=group_ind)
        shattering[ind] = m_code.compute_shattering(**kwargs)[-1]
        within_ccgp[ind] = m_code.compute_within_group_ccgp(**kwargs)
        across_ccgp[ind] = m_code.compute_across_group_ccgp(**kwargs)
    return shattering, within_ccgp, across_ccgp

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

def process_histories(hs):
    hs = np.array(hs)
    ind = (0,)*len(hs.shape)
    n_epochs = hs[ind].params['epochs']
    loss = np.zeros(hs.shape + (n_epochs,))
    loss_val = np.zeros_like(loss)
    for ind in u.make_array_ind_iterator(hs.shape):
        loss[ind] = hs[ind].history['loss']
        loss_val[ind] = hs[ind].history['val_loss']
    return loss, loss_val
