
import pickle
import numpy as np
import os
import re
import pandas as pd
import itertools as it

import general.utility as u
import modularity.analysis as ma

def get_relevant_dims(samps, m, preserve_order_if_same=True):
    n_contexts = len(m.groups)
    if np.all(np.var(m.groups, axis=0) == 0):
        rel_inds = m.groups[0]
    else:
        rel_inds = np.unique(m.groups)
    rel_stim = samps[:, rel_inds]
    rel_stim = np.concatenate((rel_stim, samps[:, -n_contexts:]), axis=1)
    return rel_stim

def save_model_information(models, folder, file_name='model_results.pkl',
                           **kwargs):
    weights = np.zeros_like(models)
    group_members = np.zeros_like(models)
    groups = np.zeros_like(models)
    for ind in u.make_array_ind_iterator(models.shape):
        weights[ind] = list(np.array(mw) for mw in models[ind].model.weights)
        group_members[ind] = models[ind].out_group_labels
        groups[ind] = models[ind].groups
    save_dict = dict(weights=weights, group_members=group_members,
                     groups=groups)
    save_dict.update(kwargs)
    file_path = os.path.join(folder, file_name)
    os.mkdir(folder)
    pickle.dump(save_dict, open(file_path, 'wb'))

@u.arg_list_decorator
def _make_lists(*args):
    return args

def _compute_group_overlap(groups):
    overlaps = []
    for g_i, g_j in it.combinations(groups, 2):
        overlaps.append(len(set(g_i).intersection(g_j)))
    return np.mean(overlaps)

default_fm_keys = ('weights', 'sim_mats', 'groups', 'group_members',
                   'threshold_mats', 'cosine_sim_mats',
                   'cosine_sim_absolute_mats')
default_fm_keys = ()
default_cluster_keys = ('sim_diffs', 'threshold_diffs', 'cosine_sim_diffs',
                        'cosine_sim_absolute_diffs', 'brim_diffs',
                        'gm', 'l2', 'l1', 'within_act_ablation',
                        'across_act_ablation', 'within_graph_ablation',
                        'across_graph_ablation',
                        'max_corr', 'within_max_corr_ablation',
                        'across_max_corr_ablation')
default_geometry_keys = ('shattering', 'within_ccgp', 'across_ccgp')
default_loss_keys = ('loss', 'val_loss')

def _add_model(df, md, full_mat_keys=default_fm_keys,
               cluster_keys=default_cluster_keys,
               group_key='groups', geometry_keys=default_geometry_keys,
               loss_keys = default_loss_keys, **kwargs):
    all_args = vars(md['args'])
    group_size = all_args.pop('group_size')
    tasks_per_group = all_args.pop('tasks_per_group')
    try:
        group_method = all_args.pop('group_method')
        group_overlap = (-1,)
    except:
        group_method = 'overlap'
        group_overlap = all_args.pop('group_overlap')
    model_type = all_args.pop('model_type')
    n_groups = all_args.pop('n_groups')
    n_tasks = n_groups*tasks_per_group
    out = _make_lists(group_size, tasks_per_group, group_method, model_type)
    group_size, tasks_per_group, group_method, model_type = out
    arg_dict = {}
    for key, v in all_args.items():
        arg_dict['args_' + key] = v
    arr_shape = list(md.values())[0].shape
    all_rows = []
    for (i, j, k, l, m, n) in u.make_array_ind_iterator(arr_shape):
        row_dict = dict(group_size=group_size[i],
                        tasks_per_group=tasks_per_group[j],
                        group_method=group_method[k],
                        model_type=model_type[l],
                        group_overlap=group_overlap[m],
                        n_groups=n_groups)
        row_dict.update(kwargs)
        row_dict.update(arg_dict)        
        for mk, vk in md.items():
            if mk != 'args':
                vk_ind = vk[i, j, k, l, m, n]
            else:
                vk_ind = None
            if mk in full_mat_keys:
                row_dict[mk] = [vk_ind]
            if mk in cluster_keys:
                row_dict[mk] = vk_ind
            if mk in geometry_keys:
                row_dict[mk] = np.mean(vk_ind)
            if mk in loss_keys:
                lv = vk_ind[vk_ind > 0][-1]
                row_dict[mk] = lv # vk_ind[-1]/n_tasks[i]
            if mk == group_key:
                row_dict['overlap'] = _compute_group_overlap(
                    vk_ind)
        df_ijkl = pd.DataFrame(row_dict)
        all_rows.append(df_ijkl)
        # df = pd.concat((df, df_ijkl), ignore_index=True)
    return all_rows

def _get_n_tasks(run_dict):
    return run_dict['args'].tasks_per_group

def get_nl_strength(run_dict):
    return run_dict['args'].dm_input_mixing

def sort_dict(sd, ordering, squeeze=True, stack_ax=0,
              no_mean=('dimensionality',)):
    ordering = np.squeeze(np.array(ordering))
    order_inds = np.argsort(ordering)
    ordering = ordering[order_inds]
    sorted_dict = {}
    for k, v in sd.items():
        v = np.squeeze(np.stack(list(v), axis=stack_ax))
        v_sort = v[order_inds]
        if k not in no_mean:
            m_range = tuple(range(2, 2 + len(v_sort.shape[2:])))
            v_sort = np.mean(v_sort,
                             axis=m_range)
        sorted_dict[k] = v_sort
    return sorted_dict, ordering


def _diff_ablation(d, add_key, sub_key):
    return np.array(d[add_key]) - np.array(d[sub_key])


def _diff_act_ablation(d):
    return _diff_ablation(d, 'within_act_ablation', 'across_act_ablation')


def _diff_graph_ablation(d):
    return _diff_ablation(d, 'within_graph_ablation', 'across_graph_ablation')


def _diff_max_corr_ablation(d):
    return _diff_ablation(d, 'within_max_corr_ablation', 'across_max_corr_ablation')


def load_run(run_ind, folder='modularity/simulation_data/',
             file_template='modularizer_([0-9]+)-{run_ind}',
             file_name='model_results.pkl',
             ordering_func=_get_n_tasks,
             take_keys=('within_ccgp', 'across_ccgp', 'shattering',
                        'within_act_ablation', 'across_act_ablation',
                        'gm', 'within_graph_ablation',
                        'across_graph_ablation',
                        'within_max_corr_ablation',
                        'across_max_corr_ablation',
                        'max_corr', 'dimensionality', 'corr_rate',
                        'model_frac', 'fdg_frac'),
             add_keys=None):
    if add_keys is None:
        add_keys = {
            'diff_act_ablation': _diff_act_ablation,
            'diff_graph_ablation': _diff_graph_ablation,
            # 'diff_max_corr_ablation': _diff_max_corr_ablation,
        }
    files = os.listdir(folder)
    f_template = file_template.format(run_ind=run_ind)
    out_dict = {}
    ordering = []
    for fl in files:
        m = re.match(f_template, fl)
        if m is not None:
            job = m.group(1)
            full_path = os.path.join(folder, fl, file_name)
            model_dict = pickle.load(open(full_path, 'rb'))
            args = vars(model_dict['args'])
            ordering.append(ordering_func(model_dict))
            for k in take_keys:
                l = out_dict.get(k, [])
                if k in model_dict.keys():
                    md_k = np.squeeze(model_dict[k])
                    md_k = np.stack(list(md_k), axis=0)

                    l.append(md_k)
                    out_dict[k] = l
    for k, func in add_keys.items():
        out_dict[k] = func(out_dict)
    return sort_dict(out_dict, ordering) + (args,)

def load_models(folder, file_template='modularizer_([0-9]+)-([0-9]+)',
                file_name='model_results.pkl'):
    files = os.listdir(folder)
    df = pd.DataFrame()
    full_list = []
    for fl in files:
        m = re.match(file_template, fl)
        if m is not None:
            arr = m.group(1)
            job = m.group(2)
            full_path = os.path.join(folder, fl, file_name)
            model_dict = pickle.load(open(full_path, 'rb'))
            l = _add_model(df, model_dict, arr_id=arr, job_id=job)
            full_list.extend(l)
    df = pd.concat(full_list, ignore_index=True)
    return df

def add_brim(row, weight_ind=2, **kwargs):
    out = ma.simple_brim(row['group_members'], row['weights'][weight_ind],
                         **kwargs)
    return out

def add_field(df, field_name, field_func, **kwargs):
    add_col = np.zeros(len(df), dtype=object)
    for i, (_, row) in enumerate(df.iterrows()):
        out = field_func(row, **kwargs)
        add_col[i] = out
    df[field_name] = add_col
    return df
