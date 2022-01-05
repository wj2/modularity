
import pickle
import numpy as np
import os
import re
import pandas as pd
import itertools as it

import general.utility as u

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
default_cluster_keys = ('sim_diffs', 'threshold_diffs', 'cosine_sim_diffs',
                        'cosine_sim_absolute_diffs')
def _add_model(df, md, full_mat_keys=default_fm_keys,
               cluster_keys=default_cluster_keys,
               group_key='groups', **kwargs):
    group_size = md['args'].group_size
    tasks_per_group = md['args'].tasks_per_group
    group_method = md['args'].group_method
    model_type = md['args'].model_type
    group_size, tasks_per_group, group_method, model_type = _make_lists(
        group_size, tasks_per_group, group_method, model_type)
    arg_dict = {}
    for key, v in vars(md['args']).items():
        arg_dict['args_' + key] = v
    arr_shape = list(md.values())[0].shape
    for (i, j, k, l, n) in u.make_array_ind_iterator(arr_shape):
        row_dict = dict(group_size=group_size[i],
                        tasks_per_group=tasks_per_group[j],
                        group_method=group_method[k],
                        model_type=model_type[l])
        row_dict.update(kwargs)
        row_dict.update(arg_dict)
        for mk, vk in md.items():
            if mk in full_mat_keys:
                row_dict[mk] = [vk[i, j, k, l, n]]
            if mk in cluster_keys:
                row_dict[mk] = vk[i, j, k, l, n]
            if mk == group_key:
                row_dict['overlap'] = _compute_group_overlap(
                    vk[i, j, k, l, n])
        df_ijkl = pd.DataFrame(row_dict)
        df = df.append(df_ijkl)
    return df        
    
def load_models(folder, file_template='modularizer_([0-9]+)-([0-9]+)',
                file_name='model_results.pkl'):
    files = os.listdir(folder)
    df = pd.DataFrame()
    for fl in files:
        m = re.match(file_template, fl)
        if m is not None:
            arr = m.group(1)
            job = m.group(2)
            full_path = os.path.join(folder, fl, file_name)
            model_dict = pickle.load(open(full_path, 'rb'))
            df = _add_model(df, model_dict, arr_id=arr, job_id=job)
    return df
