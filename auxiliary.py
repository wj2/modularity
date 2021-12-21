
import pickle
import numpy as np
import os

import general.utility as u

def save_model_information(models, sim_mats, sim_diffs, args, folder,
                           file_name='model_results.pkl'):
    weights = np.zeros_like(models)
    group_members = np.zeros_like(models)
    for ind in u.make_array_ind_iterator(models.shape):
        weights[ind] = list(np.array(mw) for mw in models[ind].model.weights)
        group_members[ind] = models[ind].out_group_labels
    save_dict = dict(weights=weights, group_members=group_members,
                     sim_mats=sim_mats, sim_diffs=sim_diffs, args=args)
    file_path = os.path.join(folder, file_name)
    os.mkdir(folder)
    pickle.dump(save_dict, open(file_path, 'wb'))
        
