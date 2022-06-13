
import argparse
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft
import tensorflow as tf
from datetime import datetime

import general.utility as u
import disentangled.data_generation as dg

import modularity.simple as ms
import modularity.analysis as ma
import modularity.auxiliary as maux

tfk = tf.keras

def create_parser():
    parser = argparse.ArgumentParser(description='fit several modularizers')
    parser.add_argument('-o', '--output_folder', default='results-n', type=str,
                        help='folder to save the output in')
    parser.add_argument('--input_dim', default=20, type=int,
                        help='dimensionality of input')
    parser.add_argument('--rep_dim', default=400, type=int,
                        help='dimensionality of representation layer')
    parser.add_argument('--fdg_epochs', default=5, type=int,
                        help='epochs to train DG for')
    
    parser.add_argument('--group_size', default=(4,), type=int,
                        help='size of groups', nargs='+')
    parser.add_argument('--tasks_per_group', default=(4,), type=int,
                        help='number of tasks for each group', nargs='+')
    parser.add_argument('--group_method', default='random', type=str,
                        help='type of group selection')
    parser.add_argument('--n_groups', default=5, type=int,
                        help='number of groups')
    parser.add_argument('--model_type', default='coloring', type=str,
                        help='kind of model')
    parser.add_argument('--model_epochs', default=40, type=int,
                        help='epochs to train model for')
    parser.add_argument('--model_batch_size', default=100, type=int,
                        help='batch size for model training')
    parser.add_argument('--cluster_method', default='threshold',
                        type=str, help='method for quantifying clustering')
    parser.add_argument('--cumu_weight', default=.5, type=float,
                        help='default cumulative weight')
    parser.add_argument('--n_reps', default=5, type=int,
                        help='number of repeats')
    parser.add_argument('--dg_batch_size', default=100, type=int)
    parser.add_argument('--brim_threshold', default=1.5, type=float)
    parser.add_argument('--fdg_weight_init', default=None, type=float)
    parser.add_argument('--fdg_layers', nargs='+', default=(300,),
                        type=int)
    return parser

selector_dict = {'random':ms.random_groups,
                 'sequential':ms.sequential_groups}
model_dict = {'xor':ms.XORModularizer,
              'coloring':ms.ColoringModularizer,
              'linear':ms.LinearModularizer}

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()

    source_distr = u.MultiBernoulli(.5, args.input_dim)
    if args.fdg_weight_init is None:
        kernel_init = None
    else:
        kernel_init = tfk.initializers.RandomNormal(stddev=args.fdg_weight_init)
        
    fdg = dg.FunctionalDataGenerator(args.input_dim, args.fdg_layers,
                                     args.rep_dim,
                                     source_distribution=source_distr,
                                     use_pr_reg=True, kernel_init=kernel_init)
    fdg.fit(epochs=args.fdg_epochs, batch_size=args.dg_batch_size, verbose=False)
    rep_dim = fdg.representation_dimensionality(participation_ratio=True)
    print('rep dim: {}'.format(rep_dim))

    cluster_dict = {'threshold':ft.partial(ma.threshold_clusters,
                                           cumu_weight=args.cumu_weight),
                    'cosine_sim':ft.partial(ma.quantify_clusters,
                                            absolute=False),
                    'cosine_sim_absolute':ft.partial(ma.quantify_clusters,
                                                     absolute=True),
                    'brim':ft.partial(ma.simple_brim,
                                      threshold=args.brim_threshold)}
    group_selector = selector_dict[args.group_method]
    model_type = model_dict[args.model_type]
    out = ma.train_variable_models(args.group_size, args.tasks_per_group,
                                   group_selector, model_type, fdg=fdg,
                                   n_groups=args.n_groups,
                                   n_reps=args.n_reps,
                                   epochs=args.model_epochs,
                                   batch_size=args.model_batch_size)
    models, histories = out
    clustering_results = {}
    for cm, func in cluster_dict.items():
        mats, diffs = ma.apply_clusters_model_list(models, func)
        clustering_results[cm + '_mats'] = mats
        clustering_results[cm + '_diffs'] = diffs

    out = ma.apply_geometry_model_list(models, fdg)
    shattering, within, across = out
    geometry_results = {}
    geometry_results['shattering'] = shattering
    geometry_results['within_ccgp'] = within
    geometry_results['across_ccgp'] = across

    history = {}
    out = ma.process_histories(histories, args.model_epochs)
    history['loss'], history['val_loss'] = out 

    all_save = {}
    all_save.update(clustering_results)
    all_save.update(geometry_results)
    all_save.update(history)
    
    maux.save_model_information(models, args.output_folder, args=args,
                                **all_save)
