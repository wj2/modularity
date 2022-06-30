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
    
    parser.add_argument('--group_size', default=(5,), type=int,
                        help='size of groups', nargs='+')
    parser.add_argument('--tasks_per_group', default=(10,), type=int,
                        help='number of tasks for each group', nargs='+')
    parser.add_argument('--n_groups', default=3, type=int,
                        help='number of groups')
    parser.add_argument('--model_type', default='coloring', type=str,
                        help='kind of model')
    parser.add_argument('--model_epochs', default=40, type=int,
                        help='epochs to train model for')
    parser.add_argument('--model_batch_size', default=100, type=int,
                        help='batch size for model training')
    parser.add_argument('--cumu_weight', default=.5, type=float,
                        help='default cumulative weight')
    parser.add_argument('--n_reps', default=5, type=int,
                        help='number of repeats')

    parser.add_argument('--kernel_init_std', default=None, type=float)
    parser.add_argument('--group_overlap', default=(0,), type=int, nargs='+')
    parser.add_argument('--group_width', default=50, type=int)
    parser.add_argument('--activity_reg', default=.01, type=float)
    parser.add_argument('--weight_init_const', default=None, type=float)
    parser.add_argument('--input_noise', default=.01, type=float)
    parser.add_argument('--train_noise', default=.1, type=float)
    
    parser.add_argument('--dg_batch_size', default=100, type=int)
    parser.add_argument('--brim_threshold', default=1.5, type=float)
    parser.add_argument('--fdg_weight_init', default=None, type=float)
    parser.add_argument('--fdg_layers', nargs='+', default=(300,),
                        type=int)
    parser.add_argument('--ccgp_n_train', default=2, type=int)                        
    parser.add_argument('--ccgp_fix_features', default=-1, type=int)
    return parser

model_dict = {'xor':ms.XORModularizer,
              'coloring':ms.ColoringModularizer,
              'linear':ms.LinearModularizer}
metric_methods = {'gm':ma.quantify_activity_clusters,
                  'l2':ma.quantify_model_l2,
                  'l1':ma.quantify_model_l1}

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()

    group_size = args.group_size
    n_overlap = args.group_overlap
    group_maker = ms.overlap_groups

    tasks_per_group = args.tasks_per_group
    n_groups = args.n_groups
    sigma = args.train_noise
    inp_noise = args.input_noise
    const_init = args.weight_init_const
    act_reg = args.activity_reg
    group_width = args.group_width

    inp_dim = args.input_dim + n_groups
    source_distr = u.MultiBernoulli(.5, inp_dim)
    if args.kernel_init_std is not None:
        kernel_init =  tfk.initializers.RandomNormal(stddev=args.kernel_init_std)
    else:
        kernel_init = None

    fdg = dg.FunctionalDataGenerator(inp_dim, (300,), args.rep_dim,
                                     source_distribution=source_distr, 
                                     use_pr_reg=True, kernel_init=kernel_init)
    fdg.fit(epochs=args.fdg_epochs, verbose=False,
            batch_size=args.dg_batch_size)
    print('dg dim', fdg.representation_dimensionality(participation_ratio=True))

    m_constructor = model_dict[args.model_type]
    out = ma.train_variable_models(
        group_size,
        tasks_per_group,
        group_maker,
        m_constructor,
        n_reps=args.n_reps,
        fdg=fdg,
        n_groups=n_groups,
        group_width=group_width,
        act_reg_weight=act_reg,
        noise=sigma,
        constant_init=const_init,
        inp_noise=inp_noise,
        n_overlap=n_overlap,
        single_output=True,
        integrate_context=True,
        batch_size=args.model_batch_size,
        epochs=args.model_epochs)
    models, histories = out

    metric_results = {}
    for cm, func in metric_methods.items():
        clust = ma.apply_act_clusters_list(models, func)
        metric_results[cm] = clust

    if args.ccgp_fix_features is None:
        fix_feats = group_size - 1
    else:
        fix_feats = args.ccgp_fix_features
    out = ma.apply_geometry_model_list(models, fdg, n_train=args.ccgp_n_train,
                                       fix_features=fix_feats)
    shattering, within, across = out
    geometry_results = {}
    geometry_results['shattering'] = shattering
    geometry_results['within_ccgp'] = within
    geometry_results['across_ccgp'] = across

    history = {}
    out = ma.process_histories(histories, args.model_epochs)
    history['loss'], history['val_loss'] = out 

    all_save = {}
    all_save.update(metric_results)
    all_save.update(geometry_results)
    all_save.update(history)
    
    maux.save_model_information(models, args.output_folder, args=args,
                                **all_save)



