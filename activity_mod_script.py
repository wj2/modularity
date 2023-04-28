import argparse
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft
import tensorflow as tf
from datetime import datetime

import general.utility as u
import disentangled.data_generation as dg
import disentangled.characterization as dc

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
    parser.add_argument('--model_layers', nargs='+', default=(),
                        type=int)
    parser.add_argument('--ccgp_n_train', default=2, type=int)                        
    parser.add_argument('--ccgp_fix_features', default=-1, type=int)
    parser.add_argument('--continuous_input', default=False,
                        action='store_true')
    parser.add_argument('--twod_shape_input', default=False,
                        action='store_true')
    img_net_address = ('https://tfhub.dev/google/imagenet/'
                       'mobilenet_v3_small_100_224/feature_vector/5')
    parser.add_argument('--image_pre_net', default=img_net_address,
                        type=str)
    parser.add_argument('--discrete_mixed_input', default=False,
                        action='store_true')
    parser.add_argument('--dm_input_mixing', default=.5, type=float)
    parser.add_argument('--dm_input_mixing_denom', default=1, type=float)
    parser.add_argument('--remove_last_inp', default=False,
                        action='store_true')
    parser.add_argument('--eval_intermediate', default=False, action='store_true')
    return parser


model_dict = {
    'xor': ms.XORModularizer,
    'coloring': ms.ColoringModularizer,
    'linear': ms.LinearModularizer,
    'linear_continuous': ms.LinearContinuousModularizer,
}
metric_methods = {
    'gm': ma.quantify_activity_clusters,
    'l2': ma.quantify_model_l2,
    'l1': ma.quantify_model_l1,
    # 'max_corr':ma.quantify_max_corr_clusters,
    # 'within_max_corr_ablation':ma.within_max_corr_ablation,
    # 'across_max_corr_ablation':ma.across_max_corr_ablation,
    'within_act_ablation': ma.within_act_ablation,
    'across_act_ablation': ma.across_act_ablation,
    'within_graph_ablation': ma.within_graph_ablation,
    'across_graph_ablation': ma.across_graph_ablation,
    'diff_graph_ablation': ma.diff_graph_ablation,
    'diff_act_ablation': ma.diff_act_ablation,
    'model_frac': ma.compute_frac_contextual,
    'fdg_frac': ma.compute_fdg_frac_contextual,
}

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
    print('id', inp_dim)
    if args.continuous_input:
        source_distr = sts.multivariate_normal([0]*inp_dim, 1)
        model_type = args.model_type + '_continuous'
    else:
        source_distr = u.MultiBernoulli(.5, inp_dim)
        model_type = args.model_type
        
    if args.kernel_init_std is not None:
        kernel_init = args.kernel_init_std
    else:
        kernel_init = None

    print(inp_dim)
    print(args.rep_dim)
    if args.discrete_mixed_input:
        mix_strength = args.dm_input_mixing/args.dm_input_mixing_denom
        fdg = dg.MixedDiscreteDataGenerator(inp_dim, n_units=args.rep_dim,
                                            mix_strength=mix_strength)
    elif args.twod_shape_input:
        twod_file = ('disentangled/datasets/'
                     'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        img_resize = (224, 224)
        img_pre_net = args.img_pre_net
        dg_use = dg.TwoDShapeGenerator(twod_file, img_size=img_resize,
                                       max_load=100, convert_color=True,
                                       pre_model=img_pre_net)
        true_inp_dim = dg_use.input_dim
        no_learn_lvs = np.array([True, False, True, False, False])
        compute_train_lvs = True

        fdg = ms.ImageDGWrapper(dg_use, ~no_learn_lvs, 'shape', 0)
    else:
        fdg = dg.FunctionalDataGenerator(inp_dim, (300,), args.rep_dim,
                                         source_distribution=source_distr,
                                         use_pr_reg=True,
                                         kernel_init=args.fdg_weight_init)
        fdg.fit(epochs=args.fdg_epochs, verbose=False,
                batch_size=args.dg_batch_size)
    print('dg dim', fdg.representation_dimensionality(participation_ratio=True))

    m_constructor = model_dict[model_type]
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
        epochs=args.model_epochs,
        remove_last_inp=args.remove_last_inp,
        kernel_init=kernel_init,
        out_kernel_init=kernel_init,
        additional_hidden=args.model_layers,
    )
    models, histories = out

    # these default metrics are not implemented for intermediate layers
    non_intermediate_metrics = (
        "within_graph_ablation",
        "across_graph_ablation",
        "diff_graph_ablation",
    )
    if args.eval_intermediate:
        for metric in non_intermediate_metrics:
            metric_methods.pop(metric)

    metric_results = {}
    for cm, func in metric_methods.items():
        clust = ma.apply_act_clusters_list(
            models,
            func,
            eval_layers=args.eval_intermediate,
        )
        metric_results[cm] = clust

    if args.ccgp_fix_features is None:
        fix_feats = group_size - 1
    else:
        fix_feats = args.ccgp_fix_features
    out = ma.apply_geometry_model_list(
        models,
        fdg,
        n_train=args.ccgp_n_train,
        fix_features=fix_feats,
        eval_layers=args.eval_intermediate,
    )
    shattering, within, across = out
    geometry_results = {}
    geometry_results['shattering'] = shattering
    geometry_results['within_ccgp'] = within
    geometry_results['across_ccgp'] = across

    history = ma.process_histories(histories, args.model_epochs)

    all_save = {}
    all_save.update(metric_results)
    all_save.update(geometry_results)
    all_save.update(history)
    
    maux.save_model_information(models, args.output_folder, args=args,
                                **all_save)



