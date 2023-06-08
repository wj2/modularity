
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

    parser.add_argument('--group_size', default=5, type=int,
                        help='size of groups')
    parser.add_argument('--tasks_per_group', default=10, type=int,
                        help='number of tasks for each group')
    parser.add_argument('--n_groups', default=3, type=int,
                        help='number of groups')
    parser.add_argument('--model_type', default='linear', type=str,
                        help='kind of model')
    parser.add_argument('--model_epochs', default=40, type=int,
                        help='epochs to train model for')
    parser.add_argument('--model_batch_size', default=100, type=int,
                        help='batch size for model training')
    parser.add_argument('--cumu_weight', default=.5, type=float,
                        help='default cumulative weight')
    parser.add_argument('--n_reps', default=5, type=int,
                        help='number of repeats')
    parser.add_argument('--n_model_train', default=1000, type=int)

    parser.add_argument('--kernel_init_std', default=None, type=float)
    parser.add_argument('--group_overlap', default=0, type=int)
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
    parser.add_argument('--reload_image_dataset', default=False, action='store_true')
    parser.add_argument('--no_geometry_analysis', default=False, action='store_true')
    parser.add_argument('--untrained_tasks', default=.5, type=float)
    parser.add_argument('--separate_untrained', default=False, action="store_true")
    parser.add_argument('--novel_tasks', default=1, type=int)
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
    if args.untrained_tasks < 1:
        untrained_tasks = int(np.round(tasks_per_group*args.untrained_tasks))
    else:
        untrained_tasks = int(args.untrained_tasks)

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

    print(args.rep_dim)
    if args.discrete_mixed_input:
        mix_strength = args.dm_input_mixing/args.dm_input_mixing_denom
        fdg = dg.MixedDiscreteDataGenerator(inp_dim, n_units=args.rep_dim,
                                            mix_strength=mix_strength)
    elif args.twod_shape_input:
        twod_file = ('disentangled/datasets/'
                     'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        if args.reload_image_dataset:
            cached_data = None
        else:
            cached_data = pickle.load(
                open('disentangled/datasets/shape_dataset.pkl', 'rb')
            )
        img_resize = (224, 224)
        img_pre_net = args.image_pre_net
        dg_use = dg.TwoDShapeGenerator(twod_file, img_size=img_resize,
                                       max_load=np.inf, convert_color=True,
                                       pre_model=img_pre_net,
                                       cached_data_table=cached_data)
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

    model_kwargs = dict(
        group_size=group_size,
        group_maker=group_maker,
        group_width=group_width,
        use_mixer=True,
        n_tasks=tasks_per_group,
        act_reg_weight=act_reg,
        noise=sigma,
        inp_noise=inp_noise,
        constant_init=const_init,
        n_overlap=n_overlap,
        single_output=True,
        integrate_context=True,
        remove_last_inp=args.remove_last_inp,
        kernel_init=kernel_init,
        out_kernel_init=kernel_init,
        additional_hidden=args.model_layers,
    )

    train_kwargs = dict(
        train_samps=args.n_model_train,
        batch_size=args.model_batch_size,
        train_epochs=args.model_epochs,
        use_early_stopping=False,
    )

    related_context_null = np.zeros((args.n_reps, args.model_epochs))
    rc_null_tasks = np.zeros((args.n_reps, args.model_epochs + 1, untrained_tasks))
    related_context = np.zeros_like(related_context_null)
    rc_tasks = np.zeros_like(rc_null_tasks)

    related_context_all_null = np.zeros((args.n_reps, args.model_epochs))
    rc_all_null_tasks = np.zeros(
        (args.n_reps, args.model_epochs + 1, args.tasks_per_group_tasks)
    )
    related_context_all = np.zeros_like(related_context_all_null)
    rc_all_tasks = np.zeros_like(rc_all_null_tasks)

    new_context_null = np.zeros_like(related_context_null)
    new_context = np.zeros_like(related_context_null)
    nc_null_tasks = np.zeros((args.n_reps, args.model_epochs + 1))
    nc_tasks = np.zeros_like(nc_null_tasks)

    new_task_null = np.zeros_like(related_context_null)
    nt_null_tasks = np.zeros((args.n_reps, args.model_epochs + 1, args.novel_tasks))
    new_task = np.zeros_like(related_context_null)
    nt_tasks = np.zeros_like(nt_null_tasks)

    print(model_type)
    for i in range(args.n_reps):
        (_, hist), (_, hist_null) = ma.new_related_context_training(
            fdg,
            model_type_str=model_type,
            untrained_tasks=untrained_tasks,
            separate_untrained=args.separate_untrained,
            **model_kwargs,
            **train_kwargs,
        )
        related_context[i] = hist.history['val_loss']
        rc_tasks[i] = np.array(hist.history['corr_rate'])[:, :untrained_tasks]
        related_context_null[i] = hist_null.history['val_loss']
        rc_null_tasks[i] = np.array(hist_null.history['corr_rate'])[:, :untrained_tasks]

        (_, hist), (_, hist_null) = ma.new_related_context_training(
            fdg,
            model_type_str=model_type,
            **model_kwargs,
            **train_kwargs,
        )
        related_context_all[i] = hist.history['val_loss']
        rc_all_tasks[i] = np.array(hist.history['corr_rate'])
        related_context_all_null[i] = hist_null.history['val_loss']
        rc_all_null_tasks[i] = np.array(hist_null.history['corr_rate'])

        (_, hist), (_, hist_null) = ma.new_context_training(
            fdg,
            model_type_str=model_type,
            **model_kwargs,
            **train_kwargs,
        )
        new_context[i] = hist.history['val_loss']
        nc_tasks[i] = np.mean(hist.history['corr_rate'], axis=1)
        new_context_null[i] = hist_null.history['val_loss']
        nc_null_tasks[i] = np.mean(hist_null.history['corr_rate'], axis=1)

        (_, hist), (_, hist_null) = ma.new_task_training(
            fdg,
            model_type_str=model_type,
            novel_tasks=args.novel_tasks,
            **model_kwargs,
            **train_kwargs,
        )
        new_task[i] = hist.history['val_loss']
        new_task_null[i] = hist_null.history['val_loss']
        cr_null = np.array(hist_null.history["corr_rate"])
        if len(cr_null.shape) == 1:
            cr_null = np.expand_dims(cr_null, 1)
        nt_null_tasks[i] = cr_null[:, :args.novel_tasks]
        cr = np.array(hist.history["corr_rate"])
        if len(cr.shape) == 1:
            cr = np.expand_dims(cr, 1)
        nt_tasks[i] = cr[:, :args.novel_tasks]

    all_save = {
        'related context': (related_context_all, related_context_all_null),
        'related context tasks': (rc_all_tasks, rc_all_null_tasks),        
        'related context inference': (related_context, related_context_null),
        'related context inference tasks': (rc_tasks, rc_null_tasks),
        'new context': (new_context, new_context_null),
        'new context tasks': (nc_tasks, nc_null_tasks),
        'new task': (new_task, new_task_null),
        'new task tasks': (nt_tasks, nt_null_tasks),
    }

    maux.save_model_information(None, args.output_folder, args=args,
                                **all_save)
