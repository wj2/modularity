import argparse
import datetime as dt
import os

import tensorflow as tf

import modularity.analysis as ma
import modularity.auxiliary as maux
import modularity.simple as ms

tfk = tf.keras


def create_parser():
    parser = argparse.ArgumentParser(description="fit several modularizers")
    parser.add_argument(
        "--output_subfolder",
        default="rep-sweep",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument(
        "--output_mainfolder",
        default=".",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument(
        "--input_dim", default=6, type=int, help="dimensionality of input"
    )
    parser.add_argument(
        "--rep_dim",
        default=400,
        type=int,
        help="dimensionality of representation layer",
    )

    parser.add_argument(
        "--group_size", default=(3,), type=int, help="size of groups", nargs="+"
    )
    parser.add_argument(
        "--tasks_per_group",
        default=(1,),
        type=int,
        help="number of tasks for each group",
        nargs="+",
    )
    parser.add_argument("--n_groups", default=2, type=int, help="number of groups")
    parser.add_argument(
        "--model_epochs", default=40, type=int, help="epochs to train model for"
    )
    parser.add_argument(
        "--model_batch_size",
        default=100,
        type=int,
        help="batch size for model training",
    )
    parser.add_argument("--n_reps", default=5, type=int, help="number of repeats")
    parser.add_argument("--n_model_train", default=1000, type=int)

    parser.add_argument("--kernel_init_std", default=None, type=float)
    parser.add_argument("--out_kernel_init_std", default=None, type=float)
    parser.add_argument("--group_overlap", default=None, type=int, nargs="+")
    parser.add_argument("--group_width", default=100, type=int)
    parser.add_argument("--activity_reg", default=0, type=float)
    parser.add_argument("--weight_init_const", default=None, type=float)
    parser.add_argument("--input_noise", default=0.01, type=float)
    parser.add_argument("--train_noise", default=0.1, type=float)

    parser.add_argument("--model_layers", nargs="+", default=(), type=int)
    parser.add_argument("--dm_input_mixing", default=0, type=float)
    parser.add_argument("--dm_input_mixing_denom", default=100, type=float)
    parser.add_argument("--remove_last_inp", default=False, action="store_true")
    parser.add_argument("--mixing_order", default=None, type=int)
    parser.add_argument("--only_nonlinear", default=False, action="store_true")
    parser.add_argument("--only_linear", default=False, action="store_true")
    parser.add_argument("--no_early_stopping", default=False, action="store_true")
    parser.add_argument("--track_reps", default=False, action="store_true")
    parser.add_argument("--act_func", default="relu")
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    args.date = dt.datetime.now()

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

    mix_strength = args.dm_input_mixing / args.dm_input_mixing_denom

    models, reps, history, histories = ma.train_n_activity_models(
        mix_strength,
        tasks_per_group,
        input_dim=args.input_dim,
        n_contexts=n_groups,
        only_linear=args.only_linear,
        only_nonlinear=args.only_nonlinear,
        rep_dim=args.rep_dim,
        mixing_order=args.mixing_order,
        group_size=group_size,
        group_maker=group_maker,
        n_reps=args.n_reps,
        group_width=group_width,
        act_reg=act_reg,
        train_noise=sigma,
        input_noise=inp_noise,
        group_overlap=n_overlap,
        const_init=const_init,
        model_batch_size=args.model_batch_size,
        model_epochs=args.model_epochs,
        remove_last_inp=args.remove_last_inp,
        kernel_init_std=args.kernel_init_std,
        out_kernel_init_std=args.out_kernel_init_std,
        model_layers=args.model_layers,
        n_model_train=args.n_model_train,
        early_stopping=not args.no_early_stopping,
        track_reps=args.track_reps,
        act_func=args.act_func,
    )

    all_save = {}
    all_save["reps"] = reps
    all_save.update(history)
    if args.track_reps:
        all_save["reps_dynamic"] = ma.get_tracked_activity(histories, args.model_epochs)

    subfolder = "_".join(
        args.output_subfolder, f"tasks-{tasks_per_group}", f"mix-{mix_strength:.2f}"
    )
    output_folder = os.path.join(args.output_mainfolder, subfolder)
    maux.save_model_information(models, output_folder, args=args, **all_save)
