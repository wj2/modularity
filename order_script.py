
import argparse
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime

import modularity.analysis as ma

tfk = tf.keras


def create_parser():
    parser = argparse.ArgumentParser(description='fit modularizers and quantify order')
    parser.add_argument('-o', '--output_file', default='order_0000.pkl', type=str,
                        help='file to save the output in')
    parser.add_argument("--mixing_strength", default=0, type=float)
    parser.add_argument("--mixing_denominator", default=100, type=float)
    parser.add_argument("--group_size", default=3, type=int)
    parser.add_argument("--overlap", default=None, type=int)
    parser.add_argument("--weight_kernel_init", default=.001, type=float)
    parser.add_argument("--weight_kernel_denom", default=1, type=float)
    parser.add_argument("--training_epochs", default=20, type=int)
    parser.add_argument("--no_axis_tasks", default=True, action="store_false")
    parser.add_argument("--eval_intermediate", default=False, action="store_true")
    return parser


metric_methods = {
    'gm': ma.quantify_activity_clusters,
    'l2': ma.quantify_model_l2,
    'l1': ma.quantify_model_l1,
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

    mixing_strength = args.mixing_strength / args.mixing_denominator
    weight_kernel_init = args.weight_kernel_init / args.weight_kernel_denom
    mddg, model, hist = ma.train_controlled_model(
        args.group_size,
        mixing_strength,
        n_overlap=args.group_size,
        axis_tasks=args.no_axis_tasks,
        kernel_init=weight_kernel_init,
        train_epochs=args.training_epochs,
    )

    orders, out_lms, out_scores = ma.analyze_model_orders(model)

    models = np.array([[model]])
    metric_results = {}
    for cm, func in metric_methods.items():
        clust = ma.apply_act_clusters_list(
            models,
            func,
            eval_layers=args.eval_intermediate,
        )
        metric_results[cm] = clust

    args_dict = vars(args)
    save_dict = {
        "args": args_dict,
        "orders": orders,
        "out_models": out_lms,
        "out_scores": out_scores,
        "metrics": metric_results,
    }

    pickle.dump(save_dict, open(args.output_file, "wb"))
