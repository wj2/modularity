import argparse
import numpy as np
import pickle
import os
from datetime import datetime

import modularity.simple as ms


def create_parser():
    parser = argparse.ArgumentParser(description="fit several MT modularizers")
    parser.add_argument(
        "-o",
        "--output_folder",
        default=".",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument(
        "--output_template", default="mt-seq-{t}_rw{r}_nms{m}_{jobid}.pkl"
    )
    parser.add_argument("--tag", default="default")
    parser.add_argument("--jobid", default="0000")
    parser.add_argument("--relational_weight", default=0, type=float)
    parser.add_argument("--mixing", default=0, type=float)
    parser.add_argument("--relational_weight_denom", default=100, type=float)
    parser.add_argument("--mixing_denom", default=100, type=float)
    parser.add_argument("--n_reps", default=10, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--n_values", default=3, type=int)
    parser.add_argument("--n_contexts", default=2, type=int)
    parser.add_argument("--reg_strength", default=0, type=float)
    parser.add_argument("--weight_scale", default=None, type=float)
    parser.add_argument("--n_train", default=500, type=int)
    parser.add_argument("--include_history", default=0, type=int)
    parser.add_argument("--train_epochs", default=10, type=int)
    parser.add_argument("--use_relational_history", default=False, action="store_true")
    parser.add_argument("--use_nonexhaustive", default=False, action="store_true")
    parser.add_argument("--mixing_order", default=None, type=int)
    parser.add_argument("--additional_hidden", default=(), nargs="+", type=int)
    parser.add_argument("--n_overlap", default=0, type=int)
    parser.add_argument("--corr", default=0, type=float)
    parser.add_argument("--corr_denom", default=100, type=float)
    parser.add_argument(
        "--use_overlapping_variables", default=False, action="store_true"
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()

    key_length = {"tracked_activity": 4}
    skip_keys = ("tracked_activity",)
    if args.corr > 0 and not args.use_overlapping_variables:
        corr_groups = {
            (0, 2): args.corr / args.corr_denom,
            (1, 3): args.corr / args.corr_denom,
        }
    else:
        corr_groups = None

    relational = args.relational_weight > 0

    out_same = {}
    out_flip = {}
    for j in range(args.n_reps):
        out = ms.make_and_train_mt_model_set_sequential(
            args.mixing / args.mixing_denom,
            relational_weight=args.relational_weight / args.relational_weight_denom,
            n_values=args.n_values,
            relational=relational,
            batch_size=args.batch_size,
            act_reg_weight=args.reg_strength,
            kernel_init=args.weight_scale,
            n_train=args.n_train,
            include_history=args.include_history,
            train_epochs=args.train_epochs,
            relational_history=args.use_relational_history,
            use_nonexhaustive=args.use_nonexhaustive,
            n_cons=args.n_contexts,
            mixing_order=args.mixing_order,
            additional_hidden=args.additional_hidden,
            overlapping_variables=args.use_overlapping_variables,
            track_rep_sim=True,
            corr_groups=corr_groups,
        )
        fdg, (m_same, h_same), (m_flip, h_flip) = out
        for i, h_same_i in enumerate(h_same):
            for sk in skip_keys:
                h_same_i.history.pop(sk, None)
                h_flip[i].history.pop(sk, None)

            for k, v in h_same_i.history.items():
                l_ = out_same.get((i, k), [])
                l_.append(v)
                out_same[(i, k)] = l_
            for k, v in h_flip[i].history.items():
                l_ = out_flip.get((i, k), [])
                l_.append(v)
                out_flip[(i, k)] = l_

    for k, v in out_same.items():
        out_same[k] = np.stack(v, axis=0)
    for k, v in out_flip.items():
        out_flip[k] = np.stack(v, axis=0)

    fname = args.output_template.format(
        r=args.relational_weight,
        m=args.mixing,
        jobid=args.jobid,
        t=args.tag,
    )
    path = os.path.join(args.output_folder, fname)
    out_dict = {
        "args": vars(args),
        "same": out_same,
        "flip": out_flip,
    }
    pickle.dump(out_dict, open(path, "wb"))
