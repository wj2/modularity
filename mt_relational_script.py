import argparse
import numpy as np
import pickle
import os
from datetime import datetime

import modularity.simple as ms


def create_parser():
    parser = argparse.ArgumentParser(description='fit several MT modularizers')
    parser.add_argument('-o', '--output_folder', default='.', type=str,
                        help='folder to save the output in')
    parser.add_argument("--output_template", default="mt-{t}_rw{r}_nms{m}_{jobid}.pkl")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--jobid", default="0000")    
    parser.add_argument("--relational_weight", default=0, type=float)
    parser.add_argument("--mixing", default=0, type=float)
    parser.add_argument("--relational_weight_denom", default=100, type=float)
    parser.add_argument("--mixing_denom", default=100, type=float)
    parser.add_argument("--n_reps", default=10, type=int)
    parser.add_argument("--batch_size", default=20, type=int)
    parser.add_argument("--n_values", default=3, type=int)
    parser.add_argument("--reg_strength", default=0, type=float)
    parser.add_argument("--weight_scale", default=None, type=float)
    parser.add_argument("--n_train", default=500, type=int)
    parser.add_argument("--include_history", default=0, type=int)
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    args.date = datetime.now()

    key_length = {"tracked_activity": 4}
    skip_keys = ("tracked_activity",)
    
    for i in range(args.n_reps):
        fdg, (m_same, h_same), (m_flip, h_flip) = ms.make_and_train_mt_model_set(
            args.mixing / args.mixing_denom,
            relational_weight=args.relational_weight / args.relational_weight_denom,
            n_values=args.n_values,
            batch_size=args.batch_size,
            act_reg_weight=args.reg_strength,
            kernel_init=args.weight_scale,
        )
        for sk in skip_keys:
            h_same.history.pop(sk)
            h_flip.history.pop(sk)
        if i == 0:
            out_same = {}
            out_flip = {}
            for k, v in h_same.history.items():
                kl = key_length.get(k, None)
                if kl is not None:
                    v_list_same = []
                    v_list_flip = []
                    for j in range(kl):
                        v_list_same.append(np.zeros((args.n_reps,) + v[j].shape))
                        v_list_flip.append(np.zeros((args.n_reps,) + v[j].shape))
                    out_same[k] = v_list_same
                    out_flip[k] = v_list_flip
                else:
                    out_same[k] = np.zeros((args.n_reps,) + np.array(v).shape)
                    out_flip[k] = np.zeros((args.n_reps,) + np.array(v).shape)
        for k, v in h_same.history.items():
            kl = key_length.get(k, None)
            v_flip = h_flip.history[k]
            if kl is None:
                out_same[k][i] = np.array(v)
                out_flip[k][i] = np.array(v_flip)
            else:
                for j in range(kl):
                    out_same[k][j][i] = np.array(v[j])
                    out_flip[k][j][i] = np.array(v_flip[j])

    fname = args.output_template.format(
        r=args.relational_weight, m=args.mixing, jobid=args.jobid, t=args.tag,
    )
    path = os.path.join(args.output_folder, fname)
    pickle.dump((out_same, out_flip), open(path, "wb"))
