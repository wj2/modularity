
import disentangled.data_generation as dg
import modularity.simple as ms


def train_sequential_contexts(
    n_decision_lv,
    n_cons,
    n_irrel_lv=3,
    mixing=0,
    mixing_order=None,
    train_samps=1000,
    track_samps=1000,
    epochs=10,
    verbose=False,
    share_n_contexts=0,
    share_pairs=None,
    con_seq=None,
    **kwargs,
):
    if con_seq is None:
        con_seq = range(n_cons)
    n_lvs = n_decision_lv + n_cons + n_irrel_lv
    fdg = dg.MixedDiscreteDataGenerator(
        n_lvs, mix_strength=mixing, mixing_order=mixing_order
    )
    if share_n_contexts > 1 and share_pairs is None:
        share_pairs = list(zip(range(share_n_contexts), (0,) * n_cons))
    mod = ms.make_modularizer(fdg, n_groups=n_cons, share_pairs=share_pairs, **kwargs)
    hist_seq = []
    loss_tracking = {}
    for i in range(n_cons):
        inp_r, _, targ_r = mod.get_x_true(n_train=track_samps, group_inds=i)
        loss_tracking[i] = (inp_r, targ_r)
    for i in con_seq:
        hist_i = mod.fit(
            n_train=train_samps,
            only_groups=(i,),
            track_corr=loss_tracking,
            epochs=epochs,
            verbose=verbose,
        )
        hist_seq.append(hist_i)
    return mod, hist_seq
