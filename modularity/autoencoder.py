import numpy as np
import matplotlib.pyplot as plt

import sequential_learning.theory.input_output as slio
import general.torch.feedforward as gtf
import general.utility as u
import general.plotting as gpl


class OutputAutoSampler:
    def __init__(
        self,
        n_tasks,
        rel_dims,
        con_dims,
        share_tasks=None,
        share_task_dims=True,
        **ddg_kwargs,
    ):
        self.gen = slio.DiscreteMixedContext(
            rel_dims + con_dims,
            context_dims=con_dims,
            **ddg_kwargs,
        )
        self.tasks = slio.make_task_set(
            n_tasks,
            rel_dims,
            con_dims,
            share_tasks=share_tasks,
            share_task_dims=share_task_dims,
        )
        self.n_contexts = len(con_dims) if u.check_list(con_dims) else con_dims

    def sample_xy_pairs(self, n_samps=1000, contexts=None, add_noise=False):
        stim, X = self.gen.sample_reps(
            n_samps=n_samps, contexts=contexts, add_noise=add_noise
        )

        y = self.tasks(stim)
        return X, y

    def sample_stim_rep_targ(self, n_samps=1000, **kwargs):
        stim, inp_rep = self.gen.sample_reps(n_samps=n_samps, **kwargs)
        targs = self.tasks(stim)
        return stim, inp_rep, targs

    def sample_yy_pairs(self, n_samps=1000, **kwargs):
        _, y = self.sample_xy_pairs(n_samps, **kwargs)
        return y, y

    def get_all_pairs(self):
        stim, reps = self.gen.get_all_stim()
        targ = self.tasks(stim)
        return stim, reps, targ

    def generator(self, batch_size=100, max_samples=10**8, **kwargs):
        for i in range(max_samples):
            yield self.sample_yy_pairs(batch_size, **kwargs)


def train_autoencoder(
    n_tasks,
    rel_dims,
    con_dims,
    hidden_units=100,
    batch_size=10,
    num_steps=500,
    track_samples=1000,
    add_noise=True,
    sigma=0.1,
    share_tasks=None,
    share_task_dims=True,
    l2_reg=0,
    l1_reg=0,
):
    sampler = OutputAutoSampler(
        n_tasks,
        rel_dims,
        con_dims,
        sigma=sigma,
        share_tasks=share_tasks,
        share_task_dims=share_task_dims,
    )

    inp, targs = sampler.sample_xy_pairs(2)
    net = gtf.AutoEncoder(n_tasks, (hidden_units,))

    out = net.fit_generator(
        sampler.generator(batch_size, add_noise=add_noise),
        l2_reg=l2_reg,
        l1_reg=l1_reg,
        num_steps=num_steps,
    )
    loss_tracker = out["loss"]
    out_dict = {
        "loss": loss_tracker,
        "network": net,
        "sampler": sampler,
    }
    return out_dict


@gpl.ax_adder()
def plot_autoencoder_contexts(sampler, net, ax=None, n_contexts=2, **kwargs):
    _, _, targ0 = sampler.sample_stim_rep_targ(contexts=(0,))
    _, _, targ1 = sampler.sample_stim_rep_targ(contexts=(1,))

    r0 = np.mean(net.get_representation(targ0).detach().numpy(), axis=0)
    r1 = np.mean(net.get_representation(targ1).detach().numpy(), axis=0)

    ax.plot(r0, r1, "o", **kwargs)
    ax.set_aspect("equal")


def train_and_plot_autoencoders(l1s, *args, axs=None, ms=1, fwid=3, **kwargs):
    if axs is None:
        f, axs = plt.subplots(
            1, len(l1s), figsize=(fwid * len(l1s), fwid), sharey="all", sharex="all"
        )
    for i, l1 in enumerate(l1s):
        out = train_autoencoder(
            *args,
            **kwargs,
            l1_reg=l1,
        )
        sampler = out["sampler"]
        net = out["network"]
        plot_autoencoder_contexts(sampler, net, ax=axs[i], ms=ms)
        axs[i].set_xlabel("context 1 activity")
        axs[i].set_ylabel("context 2 activity")
        gpl.clean_plot(axs[i], i)
        axs[i].set_title("L1 = {:.3f}".format(l1))
    return axs
