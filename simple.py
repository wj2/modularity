
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as tfhub
import functools as ft
import itertools as it

import scipy.stats as sts
import numpy as np

import general.utility as u
import disentangled.aux as da
import disentangled.regularizer as dr


tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

def xor(x):
    parity = (np.sum(x, axis=1) % 2) == 1
    return parity

def generate_linear_tasks(n_inp, n_tasks=1):
    task = np.random.default_rng().normal(size=(n_tasks, n_inp))
    task = u.make_unit_vector(task)
    return task

def apply_linear_task(x, task=None):
    x_exp = np.expand_dims(x, 1)
    task_exp = np.expand_dims(task, 0)
    return np.sum(task_exp*(x_exp - .5), axis=2) > 0

def generate_coloring(n_g, prob=.5):
    return np.random.default_rng().uniform(size=n_g) <= prob

def generate_many_colorings(n_colorings, n_g, prob=.5):
    rng = np.random.default_rng()
    inds = rng.choice(2**n_g, n_colorings, replace=False)
    out = np.array(list(it.product((0, 1), repeat=n_g)))[inds]
    return out

def apply_coloring(x, coloring=None):
    return np.all(x == coloring, axis=1)

def ident(x, *args, **kwargs):
    return x

def apply_many_colorings(x, colorings=None, merger=np.sum):
    out = np.zeros((len(x), len(colorings)))
    for i, coloring in enumerate(colorings):
        out[:, i] = apply_coloring(x, coloring)
    return merger(out, axis=1, keepdims=True) > 0

def sequential_groups(inp_dims, group_size, *args, **kwargs):
    assert group_size <= inp_dims
    return np.arange(inp_dims).reshape(-1, group_size)

def random_groups(inp_dims, group_size, n_groups, *args, **kwargs):
    rng = np.random.default_rng()
    groups = []
    for i in range(n_groups):
        g_i = rng.choice(inp_dims, group_size, replace=False)
        groups.append(g_i)
    out = np.stack(groups, axis=0)
    return out

def overlap_groups(inp_dims, group_size, n_groups, *args,
                   n_overlap=1, **kwargs):
    rng = np.random.default_rng()
    g_size = (n_groups, group_size - n_overlap)
    samp_dims = group_size*n_groups
    assert samp_dims < inp_dims
    groups = rng.choice(samp_dims, size=g_size, replace=False)
    g_o = rng.choice(range(samp_dims, inp_dims),
                     size=(1, n_overlap), replace=False)
    groups_over = np.tile(g_o, (n_groups, 1))
    groups = np.concatenate((groups, groups_over), axis=1)
    return groups

class Mixer:

    def __init__(self, inp_dims, out_dims, source_distribution=None,
                 bias_const=0, **kwargs):
        if source_distribution is None:
            self.source_distribution = u.MultiBernoulli(.5, inp_dims)
        self.inp_dims = inp_dims
        self.out_dims = out_dims
        self.model = self.make_model(inp_dims, out_dims, bias_const=bias_const,
                                     **kwargs)
        
    def make_model(self, inp, out, layer_type=tfkl.Dense,
                   out_act=tf.nn.relu, noise=.1, inp_noise=.01,
                   bias_const=0, **out_params):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=inp))
        if inp_noise > 0:
            layer_list.append(tfkl.GaussianNoise(inp_noise))
        bi = tfk.initializers.Constant(bias_const)
        layer_list.append(tfkl.Dense(out, activation=out_act,
                                     bias_initializer=bi, **out_params))
        enc = tfk.Sequential(layer_list)
        return enc

    def _compile(self, optimizer=None, loss=None):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        if loss is None:
            loss = tf.losses.BinaryCrossentropy()
        self.model.compile(optimizer, loss)
        self.compiled = True

    def get_representation(self, x):
        return self.model(x)

    def sample_representations(self, n_samples):
        samps = self.source_distribution.rvs((n_samples,))
        reps = self.get_representation(samps)
        return samps, reps    
        
    def representation_dimensionality(self, participation_ratio=True,
                                      sample_size=10**4, **pca_args):
        samples, rep = self.sample_representations(sample_size)
        if participation_ratio:
            out = u.participation_ratio(rep)
        else:
            p = skd.PCA(**pca_args)
            p.fit(rep)
            out = (p.explained_variance_ratio_, p.components_)
        return out
    
class Modularizer:

    def __init__(self, inp_dims, groups=None, group_width=2,
                 group_size=2, group_maker=sequential_groups,
                 n_groups=None, tasks_per_group=1, use_mixer=False,
                 use_dg=None, mixer_out_dims=200, mixer_kwargs=None,
                 use_early_stopping=True, early_stopping_field='val_loss',
                 single_output=False, integrate_context=False,
                 n_overlap=0, **kwargs):
        self.use_early_stopping = use_early_stopping
        self.early_stopping_field = early_stopping_field
        if n_groups is None:
            n_groups = int(np.floor(inp_dims / group_size))
        if groups is None:
            groups = group_maker(inp_dims, group_size, n_groups,
                                 n_overlap=n_overlap)
        if mixer_kwargs is None:
            mixer_kwargs = {} 
        if use_mixer and use_dg is None:
            self.mix = Mixer(inp_dims, mixer_out_dims, **mixer_kwargs)
            self.mix_func = self.mix.get_representation
            inp_net = mixer_out_dims
        elif use_dg is not None:
            self.mix = use_dg
            self.mix_func = use_dg.get_representation
            inp_net = self.mix.output_dim
        else:
            self.mix = None
            self.mix_func = ident
            inp_net = inp_dims
        out_dims = n_groups*tasks_per_group
        self.single_output = single_output
        if single_output and not integrate_context:
            inp_net = inp_net + n_groups
            out_dims = tasks_per_group
        elif single_output and integrate_context:
            inp_net = inp_net
            out_dims = tasks_per_group
            inp_dims = inp_dims - n_groups
        self.integrate_context = integrate_context
        self.inp_net = inp_net
        self.n_tasks_per_group = tasks_per_group
        self.n_groups = n_groups        
        self.group_size = group_size
        self.inp_dims = inp_dims
        self.out_dims = out_dims
        self.hidden_dims = int(round(len(groups)*group_width))
        self.out_group_labels = np.concatenate(list((i,)*tasks_per_group
                                                    for i in range(n_groups)))

        out = self.make_model(inp_net, self.hidden_dims, self.out_dims,
                              **kwargs)
        model, rep_model = out
        self.rep_model = rep_model
        self.model = model
        self.groups = groups
        self.rng = np.random.default_rng()
        self.compiled = False
    
    def make_model(self, inp, hidden, out, act_func=tf.nn.relu,
                   layer_type=tfkl.Dense, out_act=tf.nn.sigmoid,
                   noise=.1, inp_noise=.01,
                   kernel_reg_type=tfk.regularizers.L2,
                   kernel_reg_weight=0,
                   act_reg_type=tfk.regularizers.l2,
                   act_reg_weight=0,
                   constant_init=None,
                   **layer_params):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=inp))
        if inp_noise > 0:
            layer_list.append(tfkl.GaussianNoise(inp_noise))
        if kernel_reg_weight > 0:
            kernel_reg = kernel_reg_type(kernel_reg_weight)
        else:
            kernel_reg = None
        if constant_init is not None:
            k_init = tfk.initializers.Constant(constant_init)
        else:
            k_init = 'glorot_uniform'
        if act_reg_weight > 0:
            act_reg = act_reg_type(act_reg_weight)
        else:
            act_reg = None
        lh = layer_type(hidden, activation=act_func,
                        kernel_regularizer=kernel_reg,
                        activity_regularizer=act_reg,
                        kernel_initializer=k_init,
                        **layer_params)
        layer_list.append(lh)
        if noise > 0:
            layer_list.append(tfkl.GaussianNoise(noise))

        rep = tfk.Sequential(layer_list)
        layer_list.append(tfkl.Dense(out, activation=out_act,
                                     kernel_regularizer=kernel_reg))
        enc = tfk.Sequential(layer_list)
        return enc, rep

    def get_representation(self, stim, group=None):
        rep = self.rep_model(stim)
        return rep

    def sample_reps(self, n_samps=1000, context=None):
        if context is not None:
            group_inds = np.ones(n_samps, dtype=int)*context
        out = self.get_x_true(n_train=n_samps, group_inds=group_inds)
        x, true, targ = out
        rep = self.get_representation(x)
        return true, x, rep 

    def _compile(self, optimizer=None, loss=None):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        if loss is None:
            loss = tf.losses.BinaryCrossentropy()
        self.model.compile(optimizer, loss)
        self.compiled = True

    def _generate_target_so(self, xs, group_inds):
        ys = np.zeros((self.n_groups, len(xs), self.out_dims))
        for i, g in enumerate(self.groups):
            g_inp = xs[:, g]
            out = self.group_func[i](g_inp)
            if len(out.shape) == 1:
                out = np.expand_dims(out, axis=1)
            ys[i] = out
        ys = ys[group_inds]
        return ys            
        
    def generate_target(self, xs, group_inds=None):
        ys = np.zeros((self.n_groups, len(xs), self.n_tasks_per_group))
        for i, g in enumerate(self.groups):
            g_inp = xs[:, g]
            out = self.group_func[i](g_inp)
            if len(out.shape) == 1:
                out = np.expand_dims(out, axis=1)
            ys[i] = out
        if self.single_output and group_inds is None:
            raise IOError('need to provide group_inds if single_output is '
                          'True')
        elif self.single_output:
            trl_inds = np.arange(len(xs))
            ys = ys[group_inds, trl_inds]
        else:
            ys = np.concatenate(ys, axis=1)
        return ys

    def sample_stim(self, n_samps):
        return self.rng.uniform(0, 1, size=(n_samps, self.inp_dims)) < .5

    def get_x_true(self, x=None, true=None, n_train=10**5, group_inds=None):
        if true is None and x is not None:
            raise IOError('need ground truth x')
        if true is None:
            true = self.sample_stim(n_train)
        if group_inds is None and self.single_output:
            group_inds = self.rng.choice(self.n_groups,
                                         n_train)
        if self.single_output:
            to_add = np.zeros((n_train, self.n_groups))
            trl_inds = np.arange(n_train)
            to_add[trl_inds, group_inds] = 1
        if self.single_output and self.integrate_context:
            true = np.concatenate((true, to_add), axis=1)
        if x is None:
            x = self.mix_func(true)
        if self.single_output and not self.integrate_context:
            x = np.concatenate((x, to_add), axis=1)
        targ = self.generate_target(true, group_inds=group_inds)
        return x, true, targ
    
    def fit(self, train_x=None, train_true=None, eval_x=None, eval_true=None,
            n_train=2*10**5, epochs=15, batch_size=100, n_val=10**3, **kwargs): 
        if not self.compiled:
            self._compile()

        train_x, train_true, train_targ = self.get_x_true(train_x, train_true,
                                                          n_train)
        eval_x, eval_true, eval_targ = self.get_x_true(eval_x, eval_true,
                                                       n_val)

        eval_set = (eval_x, eval_targ)

        if self.use_early_stopping:
            cb = tfk.callbacks.EarlyStopping(monitor=self.early_stopping_field,
                                             mode='min', patience=2)
            curr_cb = kwargs.get('callbacks', [])
            curr_cb.append(cb)
            kwargs['callbacks'] = curr_cb


        out = self.model.fit(x=train_x, y=train_targ, epochs=epochs,
                             validation_data=eval_set, batch_size=batch_size,
                             **kwargs)
        return out

class XORModularizer(Modularizer):

    def __init__(self, *args, group_func=xor, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_func = (group_func,)*len(self.groups)

class ColoringModularizer(Modularizer):

    def _make_group_func(self, n_colorings, n_g, tasks_per_group=1, merger=np.sum):
        funcs = []
        for i in range(tasks_per_group):
            cols = generate_many_colorings(n_colorings, n_g)
            funcs.append(ft.partial(apply_many_colorings, colorings=cols,
                                    merger=merger))
        return lambda x: np.concatenate(list(f(x) for f in funcs), axis=1)
    
    def __init__(self, *args, n_colorings=None, tasks_per_group=1,
                 task_merger=np.sum, **kwargs):
        super().__init__(*args, tasks_per_group=tasks_per_group, **kwargs)
        if n_colorings is None:
            n_colorings = 2**(self.group_size - 1)
        group_func = []
        for g in self.groups:
            group_func.append(
                self._make_group_func(n_colorings, len(g),
                                      tasks_per_group=tasks_per_group,
                                      merger=task_merger))
        self.group_func = tuple(group_func)

class IdentityModularizer:

    def __init__(self, inp_dims, group_maker=sequential_groups, hidden_dims=100,
                 group_size=2, n_groups=None, **kwargs):
        self.hidden_dims = hidden_dims
        if n_groups is None:
            n_groups = int(np.floor(inp_dims / group_size))
        if group_size is None:
            group_size = int(np.floor(inp_dims / n_groups))
        self.groups = group_maker(inp_dims, group_size, n_groups)

    def get_representation(self, stim, group=None):
        return stim
        
class LinearModularizer(Modularizer):

    def _make_linear_task_func(self, n_g, n_tasks=1):
        task = generate_linear_tasks(n_g, n_tasks=n_tasks)
        return ft.partial(apply_linear_task, task=task)

    def __init__(self, *args, tasks_per_group=1, **kwargs):
        super().__init__(*args, tasks_per_group=tasks_per_group, **kwargs)
        group_func = []
        for g in self.groups:
            group_func.append(
                self._make_linear_task_func(len(g), n_tasks=tasks_per_group))
        self.group_func = tuple(group_func)
