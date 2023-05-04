import tensorflow as tf
import tensorflow_probability as tfp
import functools as ft
import itertools as it

import sklearn.preprocessing as skp
import sklearn.decomposition as skd
import numpy as np

import general.utility as u
import disentangled.aux as da
import disentangled.disentanglers as dd
import disentangled.data_generation as dg

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class ModularizerIntermediateLayers(dd.GenericIntermediateLayers):
    def __init__(self, *args, hidden_dims=None, **kwargs):
        self.hidden_dims = hidden_dims

        super().__init__(*args, **kwargs)


def mse_nanloss(label, prediction):
    nan_mask = tf.math.logical_not(tf.math.is_nan(label))
    label = tf.boolean_mask(label, nan_mask)
    prediction = tf.boolean_mask(prediction, nan_mask)
    mult = tf.square(prediction - label)
    mse_nanloss = tf.reduce_mean(mult)
    return mse_nanloss


def binary_crossentropy_nan(label, prediction):
    nan_mask = tf.math.logical_not(tf.math.is_nan(label))
    label = tf.boolean_mask(label, nan_mask)
    prediction = tf.boolean_mask(prediction, nan_mask)
    bcel = tf.keras.losses.binary_crossentropy(label, prediction)
    return bcel


def xor(x):
    parity = (np.sum(x, axis=1) % 2) == 1
    return parity


def generate_linear_tasks(n_inp, n_tasks=1, intercept_var=0, axis_tasks=False):
    rng = np.random.default_rng()
    if axis_tasks:
        inds = rng.choice(n_inp, size=(n_tasks,))
        task = np.zeros((n_tasks, n_inp))
        for i in range(n_tasks):
            task[i, inds[i]] = 1
        task = task * rng.choice((-1, 1), size=(n_tasks, 1))
    else:
        task = rng.normal(size=(n_tasks, n_inp))
        task = u.make_unit_vector(task)
    if intercept_var > 0:
        intercepts = rng.normal(0, np.sqrt(intercept_var), size=(n_tasks, 1))
    else:
        intercepts = np.zeros((n_tasks, 1))
    return task, intercepts


def apply_continuous_task(x, task=None):
    out = np.stack(list(t(x) for t in task), axis=1)
    return out


def apply_linear_task(x, task=None, intercept=0, center=0.5, renorm=False):
    x_exp = np.expand_dims(x, 1)
    task_exp = np.expand_dims(task, 0)
    bools = np.sum(task_exp * (x_exp - center) + intercept, axis=2) > 0
    if renorm:
        bools = 2 * (bools - 0.5)
    return bools


def generate_coloring(n_g, prob=0.5):
    return np.random.default_rng().uniform(size=n_g) <= prob


def generate_many_colorings(n_colorings, n_g, prob=0.5):
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


def overlap_groups(inp_dims, group_size, n_groups, *args, n_overlap=1, **kwargs):
    rng = np.random.default_rng()
    g_size = (n_groups, group_size - n_overlap)
    samp_dims = n_groups * g_size[1]
    assert g_size[1] <= inp_dims
    groups = rng.choice(samp_dims, size=g_size, replace=False)
    g_o = rng.choice(range(samp_dims, inp_dims), size=(1, n_overlap), replace=False)
    groups_over = np.tile(g_o, (n_groups, 1))
    groups = np.concatenate((groups, groups_over), axis=1).astype(int)
    return groups


class Mixer:
    def __init__(
        self, inp_dims, out_dims, source_distribution=None, bias_const=0, **kwargs
    ):
        if source_distribution is None:
            self.source_distribution = u.MultiBernoulli(0.5, inp_dims)
        self.inp_dims = inp_dims
        self.out_dims = out_dims
        self.model = self.make_model(
            inp_dims, out_dims, bias_const=bias_const, **kwargs
        )

    def make_model(
        self,
        inp,
        out,
        layer_type=tfkl.Dense,
        out_act=tf.nn.relu,
        noise=0.1,
        inp_noise=0.01,
        bias_const=0,
        **out_params
    ):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=inp))
        if inp_noise > 0:
            layer_list.append(tfkl.GaussianNoise(inp_noise))
        bi = tfk.initializers.Constant(bias_const)
        layer_list.append(
            tfkl.Dense(out, activation=out_act, bias_initializer=bi, **out_params)
        )
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

    def representation_dimensionality(
        self, participation_ratio=True, sample_size=10**4, **pca_args
    ):
        samples, rep = self.sample_representations(sample_size)
        if participation_ratio:
            out = u.participation_ratio(rep)
        else:
            p = skd.PCA(**pca_args)
            p.fit(rep)
            out = (p.explained_variance_ratio_, p.components_)
        return out


def parity(inps, minigroup=None):
    return np.mod(np.sum(inps[:, minigroup]), 2)


def make_ai_func(inp_dims, ai_dep, ai_func):
    rng = np.random.default_rng()
    subset = rng.choice(inp_dims, size=ai_dep, replace=False)
    func = ft.partial(ai_func, minigroup=subset)
    return func


class ImageDGWrapper(dg.DataGenerator):
    def __init__(self, use_dg, use_lvs, categorical_lv_name, categorical_lv_ind=0):
        self.use_dg = use_dg
        self.use_lvs = use_lvs
        self.cats = np.unique(use_dg.data_table[categorical_lv_name])
        self.n_cats = len(self.cats)
        self.input_dim = np.sum(use_lvs) + self.n_cats
        self.output_dim = use_dg.output_dim
        self.categorical_lv_ind = categorical_lv_ind

    def sample_reps(self, n_samps=1000, **kwargs):
        samps, reps = self.use_dg.sample_reps(n_samps, **kwargs)
        samp_cats = samps[:, self.categorical_lv_ind]
        samps = samps[:, self.use_lvs]
        context = np.stack(list(samp_cats == cat for cat in self.cats), axis=1)
        samps_use = np.concatenate((samps, context), axis=1)
        return samps_use, reps

    def get_representation(self, x):
        context = x[:, -self.n_cats :]
        samp = x[:, : self.n_cats]
        o_samps = self.use_dg.source_distribution.rvs(x.shape[0])
        o_samps[:, self.use_lvs] = samp
        con_arr = np.array(list(self.cats[c.astype(bool)] for c in context))
        o_samps[:, self.categorical_lv_ind] = np.squeeze(con_arr)
        return self.use_dg.get_representation(o_samps)


class TrackWeights(tfk.callbacks.Callback):
    def __init__(self, model, layer_ind, *args, **kwargs):
        self.model = model
        self.layer_ind = layer_ind
        super().__init__(*args, **kwargs)
        self.weights = []

    def on_train_begin(self, logs=None):
        self.weights = []

    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.weights
        self.weights.append(np.array(weights[self.layer_ind]))


class DimCorrCallback(tfk.callbacks.Callback):
    def __init__(self, model, *args, dim_samps=10**4, **kwargs):
        self.modu_model = model
        super().__init__(*args, **kwargs)
        self.dim = []
        self.corr = []
        self.dim_samps = dim_samps

    def on_train_begin(self, logs=None):
        self.dim = []
        self.dim_c0 = []
        self.corr = []

        _, _, reps = self.modu_model.sample_reps(self.dim_samps)
        dim = u.participation_ratio(reps)

        _, _, reps_c0 = self.modu_model.sample_reps(self.dim_samps, context=0)
        dim_c0 = u.participation_ratio(reps)
        corr = 1 - self.modu_model.get_ablated_loss()

        self.dim.append(dim)
        self.dim_c0.append(dim_c0)
        self.corr.append(corr)

    def on_epoch_end(self, epoch, logs=None):
        _, _, reps = self.modu_model.sample_reps(self.dim_samps)
        dim = u.participation_ratio(reps)

        _, _, reps_c0 = self.modu_model.sample_reps(self.dim_samps, context=0)
        dim_c0 = u.participation_ratio(reps_c0)

        corr = 1 - self.modu_model.get_ablated_loss()
        self.dim.append(dim)
        self.dim_c0.append(dim_c0)
        self.corr.append(corr)


class Modularizer:
    def __init__(
        self,
        inp_dims,
        groups=None,
        group_width=2,
        group_size=2,
        group_maker=sequential_groups,
        n_groups=None,
        tasks_per_group=1,
        use_mixer=False,
        use_dg=None,
        mixer_out_dims=200,
        mixer_kwargs=None,
        use_early_stopping=True,
        early_stopping_field="val_loss",
        single_output=False,
        integrate_context=False,
        n_overlap=0,
        augmented_inputs=0,
        common_augmented_inputs=False,
        augmented_input_dep=2,
        remove_last_inp=False,
        augmented_input_func=parity,
        renorm_stim=False,
        **kwargs
    ):
        self.continuous = False
        self.use_early_stopping = use_early_stopping
        self.early_stopping_field = early_stopping_field
        self.renorm_stim = renorm_stim
        self.remove_last_inp = remove_last_inp
        if not common_augmented_inputs:
            inp_dims_anc = inp_dims + augmented_inputs
        else:
            inp_dims_anc = inp_dims
        if n_groups is None:
            n_groups = int(np.floor(inp_dims / group_size))
        if integrate_context:
            sub_context = n_groups
        else:
            sub_context = 0
        if groups is None:
            groups = group_maker(
                inp_dims_anc - sub_context, group_size, n_groups, n_overlap=n_overlap
            )
        if common_augmented_inputs:
            ai_group = np.arange(inp_dims, inp_dims + augmented_inputs)
            groups = np.concatenate(
                (groups, np.tile(ai_group, (len(groups), 1))), axis=1
            )
        if augmented_inputs > 0:
            self.ai_funcs = list(
                make_ai_func(inp_dims, augmented_input_dep, augmented_input_func)
                for i in range(augmented_inputs)
            )
        else:
            self.ai_funcs = None
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
        out_dims = n_groups * tasks_per_group
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
        self.hidden_dims = int(round(len(groups) * group_width))
        self.out_group_labels = np.concatenate(
            list((i,) * tasks_per_group for i in range(n_groups))
        )

        out = self.make_model(inp_net, self.hidden_dims, self.out_dims, **kwargs)
        model, rep_model, out_model = out
        self.out_model = out_model
        self.rep_model = rep_model
        self.model = model
        self.groups = groups
        self.rng = np.random.default_rng()
        self.compiled = False
        self.loss = None
        self.layer_models = None
        self.no_output = False

    def make_model(
        self,
        inp,
        hidden,
        out,
        act_func=tf.nn.relu,
        layer_type=tfkl.Dense,
        out_act=tf.nn.sigmoid,
        additional_hidden=(),
        additional_same_reg=True,
        noise=0.1,
        inp_noise=0.01,
        kernel_reg_type=tfk.regularizers.L2,
        kernel_reg_weight=0,
        act_reg_type=tfk.regularizers.l2,
        act_reg_weight=0,
        constant_init=None,
        kernel_init=None,
        out_kernel_init=None,
        out_constant_init=None,
        use_bias=True,
        **layer_params
    ):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=inp))
        if kernel_init is not None:
            kernel_init = tfk.initializers.RandomNormal(stddev=kernel_init)
        elif constant_init is not None:
            kernel_init = tfk.initializers.Constant(constant_init)
        else:
            kernel_init = tfk.initializers.GlorotUniform()
        if out_kernel_init is not None:
            out_kernel_init = tfk.initializers.RandomNormal(stddev=out_kernel_init)
        elif out_constant_init is not None:
            out_kernel_init = tfk.initializers.Constant(constant_init)
        else:
            out_kernel_init = tfk.initializers.GlorotUniform()

        if inp_noise > 0:
            layer_list.append(tfkl.GaussianNoise(inp_noise))
        if kernel_reg_weight > 0:
            kernel_reg = kernel_reg_type(kernel_reg_weight)
        else:
            kernel_reg = None
        if act_reg_weight > 0:
            act_reg = act_reg_type(act_reg_weight)
        else:
            act_reg = None
        if additional_same_reg:
            use_ah = dict(
                kernel_regularizer=kernel_reg,
                activity_regularizer=act_reg,
                kernel_initializer=kernel_init,
                use_bias=use_bias,
            )
        else:
            use_ah = dict()
        use_ah.update(layer_params)

        for ah in additional_hidden:
            lh_ah = layer_type(ah, activation=act_func, **use_ah)
            layer_list.append(lh_ah)

        lh = layer_type(
            hidden,
            activation=act_func,
            kernel_regularizer=kernel_reg,
            activity_regularizer=act_reg,
            kernel_initializer=kernel_init,
            use_bias=use_bias,
            **layer_params
        )
        layer_list.append(lh)
        if noise > 0:
            layer_list.append(tfkl.GaussianNoise(noise))

        rep = tfk.Sequential(layer_list)
        layer_list.append(
            tfkl.Dense(
                out,
                activation=out_act,
                kernel_regularizer=kernel_reg,
                kernel_initializer=out_kernel_init,
                use_bias=use_bias,
            )
        )
        enc = tfk.Sequential(layer_list)
        rep_out = tfk.Sequential(layer_list[-1:])
        return enc, rep, rep_out

    def get_representation(self, stim, group=None, layer=None):
        if layer is None:
            rep = self.rep_model(stim)
        else:
            rep = self.get_layer_representation(stim, layer=layer, group=group)
        return rep

    def get_layer_representation(self, stim, layer=-1, group=None):
        use_layers = self.model.layers[:layer] + [self.model.layers[layer]]
        x = stim
        for i, layer_m in enumerate(use_layers):
            x = layer_m(x)
        return x

    def sample_reps(self, n_samps=1000, context=None, layer=None):
        if context is not None:
            group_inds = np.ones(n_samps, dtype=int) * context
        else:
            group_inds = None
        out = self.get_x_true(n_train=n_samps, group_inds=group_inds)
        x, true, targ = out
        rep = self.get_representation(x, layer=layer)
        return true, x, rep

    def _compile(self, optimizer=None, loss=None, ignore_nan=True,
                 lr=1e-3):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=lr)
        if loss is None:
            if ignore_nan:
                loss = mse_nanloss
            else:
                loss = tf.losses.MeanSquaredError()
        self.model.compile(optimizer, loss)
        self.loss = loss
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
            raise IOError("need to provide group_inds if single_output is True")
        elif self.single_output:
            trl_inds = np.arange(len(xs))

            ys = ys[group_inds, trl_inds]
        else:
            ys = np.concatenate(ys, axis=1)
        return ys

    def sample_stim(self, n_samps):
        if self.mix is not None:
            stim, _ = self.mix.sample_reps(n_samps)
            if self.integrate_context and not self.continuous:
                stim = stim[:, : -self.n_groups]
        else:
            stim = self.rng.uniform(0, 1, size=(n_samps, self.inp_dims)) < 0.5
        if self.renorm_stim:
            stim = 2 * (stim - 0.5)
        return stim

    def get_loss(self, **kwargs):
        return self.get_ablated_loss(ablation_mask=None, **kwargs)

    def get_transformed_out(self, x):
        reps = self.get_representation(x)
        ws, bs = self.out_model.weights
        out = np.dot(reps, ws) + bs
        return out

    def get_ablated_loss(
        self,
        ablation_mask=None,
        group_ind=None,
        n_samps=1000,
        ret_err_rate=True,
        layer=None,
    ):
        if not self.compiled:
            self._compile()
        x, true, targ = self.get_x_true(n_train=n_samps, group_inds=group_ind)
        if layer is None:
            reps = self.get_representation(x)
            if ablation_mask is not None:
                reps = reps * np.logical_not(ablation_mask)
            out = self.out_model(reps)
        else:
            for i, layer_m in enumerate(self.model.layers):
                x = layer_m(x)
                if i == layer and ablation_mask is not None:
                    x = x * np.logical_not(ablation_mask)
            out = x
        if ret_err_rate:
            out_binary = out > 0.5
            out = 1 - np.mean(out_binary == targ, axis=(0, 1))
        else:
            out = self.loss(targ, out)
        return out

    def get_x_true(
        self,
        x=None,
        true=None,
        n_train=10**5,
        group_inds=None,
        special_fdg=None,
        only_groups=None,
        only_tasks=None,
    ):
        if only_groups is not None:
            group_inds = self.rng.choice(only_groups, n_train)
        if true is None and x is not None:
            raise IOError("need ground truth x")
        if true is None:
            true = self.sample_stim(n_train)
        if group_inds is None and self.single_output:
            if self.continuous:
                group_inds = np.argmax(true[:, -self.n_groups:], axis=1)
            else:
                group_inds = self.rng.choice(self.n_groups, n_train)
        elif group_inds is not None and self.continuous:
            m_inds = np.argmax(true[:, -self.n_groups:], axis=1)
            if not u.check_list(group_inds):
                group_inds = np.ones(true.shape[0], dtype=int)*group_inds
            for i in range(true.shape[0]):
                m_i = m_inds[i] - self.n_groups
                g_i = group_inds[i] - self.n_groups
                s = true[i, m_i]
                true[i, m_i] = true[i, g_i]
                true[i, g_i] = s

        if self.single_output:
            to_add = np.zeros((n_train, self.n_groups))
            trl_inds = np.arange(n_train)
            to_add[trl_inds, group_inds] = 1
            if self.remove_last_inp:
                to_add = to_add[:, :-1]
        if self.single_output and self.integrate_context and not self.continuous:
            true = np.concatenate((true, to_add), axis=1)
        if x is None and special_fdg is None:
            x = self.mix_func(true)
        elif x is None and special_fdg is not None:
            x = special_fdg.get_representation(true)
        if self.single_output and not self.integrate_context and not self.continuous:
            x = np.concatenate((x, to_add), axis=1)
        if self.no_output:
            targ = None
        else:
            targ = self.generate_target(true, group_inds=group_inds)
        if only_tasks is not None:
            nan_inds = np.array(list(
                set(np.arange(targ.shape[1])).difference(only_tasks)
            ))
            targ[:, nan_inds] = np.nan
        return x, true, targ

    def fit(
        self,
        train_x=None,
        train_true=None,
        eval_x=None,
        eval_true=None,
        n_train=2 * 10**5,
        epochs=15,
        batch_size=100,
        n_val=10**3,
        track_dimensionality=True,
        special_fdg=None,
        return_training=False,
        only_groups=None,
        only_tasks=None,
        val_only_groups=None,
        val_only_tasks=None,
        **kwargs
    ):
        if val_only_groups is None:
            val_only_groups = only_groups
        if val_only_tasks is None:
            val_only_tasks = only_tasks
        if not self.compiled:
            self._compile()

        train_x, train_true, train_targ = self.get_x_true(
            train_x,
            train_true,
            n_train,
            special_fdg=special_fdg,
            only_groups=only_groups,
            only_tasks=only_tasks,
        )
        eval_x, eval_true, eval_targ = self.get_x_true(
            eval_x,
            eval_true,
            n_val,
            special_fdg=special_fdg,
            only_groups=val_only_groups,
            only_tasks=val_only_tasks,
        )

        eval_set = (eval_x, eval_targ)

        if self.use_early_stopping:
            cb = tfk.callbacks.EarlyStopping(
                monitor=self.early_stopping_field, mode="min", patience=2
            )
            curr_cb = kwargs.get("callbacks", [])
            curr_cb.append(cb)
            kwargs["callbacks"] = curr_cb
        if track_dimensionality:
            cb = kwargs.get("callbacks", [])
            d_callback = DimCorrCallback(self)
            cb.append(d_callback)
            kwargs["callbacks"] = cb

        out = self.model.fit(
            x=train_x,
            y=train_targ,
            epochs=epochs,
            validation_data=eval_set,
            batch_size=batch_size,
            **kwargs
        )
        if track_dimensionality:
            out.history["dimensionality"] = d_callback.dim
            out.history["dimensionality_c0"] = d_callback.dim_c0
            out.history["corr_rate"] = d_callback.corr
        if return_training:
            out = (out, (train_x, train_true, train_targ))
        return out


class XORModularizer(Modularizer):
    def __init__(self, *args, group_func=xor, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_func = (group_func,) * len(self.groups)


class ColoringModularizer(Modularizer):
    def _make_group_func(self, n_colorings, n_g, tasks_per_group=1, merger=np.sum):
        funcs = []
        for i in range(tasks_per_group):
            cols = generate_many_colorings(n_colorings, n_g)
            funcs.append(
                ft.partial(apply_many_colorings, colorings=cols, merger=merger)
            )
        return lambda x: np.concatenate(list(f(x) for f in funcs), axis=1)

    def __init__(
        self, *args, n_colorings=None, tasks_per_group=1, task_merger=np.sum, **kwargs
    ):
        super().__init__(*args, tasks_per_group=tasks_per_group, **kwargs)
        if n_colorings is None:
            n_colorings = 2 ** (self.group_size - 1)
        group_func = []
        for g in self.groups:
            group_func.append(
                self._make_group_func(
                    n_colorings,
                    len(g),
                    tasks_per_group=tasks_per_group,
                    merger=task_merger,
                )
            )
        self.group_func = tuple(group_func)


class IdentityModularizer(Modularizer):
    def __init__(
        self,
        inp_dims,
        group_maker=sequential_groups,
        hidden_dims=None,
        group_size=2,
        provide_groups=None,
        n_groups=None,
        integrate_context=True,
        use_mixer=True,
        use_dg=None,
        single_output=True,
        tasks_per_group=None,
        remove_last_inp=False,
        **kwargs
    ):
        if hidden_dims is None:
            if use_dg is not None:
                hidden_dims = use_dg.output_dim
            else:
                hidden_dims = 100
        self.hidden_dims = hidden_dims
        self.continuous = False
        self.remove_last_inp = remove_last_inp
        if n_groups is None:
            n_groups = int(np.floor(inp_dims / group_size))
        if group_size is None:
            group_size = int(np.floor(inp_dims / n_groups))
        if provide_groups is None:
            self.groups = group_maker(inp_dims, group_size, n_groups)
        else:
            self.groups = provide_groups
        if use_dg is not None:
            self.mix = use_dg
            self.mix_func = use_dg.get_representation
            inp_net = self.mix.output_dim
        self.single_output = True
        self.integrate_context = integrate_context
        self.n_groups = len(self.groups)
        self.compiled = True
        self.no_output = True
        self.renorm_stim = False
        self.rng = np.random.default_rng()
        self.n_tasks_per_group = tasks_per_group

    @classmethod
    def copy_groups(cls, model):
        hd = model.inp_net
        mg = model.groups
        inp_dims = model.inp_dims
        ic = model.integrate_context
        return cls(inp_dims, provide_groups=mg, hidden_dims=hd, integrate_context=ic)

    def get_representation(self, stim, group=None):
        return stim

    def model(self, stim):
        return stim


def make_linear_task_func(n_g, n_tasks=1, i_var=0, center=0.5, renorm=False, **kwargs):
    task, intercept = generate_linear_tasks(
        n_g, n_tasks=n_tasks, intercept_var=i_var, **kwargs
    )
    return ft.partial(
        apply_linear_task, task=task, intercept=intercept, center=center, renorm=renorm
    )


class LinearIdentityModularizer(IdentityModularizer):
    def _make_linear_task_func(self, n_g, n_tasks=1, i_var=0, center=0, renorm=False):
        return make_linear_task_func(
            n_g, n_tasks=n_tasks, i_var=i_var, center=center, renorm=renorm
        )

    def __init__(
        self,
        *args,
        tasks_per_group=1,
        intercept_var=0,
        center=0.5,
        renorm_tasks=False,
        **kwargs
    ):
        super().__init__(*args, tasks_per_group=tasks_per_group, **kwargs)
        group_func = []
        for g in self.groups:
            group_func.append(
                self._make_linear_task_func(
                    len(g),
                    n_tasks=tasks_per_group,
                    i_var=intercept_var,
                    center=center,
                    renorm=renorm_tasks,
                )
            )
        self.group_func = tuple(group_func)
        self.no_output = False


class LinearModularizer(Modularizer):
    def _make_linear_task_func(self, n_g, n_tasks=1, i_var=0, center=0, renorm=False):
        return make_linear_task_func(
            n_g, n_tasks=n_tasks, i_var=i_var, center=center, renorm=renorm
        )

    def __init__(
        self,
        *args,
        tasks_per_group=1,
        intercept_var=0,
        center=0.5,
        renorm_tasks=False,
        **kwargs
    ):
        super().__init__(*args, tasks_per_group=tasks_per_group, **kwargs)
        group_func = []
        for g in self.groups:
            group_func.append(
                self._make_linear_task_func(
                    len(g),
                    n_tasks=tasks_per_group,
                    i_var=intercept_var,
                    center=center,
                    renorm=renorm_tasks,
                )
            )
        self.group_func = tuple(group_func)


class LinearContinuousModularizer(LinearModularizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.continuous = True

    def _make_linear_task_func(self, n_g, n_tasks=1, offset_var=0.4, **kwargs):
        out = da.generate_partition_functions(
            n_g, n_funcs=n_tasks, offset_var=offset_var, **kwargs
        )
        return ft.partial(apply_continuous_task, task=out[0])


group_maker_dict = {
    "overlap": overlap_groups,
}
act_func_dict = {
    "relu": tf.nn.relu,
}
model_type_dict = {
    "coloring": ColoringModularizer,
    "linear": LinearModularizer,
    "identity": IdentityModularizer,
}


def make_linear_network(
    xs,
    ys,
    hiddens=(200,),
    optimizer=None,
    loss=tf.losses.MeanSquaredError(),
    n_epochs=200,
    init_std=0.0001,
    act_func=None,
    use_relu=False,
    use_bias=False,
    track_weights=True,
    **kwargs
):
    if use_relu:
        act_func = tf.nn.relu
    inp = tfk.Input(shape=xs.shape[1])
    init = tfk.initializers.RandomNormal(stddev=init_std)
    x = inp
    for h in hiddens:
        x = tfkl.Dense(
            h, kernel_initializer=init, activation=act_func, use_bias=use_bias
        )(x)
    m_rep = tfk.Model(inp, x)
    out = tfkl.Dense(
        ys.shape[1], kernel_initializer=init, use_bias=use_bias  # activation=act_func,
    )(x)
    m = tfk.Model(inp, out)
    if optimizer is None:
        optimizer = tf.optimizers.Adam(learning_rate=1e-3)
    m.compile(optimizer, loss)
    if track_weights:
        cb = kwargs.get("callbacks", [])
        weight_callback = TrackWeights(m, 0)
        cb.append(weight_callback)
        bias_callback = TrackWeights(m, 1)
        cb.append(bias_callback)
        kwargs["callbacks"] = cb

    original_weights = u.make_unit_vector(np.copy(m.weights[0]).T)
    h = m.fit(xs, ys, epochs=n_epochs, **kwargs)
    h.history["weights"] = weight_callback.weights
    h.history["bias"] = bias_callback.weights
    return m, m_rep, h, original_weights


class GatedLinearModularizerShell:
    def __init__(
        self, model, weight_generator=None, n_units_per_superset=200, n_samps=1000
    ):
        if weight_generator is None:
            self.weight_init = tf.keras.initializers.GlorotUniform()
        else:
            self.weight_init = weight_generator
        self.model = model
        # self.chamber_funcs = ma.decompose_model_tasks(model)

        _, stim_sample, _ = self.model.get_x_true(n_train=n_samps)
        out = self.make_module_set(
            self.chamber_funcs,
            stim_sample,
            n_units=n_units_per_superset,
        )
        self.gl_model, self.func_list, self.rep_models = out

    def apply_gating(self, stim, func):
        rel_stim = maux.get_relevant_dims(stim, self.model)
        return func(rel_stim)

    def make_module_set(self, func_dict, stim, n_units):
        wi = self.weight_init
        input_units = self.model.inp_net
        output_units = self.model.out_dims
        stim_input = tfk.Input(shape=input_units, name="stim_input")
        mod_dict = {}
        outs = []
        inputs = []
        ordered_funcs = []

        rep_models = {}
        for k, funcs in func_dict.items():
            interior_inputs = []
            interior_outputs = []
            interior_funcs = []
            for i, func in enumerate(funcs):
                gate_k_i_input = tfk.Input(shape=1, name=str(k + (i,)))

                mask = self.apply_gating(stim, func)

                n_units_mod = (np.mean(mask) * n_units).astype(int)

                x = tf.multiply(gate_k_i_input, tfkl.Dense(n_units_mod)(stim_input))

                model_k_i = tfk.Model([stim_input, gate_k_i_input], x)

                interior_inputs.append(gate_k_i_input)
                interior_outputs.append(x)
                interior_funcs.append(func)

                outs.append(model_k_i.output)
                inputs.append(gate_k_i_input)
                ordered_funcs.append(func)
            rep_m = tfk.Model(
                inputs=[stim_input] + interior_inputs,
                outputs=tfkl.concatenate(interior_outputs),
            )
            rep_models[k] = (rep_m, interior_funcs)

        out = tfkl.Dense(output_units)(tfkl.concatenate(outs))
        model = tfk.Model(inputs=[stim_input] + inputs, outputs=out)
        return model, ordered_funcs, rep_models

    def make_input(self, inp, stim, func_list=None):
        if func_list is None:
            func_list = self.func_list
        gate_inputs = tuple(self.apply_gating(stim, f) for f in func_list)
        inputs_all = (inp,) + gate_inputs
        return inputs_all

    def get_output(self, rep, stim):
        inputs_all = self.make_input(rep, stim)
        return self.gl_model(inputs_all)

    def get_representation(self, rep, stim, key=None):
        if key is not None:
            keys = (key,)
        else:
            keys = self.rep_models.keys()
        outs = {}
        for k in keys:
            rm, funcs = self.rep_models[k]
            input_rm = self.make_input(rep, stim, func_list=funcs)
            outs[k] = rm(input_rm)
        return outs

    def _compile(self, optimizer=None, loss=tf.losses.MeanSquaredError()):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.gl_model.compile(optimizer, loss)
        self.compiled = True

    def fit(self, n_trains=10000, norm_targs=True, **kwargs):
        self._compile()
        rep, stim, targ = self.model.get_x_true(n_train=n_trains)
        if norm_targs:
            targ = skp.StandardScaler().fit_transform(targ)
        inputs_all = self.make_input(rep, stim)

        self.gl_model.fit(inputs_all, targ, **kwargs)


def train_modularizer(
    fdg,
    verbose=False,
    params=None,
    group_maker_dict=group_maker_dict,
    model_type_str=None,
    act_func_dict=act_func_dict,
    model_type_dict=model_type_dict,
    track_dimensionality=True,
    only_groups=None,
    only_tasks=None,
    **kwargs
):
    if params is not None:
        group_size = params.getint("group_size")
        n_overlap = params.getint("n_overlap")
        group_maker = group_maker_dict[params.get("group_maker")]
        tasks_per_group = params.getint("tasks_per_group")
        n_groups = params.getint("n_groups")
        sigma = params.getfloat("sigma")
        inp_noise = params.getfloat("inp_noise")
        act_reg = params.getfloat("act_reg")
        group_width = params.getint("group_width")
        use_mixer = params.getboolean("use_mixer")
        act_func = act_func_dict[params.get("act_func")]
        hiddens = params.getlist("hiddens", typefunc=int)
        if hiddens is None:
            hiddens = ()
        train_epochs = params.getint("train_epochs")
        single_output = params.getboolean("single_output")
        integrate_context = params.getboolean("integrate_context")
        model_type = model_type_dict[params.get("model_type")]
        n_train = params.getint("modu_train_egs")

        config_dict = {
            "group_size": group_size,
            "n_groups": n_groups,
            "group_maker": group_maker,
            "use_dg": fdg,
            "group_width": group_width,
            "use_mixer": use_mixer,
            "tasks_per_group": tasks_per_group,
            "act_func": act_func,
            "additional_hidden": hiddens,
            "act_reg_weight": act_reg,
            "noise": sigma,
            "inp_noise": inp_noise,
            "n_overlap": n_overlap,
            "single_output": single_output,
            "integrate_context": integrate_context,
            "train_epochs": train_epochs,
            "model_type": model_type,
            "use_dg": fdg,
            "n_train": n_train,
        }
    else:
        config_dict = {
            "use_dg": fdg,
            "model_type": model_type_dict["linear"],
            "group_maker": group_maker_dict["overlap"],
        }
    config_dict.update(kwargs)
    inp_dim = config_dict["use_dg"].input_dim
    train_epochs = config_dict.pop("train_epochs")
    model_type = config_dict.pop("model_type")
    n_train = config_dict.pop("n_train")
    if model_type_str is not None:
        model_type = model_type_dict[model_type_str]

    m = model_type(inp_dim, **config_dict)
    if train_epochs > 0:
        h = m.fit(
            epochs=train_epochs,
            verbose=verbose,
            track_dimensionality=track_dimensionality,
            n_train=n_train,
            only_groups=only_groups,
            only_tasks=only_tasks,
        )
    else:
        h = None
    return m, h
