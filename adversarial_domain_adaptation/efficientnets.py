#Code adapted from https://github.com/leondgarse/keras_efficientnet_v2/
#Licensed under Apache 2.0 license
#Replaced BatchNormalization layers with SyncBatchNormalization


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
TF_BATCH_NORM_EPSILON = 0.001
LAYER_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


""" Wrapper for default parameters """


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def hard_swish(inputs):
    """ `out = xx * relu6(xx + 3) / 6`, arxiv: https://arxiv.org/abs/1905.02244 """
    return inputs * tf.nn.relu6(inputs + 3) / 6


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def hard_sigmoid_torch(inputs):
    """https://pytorch.org/docs/stable/generated/torch.nn.Hardsigmoid.html
    toch.nn.Hardsigmoid: 0 if x <= −3 else (1 if x >= 3 else x / 6 + 1/2)
    keras.activations.hard_sigmoid: 0 if x <= −2.5 else (1 if x >= 2.5 else x / 5 + 1/2) -> tf.clip_by_value(inputs / 5 + 0.5, 0, 1)
    """
    return tf.clip_by_value(inputs / 6 + 0.5, 0, 1)


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def mish(inputs):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function.
    Paper: [Mish: A Self Regularized Non-Monotonic Neural Activation Function](https://arxiv.org/abs/1908.08681)
    Copied from https://github.com/tensorflow/addons/blob/master/tensorflow_addons/activations/mish.py
    """
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def phish(inputs):
    """Phish is defined as f(x) = xTanH(GELU(x)) with no discontinuities in the f(x) derivative.
    Paper: https://www.techrxiv.org/articles/preprint/Phish_A_Novel_Hyper-Optimizable_Activation_Function/17283824
    """
    return inputs * tf.math.tanh(tf.nn.gelu(inputs))


def activation_by_name(inputs, activation="relu", name=None):
    """ Typical Activation layer added hard_swish and prelu. """
    if activation is None:
        return inputs

    layer_name = name and activation and name + activation
    if activation == "hard_swish":
        return keras.layers.Activation(activation=hard_swish, name=layer_name)(inputs)
    elif activation == "mish":
        return keras.layers.Activation(activation=mish, name=layer_name)(inputs)
    elif activation == "phish":
        return keras.layers.Activation(activation=phish, name=layer_name)(inputs)
    elif activation.lower() == "prelu":
        shared_axes = list(range(1, len(inputs.shape)))
        shared_axes.pop(-1 if K.image_data_format() == "channels_last" else 0)
        # print(f"{shared_axes = }")
        return keras.layers.PReLU(shared_axes=shared_axes, alpha_initializer=tf.initializers.Constant(0.25), name=layer_name)(inputs)
    elif activation.lower().startswith("gelu/app"):
        # gelu/approximate
        return tf.nn.gelu(inputs, approximate=True, name=layer_name)
    elif activation.lower() == ("hard_sigmoid_torch"):
        return keras.layers.Activation(activation=hard_sigmoid_torch, name=layer_name)(inputs)
    else:
        return keras.layers.Activation(activation=activation, name=layer_name)(inputs)


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
class EvoNormalization(tf.keras.layers.Layer):
    def __init__(self, nonlinearity=True, num_groups=-1, zero_gamma=False, momentum=0.99, epsilon=0.001, data_format="auto", **kwargs):
        # [evonorm](https://github.com/tensorflow/tpu/blob/master/models/official/resnet/resnet_model.py)
        # EVONORM_B0: nonlinearity=True, num_groups=-1
        # EVONORM_S0: nonlinearity=True, num_groups > 0
        # EVONORM_B0 / EVONORM_S0 linearity: nonlinearity=False, num_groups=-1
        # EVONORM_S0A linearity: nonlinearity=False, num_groups > 0
        super().__init__(**kwargs)
        self.data_format, self.nonlinearity, self.zero_gamma, self.num_groups = data_format, nonlinearity, zero_gamma, num_groups
        self.momentum, self.epsilon = momentum, epsilon
        self.is_channels_first = True if data_format == "channels_first" or (data_format == "auto" and K.image_data_format() == "channels_first") else False

    def build(self, input_shape):
        all_axes = list(range(len(input_shape)))
        param_shape = [1] * len(input_shape)
        if self.is_channels_first:
            param_shape[1] = input_shape[1]
            self.reduction_axes = all_axes[:1] + all_axes[2:]
        else:
            param_shape[-1] = input_shape[-1]
            self.reduction_axes = all_axes[:-1]

        self.gamma = self.add_weight(name="gamma", shape=param_shape, initializer="zeros" if self.zero_gamma else "ones", trainable=True)
        self.beta = self.add_weight(name="beta", shape=param_shape, initializer="zeros", trainable=True)
        if self.num_groups <= 0:  # EVONORM_B0
            self.moving_variance = self.add_weight(
                name="moving_variance",
                shape=param_shape,
                initializer="ones",
                synchronization=tf.VariableSynchronization.ON_READ,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN,
            )
        if self.nonlinearity:
            self.vv = self.add_weight(name="vv", shape=param_shape, initializer="ones", trainable=True)

        if self.num_groups > 0:  # EVONORM_S0
            channels_dim = input_shape[1] if self.is_channels_first else input_shape[-1]
            num_groups = int(self.num_groups)
            while num_groups > 1:
                if channels_dim % num_groups == 0:
                    break
                num_groups -= 1
            self.__num_groups__ = num_groups
            self.groups_dim = channels_dim // self.__num_groups__

            if self.is_channels_first:
                self.group_shape = [-1, self.__num_groups__, self.groups_dim, *input_shape[2:]]
                self.group_reduction_axes = list(range(2, len(self.group_shape)))  # [2, 3, 4]
                self.group_axes = 2
                self.var_shape = [-1, *param_shape[1:]]
            else:
                self.group_shape = [-1, *input_shape[1:-1], self.__num_groups__, self.groups_dim]
                self.group_reduction_axes = list(range(1, len(self.group_shape) - 2)) + [len(self.group_shape) - 1]  # [1, 2, 4]
                self.group_axes = -1
                self.var_shape = [-1, *param_shape[1:]]

    def __group_std__(self, inputs):
        # _group_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L171
        grouped = tf.reshape(inputs, self.group_shape)
        _, var = tf.nn.moments(grouped, self.group_reduction_axes, keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        std = tf.repeat(std, self.groups_dim, axis=self.group_axes)
        return tf.reshape(std, self.var_shape)

    def __batch_std__(self, inputs, training=None):
        # _batch_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L120
        def _call_train_():
            _, var = tf.nn.moments(inputs, self.reduction_axes, keepdims=True)
            # update_op = tf.assign_sub(moving_variance, (moving_variance - variance) * (1 - decay))
            delta = (self.moving_variance - var) * (1 - self.momentum)
            self.moving_variance.assign_sub(delta)
            return var

        def _call_test_():
            return self.moving_variance

        var = K.in_train_phase(_call_train_, _call_test_, training=training)
        return tf.sqrt(var + self.epsilon)

    def __instance_std__(self, inputs):
        # _instance_std, https://github.com/tensorflow/tpu/blob/main/models/official/resnet/resnet_model.py#L111
        # axes = [1, 2] if data_format == 'channels_last' else [2, 3]
        _, var = tf.nn.moments(inputs, self.reduction_axes[1:], keepdims=True)
        return tf.sqrt(var + self.epsilon)

    def call(self, inputs, training=None, **kwargs):
        if self.nonlinearity and self.num_groups > 0:  # EVONORM_S0
            den = self.__group_std__(inputs)
            inputs = inputs * tf.nn.sigmoid(self.vv * inputs) / den
        elif self.num_groups > 0:  # EVONORM_S0a
            # EvoNorm2dS0a https://github.com/rwightman/pytorch-image-models/blob/main/timm/models/layers/evo_norm.py#L239
            den = self.__group_std__(inputs)
            inputs = inputs / den
        elif self.nonlinearity:  # EVONORM_B0
            left = self.__batch_std__(inputs, training)
            right = self.vv * inputs + self.__instance_std__(inputs)
            inputs = inputs / tf.maximum(left, right)
        return inputs * self.gamma + self.beta

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "nonlinearity": self.nonlinearity,
                "zero_gamma": self.zero_gamma,
                "num_groups": self.num_groups,
                "momentum": self.momentum,
                "epsilon": self.epsilon,
                "data_format": self.data_format,
            }
        )
        return config


def batchnorm_with_activation(
    inputs, activation=None, zero_gamma=False, epsilon=1e-5, momentum=0.9, act_first=False, use_evo_norm=False, evo_norm_group_size=-1, name=None
):
    """ Performs a batch normalization followed by an activation. """
    if use_evo_norm:
        nonlinearity = False if activation is None else True
        num_groups = inputs.shape[-1] // evo_norm_group_size  # Currently using gorup_size as parameter only
        return EvoNormalization(nonlinearity, num_groups=num_groups, zero_gamma=zero_gamma, epsilon=epsilon, momentum=momentum, name=name + "evo_norm")(inputs)

    bn_axis = -1 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    if act_first and activation:
        inputs = activation_by_name(inputs, activation=activation, name=name)
    nn = keras.layers.experimental.SyncBatchNormalization(
        axis=bn_axis,
        momentum=momentum,
        epsilon=epsilon,
        gamma_initializer=gamma_initializer,
        name=name and name + "bn",
    )(inputs)
    if not act_first and activation:
        nn = activation_by_name(nn, activation=activation, name=name)
    return nn


def layer_norm(inputs, zero_gamma=False, epsilon=LAYER_NORM_EPSILON, name=None):
    """ Typical LayerNormalization with epsilon=1e-5 """
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    return keras.layers.LayerNormalization(axis=norm_axis, epsilon=epsilon, gamma_initializer=gamma_initializer, name=name and name + "ln")(inputs)


def group_norm(inputs, groups=32, epsilon=BATCH_NORM_EPSILON, name=None):
    """ Typical GroupNormalization with epsilon=1e-5 """
    from tensorflow_addons.layers import GroupNormalization

    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    return GroupNormalization(groups=groups, axis=norm_axis, epsilon=epsilon, name=name and name + "group_norm")(inputs)


def conv2d_no_bias(inputs, filters, kernel_size=1, strides=1, padding="VALID", use_bias=False, groups=1, use_torch_padding=True, name=None, **kwargs):
    """ Typical Conv2D with `use_bias` default as `False` and fixed padding """
    pad = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, (list, tuple)) else (kernel_size // 2, kernel_size // 2)
    if use_torch_padding and padding.upper() == "SAME" and max(pad) != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
        padding = "VALID"

    groups = max(1, groups)
    return keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        groups=groups,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "conv",
        **kwargs,
    )(inputs)


def depthwise_conv2d_no_bias(inputs, kernel_size, strides=1, padding="VALID", use_bias=False, use_torch_padding=True, name=None, **kwargs):
    """ Typical DepthwiseConv2D with `use_bias` default as `False` and fixed padding """
    pad = (kernel_size[0] // 2, kernel_size[1] // 2) if isinstance(kernel_size, (list, tuple)) else (kernel_size // 2, kernel_size // 2)
    if use_torch_padding and padding.upper() == "SAME" and max(pad) != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "dw_pad")(inputs)
        padding = "VALID"
    return keras.layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        padding=padding,
        use_bias=use_bias,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "dw_conv",
        **kwargs,
    )(inputs)


""" Blocks """


def output_block(inputs, filters=0, activation="relu", num_classes=1000, drop_rate=0, classifier_activation="softmax", is_torch_mode=True, act_first=False):
    nn = inputs
    if filters > 0:  # efficientnet like
        bn_eps = BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON
        nn = conv2d_no_bias(nn, filters, 1, strides=1, use_bias=act_first, use_torch_padding=is_torch_mode, name="features_")  # Also use_bias for act_first
        nn = batchnorm_with_activation(nn, activation=activation, act_first=act_first, epsilon=bn_eps, name="features_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)
    return nn


def global_context_module(inputs, use_attn=True, ratio=0.25, divisor=1, activation="relu", use_bias=True, name=None):
    """ Global Context Attention Block, arxiv: https://arxiv.org/pdf/1904.11492.pdf """
    height, width, filters = inputs.shape[1], inputs.shape[2], inputs.shape[-1]

    # activation could be ("relu", "hard_sigmoid")
    hidden_activation, output_activation = activation if isinstance(activation, (list, tuple)) else (activation, "sigmoid")
    reduction = make_divisible(filters * ratio, divisor, limit_round_down=0.0)

    if use_attn:
        attn = keras.layers.Conv2D(1, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "attn_conv")(inputs)
        attn = tf.reshape(attn, [-1, 1, 1, height * width])  # [batch, height, width, 1] -> [batch, 1, 1, height * width]
        attn = tf.nn.softmax(attn, axis=-1)
        context = tf.reshape(inputs, [-1, 1, height * width, filters])
        context = attn @ context  # [batch, 1, 1, filters]
    else:
        context = tf.reduce_mean(inputs, [1, 2], keepdims=True)

    mlp = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, name=name and name + "mlp_1_conv")(context)
    mlp = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=name and name + "ln")(mlp)
    mlp = activation_by_name(mlp, activation=hidden_activation, name=name)
    mlp = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, name=name and name + "mlp_2_conv")(mlp)
    mlp = activation_by_name(mlp, activation=output_activation, name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, mlp])


def se_module(inputs, se_ratio=0.25, divisor=8, limit_round_down=0.9, activation="relu", use_bias=True, name=None):
    """ Squeeze-and-Excitation block, arxiv: https://arxiv.org/pdf/1709.01507.pdf """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    # activation could be ("relu", "hard_sigmoid") for mobilenetv3
    hidden_activation, output_activation = activation if isinstance(activation, (list, tuple)) else (activation, "sigmoid")
    filters = inputs.shape[channel_axis]
    reduction = make_divisible(filters * se_ratio, divisor, limit_round_down=limit_round_down)
    # print(f"{filters = }, {se_ratio = }, {divisor = }, {reduction = }")
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "1_conv")(se)
    se = activation_by_name(se, activation=hidden_activation, name=name)
    se = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "2_conv")(se)
    se = activation_by_name(se, activation=output_activation, name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, se])


def eca_module(inputs, gamma=2.0, beta=1.0, name=None, **kwargs):
    """ Efficient Channel Attention block, arxiv: https://arxiv.org/pdf/1910.03151.pdf """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    beta, gamma = float(beta), float(gamma)
    tt = int((tf.math.log(float(filters)) / tf.math.log(2.0) + beta) / gamma)
    kernel_size = max(tt if tt % 2 else tt + 1, 3)
    pad = kernel_size // 2

    nn = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=False)
    nn = tf.pad(nn, [[0, 0], [pad, pad]])
    nn = tf.expand_dims(nn, channel_axis)

    nn = keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="VALID", use_bias=False, name=name and name + "conv1d")(nn)
    nn = tf.squeeze(nn, axis=channel_axis)
    nn = activation_by_name(nn, activation="sigmoid", name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, nn])


def drop_connect_rates_split(num_blocks, start=0.0, end=0.0):
    """ split drop connect rate in range `(start, end)` according to `num_blocks` """
    drop_connect_rates = tf.split(tf.linspace(start, end, sum(num_blocks)), num_blocks)
    return [ii.numpy().tolist() for ii in drop_connect_rates]


def drop_block(inputs, drop_rate=0, name=None):
    """ Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382 """
    if drop_rate > 0:
        noise_shape = [None] + [1] * (len(inputs.shape) - 1)  # [None, 1, 1, 1]
        return keras.layers.Dropout(drop_rate, noise_shape=noise_shape, name=name and name + "drop")(inputs)
    else:
        return inputs


""" Other layers / functions """


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def __anti_alias_downsample_initializer__(weight_shape, dtype="float32"):
    import numpy as np

    kernel_size, channel = weight_shape[0], weight_shape[2]
    ww = tf.cast(np.poly1d((0.5, 0.5)) ** (kernel_size - 1), dtype)
    ww = tf.expand_dims(ww, 0) * tf.expand_dims(ww, 1)
    ww = tf.repeat(ww[:, :, tf.newaxis, tf.newaxis], channel, axis=-2)
    return ww


def anti_alias_downsample(inputs, kernel_size=3, strides=2, padding="SAME", trainable=False, name=None):
    """ DepthwiseConv2D performing anti-aliasing downsample, arxiv: https://arxiv.org/pdf/1904.11486.pdf """
    return keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="SAME",
        use_bias=False,
        trainable=trainable,
        depthwise_initializer=__anti_alias_downsample_initializer__,
        name=name and name + "anti_alias_down",
    )(inputs)


def make_divisible(vv, divisor=4, min_value=None, limit_round_down=0.9):
    """ Copied from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(vv + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < limit_round_down * vv:
        new_v += divisor
    return new_v


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def __unfold_filters_initializer__(weight_shape, dtype="float32"):
    kernel_size = weight_shape[0]
    kernel_out = kernel_size * kernel_size
    ww = tf.reshape(tf.eye(kernel_out), [kernel_size, kernel_size, 1, kernel_out])
    if len(weight_shape) == 5:  # Conv3D or Conv3DTranspose
        ww = tf.expand_dims(ww, 2)
    return ww


def fold_by_conv2d_transpose(patches, output_shape=None, kernel_size=3, strides=2, dilation_rate=1, padding="SAME", compressed="auto", name=None):
    paded = kernel_size // 2 if padding else 0
    if compressed == "auto":
        compressed = True if len(patches.shape) == 4 else False

    if compressed:
        _, hh, ww, cc = patches.shape
        channel = cc // kernel_size // kernel_size
        conv_rr = tf.reshape(patches, [-1, hh * ww, kernel_size * kernel_size, channel])
    else:
        _, hh, ww, _, _, channel = patches.shape
        # conv_rr = patches
        conv_rr = tf.reshape(patches, [-1, hh * ww, kernel_size * kernel_size, channel])
    conv_rr = tf.transpose(conv_rr, [0, 3, 1, 2])  # [batch, channnel, hh * ww, kernel * kernel]
    conv_rr = tf.reshape(conv_rr, [-1, hh, ww, kernel_size * kernel_size])

    convtrans_rr = keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="VALID",
        output_padding=paded,
        use_bias=False,
        trainable=False,
        kernel_initializer=__unfold_filters_initializer__,
        name=name and name + "fold_convtrans",
    )(conv_rr)

    out = tf.reshape(convtrans_rr[..., 0], [-1, channel, convtrans_rr.shape[1], convtrans_rr.shape[2]])
    out = tf.transpose(out, [0, 2, 3, 1])
    if output_shape is None:
        output_shape = [-paded, -paded]
    else:
        output_shape = [output_shape[0] + paded, output_shape[1] + paded]
    out = out[:, paded : output_shape[0], paded : output_shape[1]]
    return out


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
class CompatibleExtractPatches(keras.layers.Layer):
    def __init__(self, sizes=3, strides=2, rates=1, padding="SAME", compressed=True, force_conv=False, **kwargs):
        super().__init__(**kwargs)
        self.sizes, self.strides, self.rates, self.padding = sizes, strides, rates, padding
        self.compressed, self.force_conv = compressed, force_conv

        self.kernel_size = sizes[1] if isinstance(sizes, (list, tuple)) else sizes
        self.strides = strides[1] if isinstance(strides, (list, tuple)) else strides
        self.dilation_rate = rates[1] if isinstance(rates, (list, tuple)) else rates
        self.filters = self.kernel_size * self.kernel_size

        if len(tf.config.experimental.list_logical_devices("TPU")) != 0 or self.force_conv:
            self.use_conv = True
        else:
            self.use_conv = False

    def build(self, input_shape):
        _, self.height, self.width, self.channel = input_shape
        if self.padding.upper() == "SAME":
            pad = self.kernel_size // 2
            self.pad_value = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
            self.height, self.width = self.height + pad * 2, self.width + pad * 2

        if self.use_conv:
            self.conv = keras.layers.Conv2D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.strides,
                dilation_rate=self.dilation_rate,
                padding="VALID",
                use_bias=False,
                trainable=False,
                kernel_initializer=__unfold_filters_initializer__,
                name=self.name and self.name + "unfold_conv",
            )
            self.conv.build([None, *input_shape[1:-1], 1])
        else:
            self._sizes_ = [1, self.kernel_size, self.kernel_size, 1]
            self._strides_ = [1, self.strides, self.strides, 1]
            self._rates_ = [1, self.dilation_rate, self.dilation_rate, 1]

    def call(self, inputs):
        if self.padding.upper() == "SAME":
            inputs = tf.pad(inputs, self.pad_value)

        if self.use_conv:
            merge_channel = tf.transpose(inputs, [0, 3, 1, 2])
            merge_channel = tf.reshape(merge_channel, [-1, self.height, self.width, 1])
            conv_rr = self.conv(merge_channel)

            # TFLite not supporting `tf.transpose` with len(perm) > 4...
            out = tf.reshape(conv_rr, [-1, self.channel, conv_rr.shape[1] * conv_rr.shape[2], self.filters])
            out = tf.transpose(out, [0, 2, 3, 1])  # [batch, hh * ww, kernel * kernel, channnel]
            if self.compressed:
                out = tf.reshape(out, [-1, conv_rr.shape[1], conv_rr.shape[2], self.filters * self.channel])
            else:
                out = tf.reshape(out, [-1, conv_rr.shape[1], conv_rr.shape[2], self.kernel_size, self.kernel_size, self.channel])
        else:
            out = tf.image.extract_patches(inputs, self._sizes_, self._strides_, self._rates_, "VALID")
            if not self.compressed:
                # [batch, hh, ww, kernel, kernel, channnel]
                out = tf.reshape(out, [-1, out.shape[1], out.shape[2], self.kernel_size, self.kernel_size, self.channel])
        return out

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "sizes": self.sizes,
                "strides": self.strides,
                "rates": self.rates,
                "padding": self.padding,
                "compressed": self.compressed,
                "force_conv": self.force_conv,
            }
        )
        return base_config


"""
Creates a EfficientNetV2 Model as defined in: Mingxing Tan, Quoc V. Le. (2021). arXiv preprint arXiv:2104.00298.
EfficientNetV2: Smaller Models and Faster Training.
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

TF_BATCH_NORM_EPSILON = 0.001
TORCH_BATCH_NORM_EPSILON = 1e-5


FILE_HASH_DICT = {
    "v1-b0": {"imagenet": "cc7d08887de9df8082da44ce40761986"},
    "v1-b1": {"imagenet": "a967f7be55a0125c898d650502c0cfd0"},
    "v1-b2": {"imagenet": "6c8d1d3699275c7d1867d08e219e00a7"},
    "v1-b3": {"imagenet": "d78edb3dc7007721eda781c04bd4af62"},
    "v1-b4": {"imagenet": "4c83aa5c86d58746a56675565d4f2051"},
    "v1-b5": {"imagenet": "0bda50943b8e8d0fadcbad82c17c40f5"},
    "v1-b6": {"imagenet": "da13735af8209f675d7d7d03a54bfa27"},
    "v1-b7": {"imagenet": "d9c22b5b030d1e4f4c3a96dbf5f21ce6"},
}


def inverted_residual_block(
    inputs,
    output_channel,
    stride,
    expand,
    shortcut,
    kernel_size=3,
    drop_rate=0,
    se_ratio=0,
    is_fused=False,
    is_torch_mode=False,
    se_activation=None,  # None for same with activation
    se_divisor=1,  # 8 for mobilenetv3
    se_limit_round_down=0.9,  # 0.95 for fbnet
    use_global_context_instead_of_se=False,
    activation="swish",
    name=None,
):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_channel = inputs.shape[channel_axis]
    bn_eps = TORCH_BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON
    hidden_channel = make_divisible(input_channel * expand, 8)

    if is_fused and expand != 1:
        nn = conv2d_no_bias(inputs, hidden_channel, 3, stride, padding="same", use_torch_padding=is_torch_mode, name=name and name + "sortcut_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name and name + "sortcut_")
    elif expand != 1:
        nn = conv2d_no_bias(inputs, hidden_channel, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name=name and name + "sortcut_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name and name + "sortcut_")
    else:
        nn = inputs

    if not is_fused:
        if is_torch_mode and kernel_size // 2 > 0:
            nn = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name and name + "pad")(nn)
            padding = "VALID"
        else:
            padding = "SAME"
        nn = keras.layers.DepthwiseConv2D(kernel_size, padding=padding, strides=stride, use_bias=False, name=name and name + "MB_dw_")(nn)
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name and name + "MB_dw_")

    if se_ratio > 0:
        se_activation = activation if se_activation is None else se_activation
        se_ratio = se_ratio / expand
        if use_global_context_instead_of_se:
            nn = global_context_module(nn, use_attn=True, ratio=se_ratio, divisor=1, activation=se_activation, use_bias=True, name=name and name + "gc_")
        else:
            nn = se_module(nn, se_ratio, divisor=se_divisor, limit_round_down=se_limit_round_down, activation=se_activation, name=name and name + "se_")

    # pw-linear
    if is_fused and expand == 1:
        nn = conv2d_no_bias(nn, output_channel, 3, strides=stride, padding="same", use_torch_padding=is_torch_mode, name=name and name + "fu_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name and name + "fu_")
    else:
        nn = conv2d_no_bias(nn, output_channel, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name=name and name + "MB_pw_")
        nn = batchnorm_with_activation(nn, activation=None, epsilon=bn_eps, name=name and name + "MB_pw_")

    if shortcut:
        nn = drop_block(nn, drop_rate, name=name and name + "drop")
        return keras.layers.Add(name=name and name + "output")([inputs, nn])
    else:
        return keras.layers.Activation("linear", name=name and name + "output")(nn)  # Identity, Just need a name here


def EfficientNetV2(
    expands=[1, 4, 4, 4, 6, 6],
    out_channels=[16, 32, 48, 96, 112, 192],
    depthes=[1, 2, 2, 3, 5, 8],
    strides=[1, 2, 2, 2, 1, 2],
    se_ratios=[0, 0, 0, 0.25, 0.25, 0.25],
    is_fused="auto",  # True if se_ratio == 0 else False
    first_conv_filter=32,
    output_conv_filter=1280,
    kernel_sizes=3,
    input_shape=(None, None, 3),
    num_classes=1000,
    dropout=0.2,
    first_strides=2,
    is_torch_mode=False,
    use_global_context_instead_of_se=False,
    drop_connect_rate=0,
    activation="swish",
    classifier_activation="softmax",
    include_preprocessing=False,
    pretrained="imagenet",
    model_name="EfficientNetV2",
    rescale_mode="torch",
    kwargs=None,
):
    # "torch" for all V1 models
    # for V2 models, "21k" pretrained are all "tf", "imagenet" pretrained "bx" models are all "torch", ["s", "m", "l", "xl"] are "tf"
    rescale_mode = "tf" if pretrained is not None and pretrained.startswith("imagenet21k") else rescale_mode

    inputs = keras.layers.Input(shape=input_shape)
    bn_eps = TORCH_BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON

    if include_preprocessing and rescale_mode == "torch":
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        Normalization = keras.layers.Normalization if hasattr(keras.layers, "Normalization") else keras.layers.experimental.preprocessing.Normalization
        mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
        std = (tf.constant([0.229, 0.224, 0.225]) * 255.0) ** 2
        nn = Normalization(mean=mean, variance=std, axis=channel_axis)(inputs)
    elif include_preprocessing and rescale_mode == "tf":
        Rescaling = keras.layers.Rescaling if hasattr(keras.layers, "Rescaling") else keras.layers.experimental.preprocessing.Rescaling
        nn = Rescaling(scale=1.0 / 128.0, offset=-1)(inputs)
    else:
        nn = inputs
    stem_width = make_divisible(first_conv_filter, 8)
    nn = conv2d_no_bias(nn, stem_width, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="stem_")

    blocks_kwargs = {  # common for all blocks
        "is_torch_mode": is_torch_mode,
        "use_global_context_instead_of_se": use_global_context_instead_of_se,
    }

    pre_out = stem_width
    global_block_id = 0
    total_blocks = sum(depthes)
    kernel_sizes = kernel_sizes if isinstance(kernel_sizes, (list, tuple)) else ([kernel_sizes] * len(depthes))
    for id, (expand, out_channel, depth, stride, se_ratio, kernel_size) in enumerate(zip(expands, out_channels, depthes, strides, se_ratios, kernel_sizes)):
        out = make_divisible(out_channel, 8)
        if is_fused == "auto":
            cur_is_fused = True if se_ratio == 0 else False
        else:
            cur_is_fused = is_fused[id] if isinstance(is_fused, (list, tuple)) else is_fused
        for block_id in range(depth):
            name = "stack_{}_block{}_".format(id, block_id)
            stride = stride if block_id == 0 else 1
            shortcut = True if out == pre_out and stride == 1 else False
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = inverted_residual_block(
                nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se_ratio, cur_is_fused, **blocks_kwargs, activation=activation, name=name
            )
            pre_out = out
            global_block_id += 1

    if output_conv_filter > 0:
        output_conv_filter = make_divisible(output_conv_filter, 8)
        nn = conv2d_no_bias(nn, output_conv_filter, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name="post_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="post_")
    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)

    model = keras.models.Model(inputs=inputs, outputs=nn, name=model_name)
    reload_model_weights(model, pretrained)
    return model


def reload_model_weights(model, pretrained="imagenet"):
    if pretrained is None:
        return
    if isinstance(pretrained, str) and pretrained.endswith(".h5"):
        print(">>>> Load pretrained from:", pretrained)
        model.load_weights(pretrained, by_name=True, skip_mismatch=True)
        return

    pretrained_dd = {"imagenet": "imagenet"}
    if not pretrained in pretrained_dd:
        print(">>>> No pretrained available, model will be randomly initialized")
        return
    pre_tt = pretrained_dd[pretrained]
    model_type = model.name.split("_")[-1]
    if model_type not in FILE_HASH_DICT or pre_tt not in FILE_HASH_DICT[model_type]:
        print(">>>> No pretrained available, model will be randomly initialized")
        return

    pre_url = "https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnet{}-{}.h5"

    url = pre_url.format(model_type, pre_tt)
    file_name = os.path.basename(url)
    file_hash = FILE_HASH_DICT[model_type][pre_tt]

    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models/efficientnetv2", file_hash=file_hash)
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return
    else:
        print(">>>> Load pretrained from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)


"""
Creates a EfficientNetV1 Model as defined in: EfficientNetV1: Self-training with Noisy Student improves ImageNet classification.
arXiv preprint arXiv:1911.04252.
"""
import tensorflow as tf



def get_expanded_width_depth(width, depth, fix_head_stem=False):
    out_channels = [ii * width for ii in [16, 24, 40, 80, 112, 192, 320]]
    depthes = [int(tf.math.ceil(ii * depth)) for ii in [1, 2, 2, 3, 3, 4, 1]]
    if fix_head_stem:
        depthes[0], depthes[-1] = 1, 1
        first_conv_filter, output_conv_filter = 32, 1280
    else:
        first_conv_filter = 32 * width
        output_conv_filter = 1280 * width
    return out_channels, depthes, first_conv_filter, output_conv_filter


def EfficientNetV1(
    expands=[1, 6, 6, 6, 6, 6, 6],
    out_channels=[16, 24, 40, 80, 112, 192, 320],
    depthes=[1, 2, 2, 3, 3, 4, 1],
    strides=[1, 2, 2, 2, 1, 2, 1],
    se_ratios=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    first_conv_filter=32,
    output_conv_filter=1280,
    kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
    drop_connect_rate=0.2,
    pretrained="imagenet",
    model_name="EfficientNetV1",
    **kwargs,
):
    kwargs.pop("kwargs", None)
    return EfficientNetV2(**locals(), **kwargs)


def EfficientNetV1B0(input_shape=(224, 224, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.0, 1.0)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b0", **kwargs)


def EfficientNetV1B1(input_shape=(240, 240, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.0, 1.1)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b1", **kwargs)


def EfficientNetV1B2(input_shape=(260, 260, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.1, 1.2)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b2", **kwargs)


def EfficientNetV1B3(input_shape=(300, 300, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.2, 1.4)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b3", **kwargs)


def EfficientNetV1B4(input_shape=(380, 380, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.4, 1.8)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b4", **kwargs)


def EfficientNetV1B5(input_shape=(456, 456, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.6, 2.2)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b5", **kwargs)


def EfficientNetV1B6(input_shape=(528, 528, 3), num_classes=1000, dropout=0.5, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.8, 2.6)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b6", **kwargs)


def EfficientNetV1B7(input_shape=(600, 600, 3), num_classes=1000, dropout=0.5, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(2.0, 3.1)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b7", **kwargs)