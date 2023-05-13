import tensorflow as tf


def _make_divisible(v, divisor=8, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def ConvBN_layer(inputs, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 initializer="he_normal"):

    pad = tf.keras.layers.ZeroPadding2D(padding=(padding, padding))
    conv = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=(kernel_size, kernel_size),
        strides=(stride, stride),
        padding='valid',
        dilation_rate=dilation,
        use_bias=False,
        groups=groups
    )
    bn = tf.keras.layers.BatchNormalization()

    x = pad(inputs)
    x = conv(x)
    return bn(x)


def ConvBNReLU_layer(x, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     initializer="he_normal"):
    """

    :param x:
    :param in_channels:
    :param out_channels:
    :param kernel_size:
    :param stride:
    :param padding:
    :param dilation:
    :param groups:
    :param initializer:
    :return:
    """
    x = tf.keras.layers.ZeroPadding2D(padding=(padding, padding))(x)
    x = tf.keras.layers.Conv2D(
        filters=out_channels,
        kernel_size=(kernel_size, kernel_size),
        strides=(stride, stride),
        padding='valid',
        dilation_rate=dilation, use_bias=False,
        groups=groups
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x


def CascadePreExBottleneck_layer(inputs, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, expand_ratio=6,
                                 reduce_factor=4, block_type='CAS'):
    # Set the block type
    if in_channels // reduce_factor < 8:
        block_type = 'MB'

    # Set the expansion ratio channels
    r_channels = in_channels // reduce_factor

    # Whether we are reducing the channels
    if stride == 1 and in_channels == out_channels:
        reduction = False
    else:
        reduction = True

    # If we are not expanding the channels
    if expand_ratio == 1:
        squeeze_conv = None
        conv1 = None
        n_channels = in_channels
    else:
        # If the block type is CAS
        if block_type == 'CAS':
            n_channels = r_channels + in_channels
        else:
            n_channels = in_channels

    if not expand_ratio == 1:
        if block_type == 'CAS':
            squeezed = ConvBNReLU_layer(inputs, in_channels, r_channels, 1, dilation=dilation)
            out = tf.keras.layers.Concatenate()([squeezed, inputs])
        else:
            out = inputs
        out = ConvBNReLU_layer(out, n_channels, n_channels * expand_ratio, 1, dilation=dilation)
    else:
        out = inputs

    out = ConvBNReLU_layer(out, n_channels * expand_ratio, n_channels * expand_ratio, kernel_size, stride,
                           (kernel_size - 1) // 2, 1, groups=n_channels * expand_ratio)

    out = ConvBN_layer(out, n_channels * expand_ratio, out_channels, 1, dilation=dilation)

    if not reduction:
        out = tf.keras.layers.Add()([inputs, out])

    return out


def make_layer(x, in_channels, block_setting, width_mult, dilation=1, initializer="he_normal"):

    for k, c, e, r, s in block_setting:

        # Make sure the output channels are divisible
        out_channels = _make_divisible(int(c * width_mult))

        # Add the layer with the specific settings
        x = CascadePreExBottleneck_layer(
            x,
            in_channels,
            out_channels,
            kernel_size=k,
            stride=s,
            dilation=dilation,
            expand_ratio=e,
            reduce_factor=r
        )

        # The next layer will have in_channels = out_channels
        in_channels = out_channels

    return x


def frostnet(
        input_shape=(224, 224, 3),
        num_classes=1000,
        width_mult=1.0,
        dropout_ratio=0.2,
        quantize=False,
        drop_rate=0.2,
        dilated=False,
        initializer="he_normal"):

    layer1_setting = [
        # kernel_size, c, e, r, s
        [3, 16, 1, 1, 1],  # 0
        [5, 24, 3, 4, 2],  # 1
        [3, 24, 3, 4, 1],  # 2
        # [, , , , ],      #3
    ]
    layer2_setting = [
        [5, 40, 3, 4, 2],  # 4
        # [, , , , ],      #5
        # [, , , , ],      #6
    ]

    layer3_setting = [
        [5, 80, 3, 4, 2],  # 7
        [5, 80, 3, 4, 1],  # 8
        [3, 80, 3, 4, 1],  # 9

        [5, 96, 3, 2, 1],  # 10
        [5, 96, 3, 4, 1],  # 11
        [5, 96, 3, 4, 1],  # 12
    ]

    layer4_setting = [
        [5, 192, 6, 4, 2],  # 13
        [5, 192, 6, 4, 1],  # 14
        [5, 192, 6, 4, 1],  # 15
    ]

    layer5_setting = [
        [5, 320, 6, 2, 1],  # 16
    ]

    # The input layer
    input_layer = tf.keras.Input(shape=input_shape)

    in_channels = _make_divisible(int(32 * min(1.0, width_mult)))

    conv1 = ConvBNReLU_layer(input_layer, 3, in_channels, 3, 2, 1, initializer=initializer)

    layer1 = make_layer(conv1, in_channels, layer1_setting, width_mult, 1)
    layer2 = make_layer(layer1, in_channels, layer2_setting, width_mult, 1)
    layer3 = make_layer(layer2, in_channels, layer3_setting, width_mult, 1)

    # Whether to dialate layer 4 and 5
    if dilated:
        dilation = 2
    else:
        dilation = 1

    layer4 = make_layer(layer3, in_channels, layer4_setting, width_mult, dilation)
    layer5 = make_layer(layer4, in_channels, layer5_setting, width_mult, dilation)

    # building last several layers
    last_in_channels = in_channels

    last_layer = ConvBNReLU_layer(layer5, last_in_channels, 1280, 1, initializer=initializer)

    last_layer_shape = last_layer.shape.as_list()

    adv_avg = tf.keras.layers.AveragePooling2D(
        pool_size=(last_layer_shape[1], last_layer_shape[2]),
        strides=(last_layer_shape[1], last_layer_shape[2])
    )(last_layer)

    dropout = tf.keras.layers.Dropout(rate=drop_rate)(adv_avg)
    conv2d = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        strides=(1, 1),
        use_bias=True
    )(dropout)

    output = tf.keras.layers.Flatten()(conv2d)

    return tf.keras.Model(
        inputs=input_layer,
        outputs=output
    )


if __name__ == "__main__":
    import tensorflow_model_optimization as tfmot

    model = frostnet(input_shape=(224, 224, 3))

    qat_model = tfmot.quantization.keras.quantize_model(model)

    qat_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    qat_model.summary()
