# -*- coding: utf-8 -*-

# pylint: disable=missing-module-docstring

from typing import Optional, Tuple, Union

import tensorflow as tf

from . import backbones

################################################################################


def build_fpn_recognizer(
        alphabet: str,
        input_shape: Union[Tuple[int, int, int], tf.TensorShape],
        backbone_name: str = "mobilenetv2",
        output_len: Optional[int] = None,
        inner_channels: int = 256,
        output_channels: int = 512,
        reduce_method: str = "concat",
        upsample_method: str = "nearest",
        pre_lstm_units: Optional[int] = None,
        post_lstm_units: Optional[int] = None,
        lstm_steps_to_discard: int = 2,
        dropout_rate: int = 0.25,
        freeze_backbone: bool = False) -> tf.keras.Model:
    """Builds a recognizer based on a Feature Pyramid Network (FPN). FPN is
    describen on: "Feature Pyramid Networks for Object Detection"
    (https://arxiv.org/pdf/1612.03144.pdf)

    Parameters
    ----------
    alphabet : str
        Alphabet recognized by the model. It will be extended with a "blank"
        character for the CTC model.
    input_shape : tuple or tf.TensorShape
        Shape of the input tensor without the batch dimension.
        Image format should be: Height, Width, # Channels
    backbone_name : str
        Neural Network architecture to use as a backbone for the FPN. To
        list the supported models call `list_supported_models()`.
    output_len: int
        Output sequence maximum length.
    inner_channels : int
        Number of filters to use inside the pyramid.
    output_channels : int
        Number of filters that output the pyramid.
    reduce_method : str
        Operation to reduce the output of the different layers of the pyramid.
        Possible values are: "concat" (default), "mean" and "sum".
    upsample_method : str
        Interpolation method used when upsampling feature maps. Possible
        options are: bilinear, bicubic and nearest
    pre_lstm_units : int
        Number of units in the LSTM encoder layer.
    post_lstm_units : int
        Number of units in the LSTM decoder layer.
    lstm_steps_to_discard : int
        Number of LSTM steps that must be discarded. Usually because the first
        steps of the RNN has little information.
    dropout_rate : float
        Dropout layers rate.
    freeze_backbone : bool
        Whether or not backbone layers should be trainable or not (frozen).

    Returns
    -------
    model : tf.keras.Model
        The FPN-based model to recognize text from images. The output of this
        model is not post-processed and thus corresponds to the logits of each
        time step.
    backbone : tf.keras.Model
        Backbone used to extract features from the input images.
    """
    # pylint: disable=too-many-arguments
    # pylint: disable=too-many-locals

    if len(input_shape) != 3:
        raise ValueError("InputShape must have 3 components "
                         "(height, width, #channels)")

    # Infer some values if necesseary
    output_len = output_len or input_shape[0]
    pre_lstm_units = pre_lstm_units or len(alphabet) * 2
    post_lstm_units = post_lstm_units or len(alphabet) * 5

    input_tensor = tf.keras.Input(input_shape)

    # FPN [BEGIN]
    n_levels = 4

    backbone = backbones.build_fpn_backbone(
        name=backbone_name,
        input_tensor=input_tensor,
        n_levels=n_levels)

    if freeze_backbone:
        for layer in backbone.layers:
            layer.trainable = False

    features = _build_and_reduce_fpn(backbone, inner_channels, output_channels,
                                     upsample_method, reduce_method)
    # FPN [END]

    # 2D Features to Sequence (Tx = width)
    _, features_h, features_w, features_c = features.shape

    # After extracting the features we reshape the data to prepare it
    # for the sequence Encoder-Decoder model
    features = tf.keras.layers.Permute((2, 1, 3), name="permute_hw")(features)
    features = tf.keras.layers.Reshape(
        target_shape=(features_w, features_h * features_c),
        name="reshape_2d_to_seq")(features)

    # Attention [BEGIN]
    att_repeat_state = tf.keras.layers.RepeatVector(features_w,
                                                    name="repeat_h_state")
    att_concatenate = tf.keras.layers.Concatenate(axis=-1,
                                                  name="att_concat")
    att_dense1 = tf.keras.layers.Dense(units=output_channels,
                                       activation="tanh",
                                       name="att_intermediate_energies")
    att_dense2 = tf.keras.layers.Dense(units=1,
                                       activation="relu",
                                       name="att_energies")
    att_weights = tf.keras.layers.Softmax(name="att_weights")
    att_context = tf.keras.layers.Dot(axes=1, name="att_context")

    def _compute_attention(encoded: tf.Tensor,
                           prev_state: tf.Tensor) -> tf.Tensor:
        prev_state = att_repeat_state(prev_state)       # (m, Tx, post_units)
        concat = att_concatenate([encoded, prev_state])
        energies1 = att_dense1(concat)             # dense->tannh
        energies2 = att_dense2(energies1)          # dense->relu
        weights = att_weights(energies2)           # softmax
        return att_context([weights, encoded])     # dot mult
    # Attention [END]

    # Sequence Model [BEGIN]
    features = tf.keras.layers.Dropout(dropout_rate,
                                       name="fpn_dropout")(features)
    encoded_features = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(pre_lstm_units,
                             return_sequences=True,
                             name="lstm_encodre"),
        name="bilstm_encoder")(features)

    context_dropout = tf.keras.layers.Dropout(dropout_rate,
                                              name="context_dropout")
    post_lstm = tf.keras.layers.LSTM(post_lstm_units,
                                     return_state=True,
                                     name="lstm_decoder")
    alphabet_dense = tf.keras.layers.Dense(
        len(alphabet) + 1, kernel_initializer="he_normal",
        name="alphabet")
    alphabet_softmax = tf.keras.layers.Softmax(name="alphabet_softmax")

    # Generate LSTM initial state (zero vector)
    batch_size = tf.keras.layers.Lambda(lambda x: tf.shape(x)[0],
                                        name="batch_size")(input_tensor)
    h_state = tf.keras.layers.Lambda(lambda x: tf.zeros([x, post_lstm_units]),
                                     name="h_state0")(batch_size)
    c_state = tf.keras.layers.Lambda(lambda x: tf.zeros([x, post_lstm_units]),
                                     name="c_state0")(batch_size)

    outputs = []
    for i in range(output_len + lstm_steps_to_discard):
        context = _compute_attention(encoded_features, h_state)
        context = context_dropout(context)

        h_state, _, c_state = post_lstm(context,
                                        initial_state=[h_state, c_state])
        if i >= lstm_steps_to_discard:
            out = alphabet_dense(h_state)
            outputs.append(alphabet_softmax(out))
    # Sequence Model [END]

    output = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1, name="stack"),
                                    name="stack_output")(outputs)
    return (tf.keras.Model(inputs=[input_tensor], outputs=output),
            backbone)


def compile_fpn_recognizer_with_ctc_loss(
        fpn_recognizer: tf.keras.Model,
        optimizer: Optional[tf.optimizers.Optimizer] = None) -> tf.keras.Model:
    """Creates and compiles a model to train the given `fpn_recognizer` model.

    Parameters
    ----------

    """
    optimizer = optimizer or tf.optimizers.Adam()
    max_seq_len = fpn_recognizer.output.shape[1]  # batch, seq_len, logits

    labels = tf.keras.Input(shape=[max_seq_len], dtype=tf.int32, name="label")
    label_length = tf.keras.Input(shape=[], dtype=tf.int32,
                                  name="label_length")
    logit_length = tf.keras.Input(shape=[], dtype=tf.int32,
                                  name="logit_length")

    loss = tf.keras.layers.Lambda(
        lambda inputs: tf.nn.ctc_loss(labels=inputs[0],
                                      logits=inputs[1],
                                      label_length=inputs[2],
                                      logit_length=inputs[3],
                                      logits_time_major=False),
        name="ctc_loss"
    )([labels, fpn_recognizer.output, label_length, logit_length])

    model = tf.keras.Model(inputs=[fpn_recognizer.input, labels,
                                   label_length, logit_length],
                           outputs=loss)
    model.compile(
        loss=lambda _, y_pred: y_pred,
        optimizer=optimizer)

    return model


def _build_and_reduce_fpn(backbone: tf.keras.Model,
                          inner_channels: int,
                          output_channels: int,
                          upsample_method: str,
                          reduce_method: str):
    feature_maps = _build_fpn(backbone, inner_channels, upsample_method)

    largest_size = feature_maps[0].shape[1:-1]
    reduce_upsample = tf.keras.layers.Lambda(
        lambda x: tf.image.resize(x, largest_size, method=upsample_method),
        name="fpn_reduce_upsample")

    feature_maps = [reduce_upsample(feats) for feats in feature_maps]

    # Reduce the differnt feature maps to `output_channels`
    if reduce_method == "concat":
        features = tf.keras.layers.Concatenate(
            axis=-1, name="fpn_reduce_concat")(feature_maps)
    elif reduce_method == "mean":
        features = tf.keras.layers.Average(name="fpn_reduce_mean")(feature_maps)
    elif reduce_method == "sum":
        features = tf.keras.layers.Add(name="fpn_reduce_sum")(feature_maps)
    else:
        raise ValueError("Reduce output must be either 'concat', 'mean' "
                         f"or 'sum'. {reduce_method} is not supported")

    return tf.keras.layers.Conv2D(filters=output_channels,
                                  kernel_size=1,
                                  padding='same',
                                  name='fpn_pool_reduced')(features)


def _build_fpn(backbone: tf.keras.Model,
               inner_channels: int,
               upsample_method: str):
    # pylint: disable=invalid-name
    # We feed each backbone feature map through a pixel-wise to
    # reduce the number of channels to `inner_channels`
    features = [
        tf.keras.layers.Conv2D(filters=inner_channels,
                               kernel_size=1,
                               name=f"pixel_wise_C{i+2}")(feats)
        for i, feats in enumerate(backbone.output)
    ]

    # Propagate deeper layers feature maps up and reduce antialiasing
    c2, c3, c4, c5 = features

    p5 = c5
    p5_up = tf.keras.layers.Lambda(
        lambda x: tf.image.resize(x, c4.shape[1:-1],
                                  method=upsample_method),
        name="upsample_P5")(p5)

    p4 = tf.keras.layers.Add(name="add_C4_P5")([c4, p5_up])
    p4_up = tf.keras.layers.Lambda(
        lambda x: tf.image.resize(x, c3.shape[1:-1],
                                  method=upsample_method),
        name="upsample_P4")(p4)
    p4 = tf.keras.layers.Conv2D(filters=inner_channels,
                                kernel_size=3,
                                padding="same",
                                name="antialiasing_P4")(p4)

    p3 = tf.keras.layers.Add(name="add_C3_P4")([c3, p4_up])
    p3_up = tf.keras.layers.Lambda(
        lambda x: tf.image.resize(x, c2.shape[1:-1],
                                  method=upsample_method),
        name="upsample_P3")(p3)
    p3 = tf.keras.layers.Conv2D(filters=inner_channels,
                                kernel_size=3,
                                padding="same",
                                name="antialiasing_P3")(p3)

    p2 = tf.keras.layers.Add(name="add_C2_P3")([c2, p3_up])
    p2 = tf.keras.layers.Conv2D(filters=inner_channels,
                                kernel_size=3,
                                padding="same",
                                name="antialiasing_P2")(p2)

    return p2, p3, p4, p5
