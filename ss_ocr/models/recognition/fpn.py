# -*- coding: utf-8 -*-

# pylint: disable=missing-module-docstring
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals

import collections
from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf

from ss_ocr.models import backbones, utils

################################################################################


class FpnRecognizerModel:
    """Builds a recognizer model based on a Feature Pyramid Network (FPN). FPN
    is describen on: "Feature Pyramid Networks for Object Detection"
    (https://arxiv.org/pdf/1612.03144.pdf)

    This class builds the encoder part and provides different methods to build
    different heads on top of it. After building them these models will be
    accessible as variables stored in the class itself.
    """

    def __init__(self,
                 alphabet: str,
                 input_shape: Union[Tuple[int, int, int], tf.TensorShape],
                 backbone_name: str = "mobilenetv2",
                 inner_channels: int = 256,
                 output_channels: int = 512,
                 reduce_method: str = "concat",
                 upsample_method: str = "nearest",
                 lstm_enc_units: Optional[Sequence[int]] = None,
                 lstm_enc_merge: str = "concat",
                 lstm_enc_input_feats: Optional[int] = None,
                 lstm_steps_to_discard: int = 2,
                 dropout_rate: int = 0.25,
                 freeze_backbone: bool = False):
        # set up config
        self.config = collections.OrderedDict([
            ("alphabet", alphabet),
            ("input_shape", input_shape),
            ("backbone_name", backbone_name),
            ("inner_channels", inner_channels),
            ("output_channels", output_channels),
            ("reduce_method", reduce_method),
            ("upsample_method", upsample_method),
            ("lstm_enc_units", lstm_enc_units),
            ("lstm_enc_merge", lstm_enc_merge),
            ("lstm_enc_input_feats", lstm_enc_input_feats),
            ("lstm_steps_to_discard", lstm_steps_to_discard),
            ("dropout_rate", dropout_rate),
            ("freeze_backbone", freeze_backbone)
        ])

        self.encoder, self.backbone = build_encoder(
            alphabet=alphabet,
            input_shape=input_shape,
            backbone_name=backbone_name,
            inner_channels=inner_channels,
            reduce_method=reduce_method,
            upsample_method=upsample_method,
            lstm_enc_units=lstm_enc_units,
            lstm_enc_merge=lstm_enc_merge,
            lstm_enc_input_feats=lstm_enc_input_feats,
            freeze_backbone=freeze_backbone)
        self.att_model = None
        self.ctc_model = None
        self.ctc_att_model = None

    def build_attention_decoder(self):
        """
        """
        if self.att_model is None:
            self.att_model = build_attention_decoder(
                encoder=self.encoder,
                alphabet=self.config["alphabet"],
                lstm_steps_to_discard=self.config["lstm_steps_to_discard"],
                attention_type="luong",
                dropout_rate=self.config["dropout_rate"]
            )

    def build_ctc_decoder(self):
        """
        """
        if self.ctc_model is None:
            self.ctc_model = build_ctc_decoder(
                encoder=self.encoder,
                alphabet=self.config["alphabet"],
                lstm_steps_to_discard=self.config["lstm_steps_to_discard"],
                dropout_rate=self.config["dropout_rate"]
            )

    def build_ctc_joint_attention_decoder(self):
        """
        """
        if self.ctc_att_model is None:
            self.build_attention_decoder()
            self.build_ctc_decoder()
            self.ctc_att_model = tf.keras.Model(
                inputs=self.encoder.inputs,
                outputs=self.att_model.outputs + self.ctc_model.outputs)

    def compile_ctc_joint_attention(
            self,
            optimizer: Union[str, tf.optimizers.Optimizer],
            padding_value: int,
            ctc_loss_weight: float = 0.75,
            clip_norm: Optional[float] = None) -> utils.TrainingModel:
        # Losses
        att_loss = utils.PaddedSparseCategoricalCrossEntropyLoss(
            padding_value=padding_value,
            from_logits=True)

        ctc_loss = utils.CTCLoss(
            padding_value=padding_value,
            logits_time_major=False,
            blank_index=len(self.config["alphabet"])
        )

        # New model with ctc loss
        model = utils.TrainingModel(inputs=self.encoder.inputs,
                                    outputs={"attention": self.att_model.output,
                                            "ctc": self.ctc_model.output},
                                    clip_norm=clip_norm)

        model.compile(optimizer=optimizer,
                      loss={"attention": att_loss, "ctc": ctc_loss},
                      loss_weights={"attention": 1.0 - ctc_loss_weight,
                                    "ctc": ctc_loss_weight})
        return model

    def __repr__(self):
        return ("FpnRecognizer(" +
                "\n              ".join(f"{k}={repr(v)}"
                                        for k, v in self.config.items()) +
                ")")


################################################################################

def build_encoder(
        alphabet: str,
        input_shape: Union[Tuple[int, int, int], tf.TensorShape],
        backbone_name: str = "mobilenetv2",
        inner_channels: int = 256,
        output_channels: int = 512,
        reduce_method: str = "concat",
        upsample_method: str = "nearest",
        lstm_enc_units: Optional[Sequence[int]] = None,
        lstm_enc_merge: str = "concat",
        lstm_enc_input_feats: Optional[int] = None,
        dropout_rate: float = 0.25,
        freeze_backbone: bool = False) -> tf.keras.Model:
    """Builds a recognizer based on a Feature Pyramid Network (FPN). FPN is
    describen on: "Feature Pyramid Networks for Object Detection"
    (https://arxiv.org/pdf/1612.03144.pdf)

    The output of the model are the logits (no softmax applied) of each
    position. Remember to post-process them if necessary.

    Parameters
    ----------
    alphabet : str
        Alphabet recognized by the model.
    input_shape : tuple or tf.TensorShape
        Shape of the input tensor without the batch dimension.
        Image format should be: Height, Width, # Channels
    backbone_name : str
        Neural Network architecture to use as a backbone for the FPN. To
        list the supported models call `list_supported_models()`.
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
    lstm_enc_units : Sequence[int]
        Number of units in each LSTM encoder layer.
    lstm_enc_merge: str
        How the bidirectional encoder should merge the features of each
        timestep flowing in opposite  directions.
    lstm_enc_input_feats: Optional[int]
        Number of features in each time step for the encoder. The idea is
        reduce the features outputed by the FPN network to use less parameters,
        if not specified this reduction will not be applied.
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
    lstm_enc_units = lstm_enc_units or [len(alphabet) * 2] * 4
    lstm_enc_input_feats = lstm_enc_input_feats or max(lstm_enc_units)

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

    # Stacked LSTM Encoder [BEGIN]
    features = tf.keras.layers.Dense(lstm_enc_input_feats,
                                    activation="relu",
                                    name="pre_lstm_enc")(features)
    features = tf.keras.layers.SpatialDropout1D(
        dropout_rate, name="lstm_enc_dropout")(features)

    for i, units in enumerate(lstm_enc_units, start=1):
        if i < len(lstm_enc_units):
            name = f"bilstm_enc_{i}"
        else:
            name = "bilstm_enc_last"

        features = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(units,
                                 return_sequences=True,
                                 kernel_initializer="he_normal",
                                 name=f"lstm_enc_{i+1}"),
            merge_mode=lstm_enc_merge,
            name=name)(features)
    # Stacked LSTM Encoder [END]

    return (tf.keras.Model(inputs=[input_tensor], outputs=[features]),
            backbone)


def build_attention_decoder(encoder: tf.keras.Model,
                            alphabet: str,
                            lstm_steps_to_discard: int = 0,
                            attention_type: str = "luong",
                            dropout_rate: float = 0.25):
    """Builds an attention decoder head on top of the given FPN encoder.

    Parameters
    ----------
    encoder : tf.keras.Model
        Image encoder model.
    alphabet: str
        Alphabet recognized by the model.
    lstm_steps_to_discard : int
        Number of LSTM steps that must be discarded. Usually because the first
        steps of the RNN has little information.
    attention_type : str
        Attention type: luong, multiplicative, bahdanau or additive.
    dropout_rate: float
        Dropout layers rate.

    Returns
    -------
    decoder : tf.keras.Model
        A new keras model that connects an attention decoder head on top
        of the given encoder.
    """
    output_len = encoder.input.shape[1]
    lstm_dec_units = encoder.output.shape[2]

    # Build attention layer
    if attention_type.lower() in ("luong", "multiplicative"):
        attention = tf.keras.layers.Attention(dropout=dropout_rate,
                                              name="attention")
    elif attention_type.lower() in ("bahdanau", "additive"):
        attention = tf.keras.layers.AdditiveAttention(dropout=dropout_rate,
                                                      name="attention")
    else:
        raise ValueError("unknown attention type '{attention_type}', expected: "
                         "luong, multiplicative, bahdanau or additive")


    context_dropout = tf.keras.layers.Dropout(dropout_rate,
                                              name="att_context_dropout")
    add_time_dim = tf.keras.layers.Reshape((1, lstm_dec_units),
                                           name="add_time_dim")

    # LSTM and alphabet layers
    lstm_dec = tf.keras.layers.LSTM(lstm_dec_units,
                                    return_state=True,
                                    kernel_initializer="he_normal",
                                    name="att_lstm_decoder")
    alphabet_dense = tf.keras.layers.Dense(units=len(alphabet),
                                           kernel_initializer="he_normal",
                                           name="att_alphabet")

    # Generate LSTM initial state (zero vector)
    batch_size = tf.keras.layers.Lambda(lambda x: tf.shape(x)[0],
                                        name="att_batch_size")(encoder.output)
    h_state = tf.keras.layers.Lambda(lambda x: tf.zeros([x, lstm_dec_units]),
                                     name="att_h_state0")(batch_size)
    c_state = tf.keras.layers.Lambda(lambda x: tf.zeros([x, lstm_dec_units]),
                                     name="att_c_state0")(batch_size)

    # LSTM loop
    outputs = []
    for i in range(output_len):
        context = attention([add_time_dim(h_state), encoder.output])
        context = context_dropout(context)

        h_state, _, c_state = lstm_dec(context,
                                       initial_state=[h_state, c_state])
        if i >= lstm_steps_to_discard:
            out = alphabet_dense(h_state)
            outputs.append(out)

    output = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=1, name="stack"),
                                    name="att_stack_output")(outputs)
    return tf.keras.Model(inputs=encoder.inputs, outputs=output)


def build_ctc_decoder(encoder: tf.keras.Model,
                      alphabet: str,
                      lstm_steps_to_discard: int = 0,
                      dropout_rate: float = 0.25) -> tf.keras.Model:
    """Builds an attention decoder head on top of the given FPN encoder.

    Parameters
    ----------
    encoder : tf.keras.Model
        Image encoder model.
    alphabet: str
        Alphabet recognized by the model.
    lstm_steps_to_discard : int
        Number of LSTM steps that must be discarded. Usually because the first
        steps of the RNN has little information.
    dropout_rate: float
        Dropout layers rate.

    Returns
    -------
    decoder : tf.keras.Model
        A new keras model that connects an attention decoder head on top
        of the given encoder.
    """
    # Drop encoder steps if necessary
    if lstm_steps_to_discard > 0:
        output = tf.keras.layers.Lambda(
            lambda x: x[:, lstm_steps_to_discard:],
            name="ctc_discard")(encoder.output)
    else:
        output = encoder.output

    output = tf.keras.layers.SpatialDropout1D(rate=dropout_rate,
                                              name="ctc_dropout")(output)
    output = tf.keras.layers.Dense(
        units=len(alphabet) + 1,  # ctc_blank
        kernel_initializer="he_normal",
        name="ctc_alphabet")(output)

    return tf.keras.Model(inputs=encoder.inputs, outputs=output)


def compile_fpn_recongizer_with_cross_entropy_loss(
        fpn_recognizer: tf.keras.Model,
        optimizer: Union[str, tf.optimizers.Optimizer],
        padding_value: int,
        **kwargs) -> None:
    """Compiles the base FPN recognizer model to be trained with cross entropy
    loss

    Parameters
    ----------
    fpn_recognizer : tf.keras.Model
        A fpn recognizer model created by `build_fpn_recognizer`.
    optimizer : str or tf.optimizers.Optimizer
        The name of the optimizer or an optimizer instance.
    padding_value : int
        fpn_recognizer: tf.keras.Model,
        optimizer: Union[str, tf.optimizers.Optimizer],
        padding_value: int,
    **kwargs
        Additional tf.keras.Model.compile parameters.
    """
    loss = utils.PaddedSparseCategoricalCrossEntropyLoss(
        padding_value=padding_value,
        from_logits=True)

    fpn_recognizer.compile(loss=loss, optimizer=optimizer, **kwargs)


def compile_fpn_recognizer_with_ctc_loss(
        fpn_recognizer: tf.keras.Model,
        optimizer: Optional[tf.optimizers.Optimizer] = None,
        clip_norm: Optional[float] = None,
        blank_index: int = 0) -> tf.keras.Model:
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
        lambda inputs: tf.nn.ctc_loss(
            labels=inputs[0],
            logits=inputs[1],
            label_length=inputs[2],
            logit_length=inputs[3],
            logits_time_major=False,
            blank_index=blank_index),
        name="ctc_loss"
    )([labels, fpn_recognizer.output, label_length, logit_length])

    model = utils.TrainingModel(
        inputs=[fpn_recognizer.input, labels,
                label_length, logit_length],
        outputs=loss,
        clip_norm=clip_norm)

    model.compile(
        optimizer=optimizer,
        loss=lambda _, y_pred: y_pred)

    return model


################################################################################

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
