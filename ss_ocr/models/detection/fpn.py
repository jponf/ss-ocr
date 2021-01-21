# -*- coding: utf-8 -*-

from typing import Optional, Sequence, Tuple, Union

import tensorflow as tf

from ss_ocr.models import backbones, utils

_SUPPORTED_MODELS = '\n'.join(f'- {o}' 
                              for o in backbones.list_supported_models())


def build_fpn_detector(
        input_shape: Union[Tuple[int, int, int], tf.TensorShape],
        backbone: str,
        n_classes: int,
        activation: Optional[Union[str, tf.keras.layers.Activation]] = None,
        inner_channels: int = 256,
        reduce_method: str = 'concat',
        upsample_method: str = 'nearest',
        freeze_backbone: bool = False) -> tf.keras.Model:

    f"""
    Builds the Feature Pyramid Network (FPN) described on 
    "Feature Pyramid Networks for Object Detection" 
    (https://arxiv.org/pdf/1612.03144.pdf)

    Parameters
    ----------
    input_shape : tuple or tf.TensorShape
        Shape of the input tensor without the batch dimension.
        Image format should be: Height, Width, # Channels
    backbone: str
        Neural Network architecture to use as a backbone. As now we support:
        {_SUPPORTED_MODELS}
    n_classes: int
        Channels of last convolution. If using the FPN for detection this is 
        the number of classes
    activation: str
        Activation of the last convolution.
    inner_channels: int, default 256
        Filters to use inside the pyramid
    upsample_method : str, default "nearest"
        Interpolation method used when upsampling feature maps. Possible
        options are: bilinear, bicubic and nearest
    freeze_backbone : bool
        Whether or not backbone layers should be trainable or not (frozen).
    """

    input_tensor = tf.keras.Input(input_shape)

    n_levels = 4

    backbone = backbones.build_fpn_backbone(
        name=backbone,
        input_tensor=input_tensor,
        n_levels=n_levels)
    
    if freeze_backbone:
        for layer in backbone.layers:
            layer.trainable = False

    features = utils.build_and_reduce_fpn(backbone, inner_channels, 
                                          n_classes, upsample_method, 
                                          reduce_method)

    if isinstance(activation, str):
        output = tf.keras.layers.Activation(activation)(features)
    elif isinstance(activation, tf.keras.layers.Activation):
        output = activation(features)
    else:
        raise ValueError(f'Type {type(activation)} not supported. '
                          'activation must be either str or Activation')

    return tf.keras.Model(input_tensor, output)

