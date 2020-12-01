# -*- coding: utf-8 -*-

from typing import Optional, Sequence, Tuple

import tensorflow as tf

from . import backbones

_SUPPORTED_MODELS = '\n'.join(f'- {o}' 
                              for o in backbones.list_supported_models())

class FPNDetector(tf.keras.Model):

    f"""
    Builds the Feature Pyramid Network (FPN) described on 
    "Feature Pyramid Networks for Object Detection" 
    (https://arxiv.org/pdf/1612.03144.pdf)

    Parameters
    ----------
    backbone: str
        Neural Network architecture to use as a backbone. As now we support:
        {_SUPPORTED_MODELS}
    image_shape: Tuple[int, int, int]
        Shape tuple to specify the input image size, including the channels
    output_filters: int
        Channels of last convolution. If using the FPN for detection this is 
        the number of classes
    output_activation: str
        Activation of the last convolution.
    inner_channels: int, default 256
        Filters to use inside the pyramid
    """
    def __init__(self, 
                 backbone: str,
                 image_shape: Tuple[int, int, int], 
                 output_filters: int,
                 output_activation: Optional[str] = None,
                 inner_channels: int = 256,
                 reduce_output: str = 'concat') -> None:

        super(FPNDetector, self).__init__()
        self.backbone_name = backbone
        self.image_shape = image_shape
        self.inner_channels = inner_channels
        self.reduce_output = reduce_output
        self.output_filters = output_filters
        self.output_activation = output_activation

        if self.reduce_output == 'concat':
            self.reduces_out_layer = tf.keras.layers.Concatenate(axis=-1)
        elif self.reduce_output == 'mean':
            self.reduces_out_layer = tf.keras.layers.Average()
        elif self.reduce_output == 'sum':
            self.reduces_out_layer = tf.keras.layers.Add()
        else:
            raise ValueError('Reduce output must be either "concat", "mean", '
                             f'"sum" or None. {reduce_output} is not supported')

        n_levels = 4 # Using 5 levels increases a lot the memory usage
        self.backbone = backbones.build_fpn_backbone(
            self.backbone_name, 
            input_shape=image_shape,
            n_levels=n_levels)

        levels = list(range(2, n_levels + 2))
        self.pixel_wise_convs = [tf.keras.layers.Conv2D(inner_channels, 
                                                        kernel_size=1, 
                                                        name=f'pixel_wise_C{i}')
                                 for i in levels]

        self.antialias_convs = [tf.keras.layers.Conv2D(
                                    inner_channels,
                                    kernel_size=3,
                                    padding='same',
                                    name=f'antialiasing_P{i}') 
                                for i in levels[1:][::-1]]

        self.pool_reduced_conv = tf.keras.layers.Conv2D(
            inner_channels,
            kernel_size=1,
            padding='same',
            name='pool_merged_features')

        self.output_conv = tf.keras.layers.Conv2D(
            self.output_filters,
            kernel_size=3,
            padding='same',
            name='detection',
            activation=self.output_activation)

    def call(self, 
             x: tf.Tensor, 
             training: Optional[bool] = None) -> Sequence[tf.Tensor]:

        features = self.backbone(x)
        features = [self.pixel_wise_convs[i](o, training=training) 
                    for i, o in enumerate(features)]

        C2, C3, C4, C5 = features

        P5 = C5
        P5_up = tf.image.resize(P5, C4.shape[1:-1], method='nearest')

        P4 = C4 + P5_up
        P4_up = tf.image.resize(P4, C3.shape[1:-1], method='nearest')
        P4 = self.antialias_convs[0](P4, training=training)

        P3 = C3 + P4_up
        P3_up = tf.image.resize(P3, C2.shape[1:-1], method='nearest')
        P3 = self.antialias_convs[1](P3, training=training)

        P2 = P3_up + C2
        P2 = self.antialias_convs[2](P2, training=training)

        largest_size = P2.shape[1:-1]
        
        x = self.reduces_out_layer([
            tf.image.resize(o, largest_size, method='nearest') 
            for o in (P2, P3, P4, P5)])
        x = self.pool_reduced_conv(x)

        return self.output_conv(x)
