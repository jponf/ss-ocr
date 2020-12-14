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
   n_classes: int
        Channels of last convolution. If using the FPN for detection this is 
        the number of classes
    activation: str
        Activation of the last convolution.
    inner_channels: int, default 256
        Filters to use inside the pyramid
    """
    def __init__(self, 
                 backbone: str,
                 n_classes: int,
                 activation: Optional[str] = None,
                 inner_channels: int = 256,
                 reduce_output: str = 'concat') -> None:

        super(FPNDetector, self).__init__()
        self.backbone_name = backbone
        self.image_shape = image_shape
        self.inner_channels = inner_channels
        self.reduce_output = reduce_output
        self.n_classes = n_classes
        self.activation = activation

        # Layer to merge all the feature maps resulting from the
        # differents FPN levels
        if self.reduce_output == 'concat':
            self.reduces_out_layer = tf.keras.layers.Concatenate(axis=-1)
        elif self.reduce_output == 'mean':
            self.reduces_out_layer = tf.keras.layers.Average()
        elif self.reduce_output == 'sum':
            self.reduces_out_layer = tf.keras.layers.Add()
        else:
            raise ValueError('Reduce output must be either "concat", "mean", '
                             f'"sum" or None. {reduce_output} is not supported')

        self.n_levels = 4 # Using 5 levels increases a lot the memory usage

        levels = list(range(2, self.n_levels + 2))

        # We feed each backbone feature map through a pixel-wise
        # so we reduce the number of channels to `inner_channels`
        self.pixel_wise_convs = [tf.keras.layers.Conv2D(inner_channels, 
                                                        kernel_size=1, 
                                                        name=f'pixel_wise_C{i}')
                                 for i in levels]

        # To reduce the antialiasing after merging feature maps
        self.antialias_convs = [tf.keras.layers.Conv2D(
                                    inner_channels,
                                    kernel_size=3,
                                    padding='same',
                                    name=f'antialiasing_P{i}') 
                                for i in levels[1:][::-1]]
        
        # After merging all the feature maps we apply a pixel-wise
        # to combine all the learned features at each FPN scale
        self.pool_reduced_conv = tf.keras.layers.Conv2D(
            inner_channels,
            kernel_size=1,
            padding='same',
            name='pool_merged_features')

        # Classification layer
        self.output_conv = tf.keras.layers.Conv2D(
            self.n_classes,
            kernel_size=3,
            padding='same',
            name='detection',
            activation=self.activation)

    def build(self, input_shape: tf.TensorShape) -> None:
        self.backbone = backbones.build_fpn_backbone(
            self.backbone_name,
            input_shape=input_shape[1:],
            n_levels=self.n_levels)

        super(UNet, self).build(input_shape)


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
