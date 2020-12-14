# -*- coding: utf-8

from typing import Optional, Tuple, Sequence

import tensorflow as tf

from . import backbones

_SUPPORTED_MODELS = '\n'.join(f'- {o}' 
                              for o in backbones.list_supported_models())


class _Conv3x3BnReLU(tf.keras.layers.Layer):

    def __init__(self, 
                 filters: int, 
                 use_batchnorm: bool = True,
                 n_layers: int = 1,
                 name: Optional[str] = None, 
                 **kwargs) -> None:

        self.filters = filters
        self.use_batchnorm = use_batchnorm
        self.n_layers = n_layers

        self.conv = [tf.keras.layers.Conv2D(filters, 
                                            kernel_size=3, 
                                            padding='same',
                                            name=f'conv_3_3_{i + 1}')
                     for i in range(self.n_layers)]

        self.relu = tf.keras.layers.ReLU()

        if self.use_batchnorm:
            self.bn = [tf.keras.layers.BatchNormalization(name=f'bn_{i + 1}') 
                       for i in range(self.n_layers)]

        super(_Conv3x3BnReLU, self).__init__(name=name, **kwargs)

    def call(self, 
             x: tf.Tensor, 
             training: Optional[bool] = None) -> tf.Tensor:

        for i in range(self.n_layers):
            x = self.conv[i](x)

            if self.use_batchnorm:
                x = self.bn[i](x)

            x = self.relu(x)

        return x


class _DecoderBlock(tf.keras.layers.Layer):

    def __init__(self, 
                 filters: int, 
                 use_batchnorm: bool = True, 
                 upsample_strategy: str = 'upsample') -> None:

        super(_DecoderBlock, self).__init__()

        self.filters = filters
        self.use_batchnorm = use_batchnorm
        self.upsample_strategy = upsample_strategy

        if self.upsample_strategy == 'conv':
            self.upsample = tf.keras.layers.Conv2DTranspose(
                filters, kernel_size=(4, 4), strides=(2, 2), padding='same',
                name='upsample_conv')

            if self.use_batchnorm:
                self.bn = tf.keras.layers.BatchNormalization()
                self.relu = tf.keras.layers.ReLU()

            self.conv = _Conv3x3BnReLU(filters, self.use_batchnorm)

        elif self.upsample_strategy == 'upsample':
            self.upsample = tf.keras.layers.UpSampling2D(name='upsample')
            self.conv = _Conv3x3BnReLU(filters, self.use_batchnorm, n_layers=2)
            # Set to False to avoid using bn layer on `call` method
            self.use_batchnorm = False

        else:
            raise ValueError('`upsample strategy` must be either "upsample" '
                             'or "conv"')

    def call(self, 
             x: Tuple[tf.Tensor, tf.Tensor], 
             training: Optional[bool] = None) -> tf.Tensor:

        x, residual = x

        x = self.upsample(x)
        if self.use_batchnorm:
            x = self.bn(x)
            x = self.relu(x)

        if residual is not None:
            x = tf.concat([x, residual], axis=-1)

        return self.conv(x)


class UNet(tf.keras.Model):
    f"""
    Builds a modified version of the UNet described on 
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    (https://arxiv.org/abs/1505.04597)

    Parameters
    ----------
    backbone: str
        Neural Network architecture to use as a backbone. As now we support:
        {_SUPPORTED_MODELS}
    n_classes: int
        Channles of the las convolution. If using the model directly for 
        detection this translates to the number of classes
    activation: str
        Activation of the last convolution
    decoder_filters: Sequence[int], default (256, 128, 64, 32, 16)
        Decoder is composed of five layers. This is the convolutional filters
        used in each decoder layer.
    use_batchnorm: bool, default True
        Whether to use Batch Normalization after convolutions or not.
    upsample_strategy: default 'upsample'
        Method to upsample feature maps in the decoder. It can either be:
         - "upsample": Uses nearest interpolation to upsample feature maps.
         - "conv": Uses Transposed convs to upsample the feature maps.
    """
    def __init__(self, 
                 backbone: str,
                 n_classes: int,
                 activation: str = 'sigmoid',
                 decoder_filters: Sequence[int] = (256, 128, 64, 32, 16),
                 use_batchnorm: bool = True, 
                 upsample_strategy: str = 'upsample') -> None:

        super(UNet, self).__init__()

        if len(decoder_filters) != 5:
            raise ValueError("Decoder has 5 layers."
                             f"You specified {len(decoder_filters)} filters")

        self.backbone_name = backbone
        self.n_classes = n_classes
        self.activation = activation
        self.decoder_filters = decoder_filters
        self.use_batchnorm = use_batchnorm
        self.upsample_strategy = upsample_strategy

        self.n_levels = 5

        self.decoders = [_DecoderBlock(filters=f, 
                                       use_batchnorm=self.use_batchnorm,
                                       upsample_strategy=self.upsample_strategy)
                         for f in decoder_filters]

        self.output_conv = tf.keras.layers.Conv2D(self.n_classes,
                                                  kernel_size=3,
                                                  activation=self.activation,
                                                  padding='same')

    def build(self, input_shape: tf.TensorShape) -> None:
        self.backbone = backbones.build_fpn_backbone(
            self.backbone_name,
            input_shape=input_shape[1:],
            n_levels=self.n_levels)

        super(UNet, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:

        residuals = self.backbone(x)[::-1]

        x = residuals[0]
        for i in range(self.n_levels):
            if i < len(residuals) - 1:
                residual = residuals[i + 1]
            else:
                residual = None
            x = self.decoders[i]([x, residual])

        return self.output_conv(x)


if __name__ == "__main__":
    model = UNet('mobilenetv2', 3, upsample_strategy='conv')
    model.build([None, 512, 512, 3])

    im = tf.random.uniform([1, 512, 512, 3])
    out = model(im)
    print(out.shape)

