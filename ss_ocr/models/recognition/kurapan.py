# -*- coding:utf-8 -*-

from typing import Sequence

import numpy as np
import tensorflow as tf


################################################################################

class _STN(tf.keras.layers.Layer):

    def __init__(self):
        super(_STN, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(16, (5, 5), padding='same',
                                             activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(32, (5, 5), padding='same', 
                                             activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense_1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(6, 
                weights=[tf.zeros((64, 6), dtype='float32'),
                         tf.constant([1, 0, 0, 0, 1, 0], dtype=tf.float32)])

    def _transform(self, 
                   locnet_x: tf.Tensor, 
                   locnet_y: tf.Tensor) -> tf.Tensor:

        output_size = [self.height, self.width, self.filters]
        batch_size = tf.shape(locnet_x)[0]
        height = self.height
        width = self.width
        num_channels = self.filters

        locnet_y = tf.reshape(locnet_y, shape=(batch_size, 2, 3))

        locnet_y = tf.reshape(locnet_y, (-1, 2, 3))
        locnet_y = tf.cast(locnet_y, 'float32')

        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = _meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1])
        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, tf.stack([batch_size, 3, -1]))

        transformed_grid = tf.matmul(locnet_y, indices_grid)
        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x = tf.reshape(x_s, [-1])
        y = tf.reshape(y_s, [-1])

        # Interpolate
        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width = output_size[1]

        x = tf.cast(x, dtype='float32')
        y = tf.cast(y, dtype='float32')
        x = .5 * (x + 1.0) * width_float
        y = .5 * (y + 1.0) * height_float

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1, dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width * height
        pixels_batch = tf.range(batch_size) * flat_image_dimensions
        flat_output_dimensions = output_height * output_width
        base = _repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0 * width
        base_y1 = base + y1 * width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(locnet_x, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        transformed_image = tf.add_n([
            area_a * pixel_values_a, area_b * pixel_values_b, area_c * pixel_values_c,
            area_d * pixel_values_d
        ])
        # Finished interpolation

        return tf.reshape(
                transformed_image,
                [batch_size, output_height, output_width, num_channels])

    def build(self, input_shape: tf.TensorShape) -> None:
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.filters = input_shape[3]
        super(_STN, self).build(input_shape)

    def call(self, inp: tf.Tensor) -> tf.Tensor:
        x = self.conv_1(inp)
        x = self.conv_2(x)
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self._transform(inp, x)


class _ConvStack(tf.keras.layers.Layer):

    def __init__(self, 
                 filters: Sequence[int],
                 pool_size: int = 0,
                 use_bn: bool = True,
                 name: str = 'conv_stack') -> None:
        super(_ConvStack, self).__init__(name=name)

        self.filters = filters
        self.pool_size = pool_size
        self.use_bn = use_bn
        self.convs = [tf.keras.layers.Conv2D(f, 3, 
                                             activation='relu', 
                                             padding='same') for f in filters]
        if self.use_bn:
            self.bn = tf.keras.layers.BatchNormalization()

        if self.pool_size > 0:
            self.max_pool = tf.keras.layers.MaxPool2D(pool_size)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        for c in self.convs:
            x = c(x)
        
        if self.use_bn:
            x = self.bn(x)

        if self.pool_size > 0:
            x = self.max_pool(x)

        return x


class KurapanRecognizer(tf.keras.Model):

    def __init__(self, 
                 alphabet: Sequence[str],
                 color: bool = False,
                 filters: Sequence[int] = (64, 128, 256, 256, 512, 512, 512),
                 rnn_units: Sequence[int] = (128, 128),
                 dropout: float = .25,
                 pool_size: int = 2,
                 stn: bool = True):

        super(KurapanRecognizer, self).__init__()

        assert len(filters) == 7, '7 CNN filters must be provided.'
        assert len(rnn_units) == 2, '2 RNN filters must be provided.'

        self.alphabet = alphabet
        self.color = color
        self.filters = filters
        self.rnn_units = rnn_units
        self.dropout_rate = dropout
        self.pool_size = pool_size
        self.stn = stn

        self.convs1 = _ConvStack(filters[:3], 
                                 pool_size=pool_size, 
                                 use_bn=True,
                                 name='conv_stack_1')
        self.convs2 = _ConvStack(filters[3:5], 
                                 pool_size=pool_size, 
                                 use_bn=True,
                                 name='conv_stack_2')
        self.convs3 = _ConvStack(filters[5:], 
                                 use_bn=True, 
                                 name='conv_stack_2') 
        if self.stn:
            self.stn_layer = _STN()

        self.dense = tf.keras.layers.Dense(rnn_units[0], activation='relu')

        self.bilstm_1 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(rnn_units[0], 
                                     return_sequences=True,
                                     kernel_initializer="he_normal"),
                merge_mode='sum')

        self.bilstm_2 = tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(rnn_units[1], 
                                     return_sequences=False,
                                     kernel_initializer="he_normal"),
                merge_mode='concat')

        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        self.classifier = tf.keras.layers.Dense(len(alphabet) + 1,
                                                kernel_initializer='he_normal',
                                                activation='softmax')

    def build(self, input_shape: tf.TensorShape) -> None:
        self.height = input_shape[1]
        self.width = input_shape[2]
        super(KurapanRecognizer, self).build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = tf.transpose(x, [0, 2, 1, 3])
        x = tf.reverse(x, axis=[-1])

        x = self.convs1(x)
        x = self.convs2(x)
        x = self.convs3(x)

        if self.stn:
            x = self.stn_layer(x)

        x_shape = [None, 
                   int(self.width // self.pool_size ** 2),
                   int(self.height // self.pool_size ** 2),
                   self.filters[-1]]
        x.set_shape(x_shape)
        x = tf.reshape(x, [-1, 
                           int(self.width // self.pool_size ** 2),
                           int(self.height // self.pool_size ** 2 
                               * self.filters[-1])])

        x = self.dense(x)
        x = self.bilstm_1(x)
        x = self.bilstm_2(x)
        x = self.dropout(x)

        return self.classifier(x)


################################################################################

def _repeat(x, num_repeats):
    ones = tf.ones((1, num_repeats), dtype='int32')
    x = tf.reshape(x, shape=(-1, 1))
    x = tf.matmul(x, ones)
    return tf.reshape(x, [-1])


def _meshgrid(height, width):
    x_linspace = tf.linspace(-1., 1., width)
    y_linspace = tf.linspace(-1., 1., height)
    x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    x_coordinates = tf.reshape(x_coordinates, shape=(1, -1))
    y_coordinates = tf.reshape(y_coordinates, shape=(1, -1))
    ones = tf.ones_like(x_coordinates)
    indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
    return indices_grid


if __name__ == '__main__':
    model = KurapanRecognizer(alphabet='abc',
                              color=False)
    model.build(tf.TensorShape([None, 32, 200, 1]))
    im = tf.random.uniform([1, 32, 200, 1])
    out = model(im)
    print(out.shape)


