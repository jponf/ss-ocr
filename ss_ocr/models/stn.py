# -*- coding: utf-8 -*-

# pylint: disable=invalid-name

import collections
from typing import Optional, Sequence

import tensorflow as tf


################################################################################

class StnGridGenerator(tf.keras.layers.Layer):
    """Spatial Transformer Network - Grid Generation Layer

    Paramters
    ---------
    height : int
        Output size height.
    width : int
        Output size width.
    """

    def __init__(self, height: int, width: int,
                 *args,
                 name: str = "stn_grid_generator",
                 **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self.height = height
        self.width = width

    def compute_output_shape(self, input_shape):
        _, _, _, channels = input_shape
        return [None, self.height, self.width, channels]

    def get_config(self):
        return {
            "height": self.height,
            "width": self.width
        }

    def call(self, inputs, **kwargs):
        # inputs: theta
        # Normalized grid (-1, 1) of shape (num_batch, 2, H, W)
        n_batch = tf.shape(inputs)[0]

        # Normalized 2D grid
        x = tf.linspace(-1.0, 1.0, self.width)
        y = tf.linspace(-1.0, 1.0, self.height)
        xx, yy = tf.meshgrid(x, y)

        # Reshape to [xx, yy, 1] (homogeneous form)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))
        grid = tf.stack([xx, yy, tf.ones_like(xx)])

        # Repeat grid n_batch times
        grid = tf.expand_dims(grid, axis=0)
        grid = tf.tile(grid, [n_batch, 1, 1])

        # transforms the sampling grid
        batch_grids = tf.matmul(inputs, grid)  # (num_batch, 2, H * W)
        batch_grids = tf.reshape(batch_grids, (n_batch, 2,
                                               self.height, self.width))
        batch_grids = tf.transpose(batch_grids, (0, 2, 3, 1))

        # batch_grids = tf.transpose(batch_grids, (0, 2, 1))
        # batch_grids = tf.reshape(batch_grids, (n_batch,
        #                                        self.height,
        #                                        self.width, 2))

        return batch_grids


class StnBilinearSampler(tf.keras.layers.Layer):
    """Spatial Transformer Network - Bilinear Sampler Layer
    """

    def __init__(self,
                *args,
                 name: str = "stn_bilinear_sampler",
                 **kwargs):
        super().__init__(*args, name=name, **kwargs)
        self._height = None
        self._width = None
        self._channels = None

    def build(self, input_shape):
        image_shape, grid_shape = input_shape
        _, _, _, channels = image_shape
        _, height, width, _ = grid_shape
        self._height = height
        self._width = width
        self._channels = channels

    def compute_output_shape(self, input_shape):
        super().compute_output_shape(input_shape)  # calls build if necessary
        return tf.TensorShape([None, self._height, self._width, self._channels])

    def get_config(self):
        return { }

    def call(self, inputs, **kwargs):  # pylint: disable=too-many-locals
        images, sampling = inputs

        _, img_height, img_width, _ = images.shape
        # _, out_height, out_width, _ = sampling.shape
        x_sampling = sampling[:, :, :, 0]
        y_sampling = sampling[:, :, :, 1]

        # rescale X and Y to [0, W-1, H-1]
        x_sampling = 0.5 * (x_sampling + 1.0) * (img_width - 1.0)
        y_sampling = 0.5 * (y_sampling + 1.0) * (img_height - 1.0)

        # 4 corner points for each (x, y) [north, south, west, east]
        x_nw = tf.floor(x_sampling)  # North-West
        y_nw = tf.floor(y_sampling)
        x_ne = x_nw + 1              # North-East
        y_ne = y_nw
        x_sw = x_nw                  # South-West
        y_sw = y_nw + 1
        x_se = x_nw + 1              # South-East
        y_se = y_nw + 1

        # Interpolation
        nw = (x_se - x_sampling) * (y_se - y_sampling)
        ne = (x_sampling - x_sw) * (y_sw - y_sampling)
        sw = (x_ne - x_sampling) * (y_sampling - y_ne)
        se = (x_sampling - x_nw) * (y_sampling - y_nw)

        # Get corner points (advanced indexing)
        x_nw = tf.cast(x_nw, tf.int32)
        x_ne = tf.cast(x_ne, tf.int32)
        x_sw = tf.cast(x_sw, tf.int32)
        x_se = tf.cast(x_se, tf.int32)

        x_nw = tf.clip_by_value(x_nw, 0, img_width - 1)
        x_ne = tf.clip_by_value(x_ne, 0, img_width - 1)
        x_sw = tf.clip_by_value(x_sw, 0, img_width - 1)
        x_se = tf.clip_by_value(x_se, 0, img_width - 1)

        y_nw = tf.cast(y_nw, tf.int32)
        y_ne = tf.cast(y_ne, tf.int32)
        y_sw = tf.cast(y_sw, tf.int32)
        y_se = tf.cast(y_se, tf.int32)
        
        y_nw = tf.clip_by_value(y_nw, 0, img_height - 1)
        y_ne = tf.clip_by_value(y_ne, 0, img_height - 1)
        y_sw = tf.clip_by_value(y_sw, 0, img_height - 1)
        y_se = tf.clip_by_value(y_se, 0, img_height - 1)

        Inw = self._indexing(images, x_nw, y_nw)
        Ine = self._indexing(images, x_ne, y_ne)
        Isw = self._indexing(images, x_sw, y_sw)
        Ise = self._indexing(images, x_se, y_se)

        # Add dimension for addition
        nw = tf.expand_dims(nw, axis=3)
        ne = tf.expand_dims(ne, axis=3)
        sw = tf.expand_dims(sw, axis=3)
        se = tf.expand_dims(se, axis=3)

        out = tf.add_n([nw * Inw, ne * Ine, sw * Isw, se * Ise])

        return out

    def _indexing(self, inputs, x, y):
        shape = tf.shape(inputs)
        batch_size = shape[0]

        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, self._height, self._width))
        indices = tf.stack([b, y, x], 3)

        return tf.gather_nd(inputs, indices)


################################################################################

StnLocalizationConv = collections.namedtuple("StnLocalizationConv",
                                             ["filters", "kernel", "padding",
                                              "batch_norm", "pool"])

_DEFAULT_STN_LOCALIZATION_CONV = (
    StnLocalizationConv(filters=16, kernel=(5, 5), padding="same",
                        batch_norm=True, pool=(2, 2)),
    StnLocalizationConv(filters=32, kernel=(5, 5), padding="same",
                        batch_norm=True, pool=(2, 2)),
    StnLocalizationConv(filters=64, kernel=(3, 3), padding="same",
                        batch_norm=True, pool=(2, 2)),
)


def build_stn(input_tensor: tf.keras.Input,
              name: str = "stn",
              steps: Optional[Sequence[StnLocalizationConv]] = None,
              out_height: Optional[int] = None,
              out_width: Optional[int] = None):
    """Builds a spatial transformer network.

    Parameters
    ----------
    steps : Optional[Sequence[StnLocalizationConv]]
        Localization layer convolution steps.
    out_height: Optional[int]
        Output image height, if None it will match the input image height.
    out_width: Optional[int]
        Output image width, if None it will match the input image width.
    """
    steps = steps or _DEFAULT_STN_LOCALIZATION_CONV

    _, in_height, in_width, _ = input_tensor.shape
    out_height = out_height or in_height
    out_width = out_width or in_width

    # Initialize
    x = input_tensor

    # Convolutional layers
    for i, step in enumerate(steps, start=1):
        x = tf.keras.layers.Conv2D(filters=step.filters,
                                   kernel_size=step.kernel,
                                   name=f"{name}_conv_{i}")(x)
        if step.batch_norm:
            x = tf.keras.layers.BatchNormalization(
                name=f"{name}_conv_{i}_bn")(x)

        x = tf.keras.layers.ReLU(name=f"{name}_conv_{i}_relu")(x)

        if step.pool:
            x = tf.keras.layers.MaxPool2D(pool_size=step.pool,
                                          name=f"{name}_conv_{i}_pool")(x)

    # Dense layers
    x = tf.keras.layers.Flatten(name=f"{name}_flatten")(x)

    x = tf.keras.layers.Dense(units=64, name=f"{name}_fc_1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_fc_1_relu")(x)
    x = tf.keras.layers.Dense(
        units=6,
        kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.constant([1.0, 0.0, 0.0,
                                                         0.0, 1.0, 0.0]),
        name=f"{name}_theta")(x)

    x = tf.keras.layers.Reshape((2, 3), name=f"{name}_theta_reshape")(x)

    x = StnGridGenerator(height=out_height, width=out_width,
                         name=f"{name}_grid")(x)
    output_tensor = StnBilinearSampler(
        name=f"{name}_sampler")([input_tensor, x])

    return tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
