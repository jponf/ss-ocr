# -*- coding: utf-8 -*-

# pylint: disable=missing-module-docstring

from typing import Optional

import tensorflow as tf


################################################################################

class TrainingModel(tf.keras.Model):
    """With some versions of keras gradient clipping is not applied properly.

    This class overrides the training step to correct any issue with the
    current keras implementation.

    Parameters
    ----------
    clip_norm : Optional[float]
        Clips the gradients so that their norm is equal to the given value.
    """
    # pylint: disable=abstract-method

    def __init__(self,
                 clip_norm: Optional[float] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.clip_norm = clip_norm

    def train_step(self, data):
        x, y = data  # pylint: disable=invalid-name

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            loss = self.compiled_loss(y, y_pred)
            # Add any extra losses created during the forward pass.
            loss += sum(self.losses)

        # Compute gradients
        grads = tape.gradient(loss, self.trainable_variables)

        # Gradient clipping
        if self.clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.clip_norm)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

################################################################################

def build_and_reduce_fpn(backbone: tf.keras.Model,
                         inner_channels: int,
                         output_channels: int,
                         upsample_method: str,
                         reduce_method: str):

    feature_maps = build_fpn(backbone, inner_channels, upsample_method)

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


def build_fpn(backbone: tf.keras.Model,
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
