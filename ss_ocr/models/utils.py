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
