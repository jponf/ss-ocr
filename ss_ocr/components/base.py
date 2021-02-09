# -*- coding: utf-8 -*-

# pylint: disable=missing-module-docstring
# pylint: disable=too-few-public-methods

import abc
from typing import List, Sequence, Union

import numpy as np


################################################################################

InputImage = Union[np.ndarray, str]

################################################################################


class Recognizer(metaclass=abc.ABCMeta):
    """Abstract Character Recognizer."""

    def recognize(self,
                  images: Sequence[InputImage]) -> List[str]:
        """Recognizes the characters in the given image.

        Parameters
        ----------
        images : [np.ndarray or str]
            Set of images to perform recognition on.

        Returns
        -------
        text : [str]
            Sequence with the detected text in the same order as `images`.
        """
        raise NotImplementedError