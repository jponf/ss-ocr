# -*- coding: utf-8 -*-

from ss_ocr.models.fpn import FpnRecognizerModel
from ss_ocr.utils import io as ioutils
from .base import Recognizer

################################################################################


class FpnRecognizer(Recognizer):

    def __init__(self):
        self.recognizer = FpnRecognizerModel()
        # 1Za4OySERY6GrQIFC3SoHfC46shDNozNH


    def recognize(self,
                  images: Sequence[InputImage]) -> List[str]:
        return []
