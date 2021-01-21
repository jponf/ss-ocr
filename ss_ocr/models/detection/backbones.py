# -*- coding: utf-8 -*-

from typing import Sequence, Tuple

import tensorflow as tf
import tensorflow.keras.applications as ka


_MODELS = {

    # ResNets
    'resnet50': [ka.ResNet50, ka.resnet.preprocess_input],
    'resnet101': [ka.ResNet101, ka.resnet.preprocess_input],
    'resnet152': [ka.ResNet152, ka.resnet.preprocess_input],

    'resnet50v2': [ka.ResNet50V2, ka.resnet_v2.preprocess_input],
    'resnet101v2': [ka.ResNet101V2, ka.resnet_v2.preprocess_input],
    'resnet152v2': [ka.ResNet152V2, ka.resnet_v2.preprocess_input],

    # VGG
    'vgg16': [ka.VGG16, ka.vgg16.preprocess_input],
    'vgg19': [ka.VGG19, ka.vgg19.preprocess_input],

    # Densnet
    'densenet121': [ka.DenseNet121, ka.densenet.preprocess_input],
    'densenet169': [ka.DenseNet169, ka.densenet.preprocess_input],
    'densenet201': [ka.DenseNet201, ka.densenet.preprocess_input],

    # Inception
    'inceptionresnetv2': [ka.InceptionResNetV2,
                          ka.inception_resnet_v2.preprocess_input],
    'inceptionv3': [ka.InceptionV3, ka.inception_v3.preprocess_input],
    'xception': [ka.xception.Xception, ka.xception.preprocess_input],

    # Nasnet
    'nasnetlarge': [ka.NASNetLarge, ka.nasnet.preprocess_input],
    'nasnetmobile': [ka.NASNetMobile, ka.nasnet.preprocess_input],

    # MobileNet
    'mobilenet': [ka.MobileNet, ka.mobilenet.preprocess_input],
    'mobilenetv2': [ka.MobileNetV2, 
                    ka.mobilenet_v2.preprocess_input],
    
    # EfficientNets
    'efficientnetb0': [ka.EfficientNetB0, ka.efficientnet.preprocess_input],
    'efficientnetb1': [ka.EfficientNetB1, ka.efficientnet.preprocess_input],
    'efficientnetb2': [ka.EfficientNetB2, ka.efficientnet.preprocess_input],
    'efficientnetb3': [ka.EfficientNetB3, ka.efficientnet.preprocess_input],
    'efficientnetb4': [ka.EfficientNetB4, ka.efficientnet.preprocess_input],
    'efficientnetb5': [ka.EfficientNetB5, ka.efficientnet.preprocess_input],
    'efficientnetb6': [ka.EfficientNetB6, ka.efficientnet.preprocess_input],
    'efficientnetb7': [ka.EfficientNetB7, ka.efficientnet.preprocess_input],
}

_DEFAULT_FEATURE_LAYERS = {

    # VGG
    'vgg16': ('block5_conv3', 'block4_conv3', 'block3_conv3', 
              'block2_conv2', 'block1_conv2'),
    'vgg19': ('block5_conv4', 'block4_conv4', 'block3_conv4', 
              'block2_conv2', 'block1_conv2'),

    # ResNets
    'resnet50': ('conv5_block3_out', 'conv4_block6_out', 
                 'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
    'resnet101': ('conv5_block3_out', 'conv4_block23_out', 
                  'conv3_block4_out', 'conv2_block3_out', 'conv1_relu'),
    'resnet152': ('conv5_block3_out', 'conv4_block36_out', 
                  'conv3_block8_out', 'conv2_block3_out', 'conv1_relu'),

    # TODO: Resnets v2

    # DenseNet
    'densenet121': ('relu', 'pool4_conv', 
                    'pool3_conv', 'pool2_conv',
                    'conv1/relu'),
    'densenet169': ('relu', 'pool4_conv', 
                    'pool3_conv', 'pool2_conv',
                    'conv1/relu'),
    'densenet201': ('relu', 'pool4_conv', 
                    'pool3_conv', 'pool2_conv',
                    'conv1/relu'),

    # Mobile Nets
    'mobilenet': ('conv_pw_13_relu', 'conv_pw_11_relu', 'conv_pw_5_relu', 
                  'conv_pw_3_relu', 'conv_pw_1_relu'),
    'mobilenetv2': ('out_relu', 'block_13_expand_relu', 
                    'block_6_expand_relu', 'block_3_expand_relu', 
                    'block_1_expand_relu'),

    # EfficientNets
    'efficientnetb0': ('top_activation',
                       'block6a_expand_activation', 
                       'block4a_expand_activation',
                       'block3a_expand_activation', 
                       'block2a_expand_activation'),
    'efficientnetb1': ('top_activation',
                       'block6a_expand_activation', 
                       'block4a_expand_activation',
                       'block3a_expand_activation', 
                       'block2a_expand_activation'),
    'efficientnetb2': ('top_activation',
                       'block6a_expand_activation', 
                       'block4a_expand_activation',
                       'block3a_expand_activation', 
                       'block2a_expand_activation'),
    'efficientnetb3': ('top_activation',
                       'block6a_expand_activation',
                       'block4a_expand_activation',
                       'block3a_expand_activation', 
                       'block2a_expand_activation'),
    'efficientnetb4': ('top_activation',
                       'block6a_expand_activation', 
                       'block4a_expand_activation',
                       'block3a_expand_activation', 
                       'block2a_expand_activation'),
    'efficientnetb5': ('top_activation',
                       'block6a_expand_activation', 
                       'block4a_expand_activation',
                       'block3a_expand_activation', 
                       'block2a_expand_activation'),
    'efficientnetb6': ('top_activation',
                       'block6a_expand_activation', 
                       'block4a_expand_activation',
                       'block3a_expand_activation', 
                       'block2a_expand_activation'),
    'efficientnetb7': ('top_activation',
                       'block6a_expand_activation', 
                       'block4a_expand_activation',
                       'block3a_expand_activation', 
                       'block2a_expand_activation'),
}


def list_supported_models() -> Sequence[str]:
    return list(_MODELS)


def build_fpn_backbone(name: str, 
                       input_shape: Tuple[int, int, int],
                       n_levels: int = 4) -> tf.keras.Model:

    if name not in _MODELS:
        supported_models = list_supported_models()
        supported_models = '\n'.join(f'- {o}' for o in supported_models)
        raise ValueError(f"Backbone {name} is not supported. "
                         f"Supported backbones are: \n {supported_models}")

    model_cls, _ = _MODELS[name]
    model = model_cls(input_shape=input_shape, 
                      include_top=False,
                      weights='imagenet')

    outputs = [model.get_layer(o).output
               for o in _DEFAULT_FEATURE_LAYERS[name][:n_levels]]

    return tf.keras.Model(inputs=model.inputs, outputs=outputs[::-1])
