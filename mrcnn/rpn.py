from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose,
                                     Dense, Input, Lambda, Layer, MaxPooling2D,
                                     Reshape, TimeDistributed, UpSampling2D,
                                     ZeroPadding2D)


def RPN(anchor_stride: int, anchors_per_location: int, depth: int) -> Model:
    """Builds a Model of the Region Proposal Network.
    It wraps the RPN graph so it can be used multiple times with shared
    weights.

    Arguments:
        anchors_per_location: number of anchors per pixel in the feature map
        anchor_stride: Controls the density of anchors. Typically 1 (anchors
            for every pixel in the feature map), or 2 (every other pixel).
        depth: Depth of the backbone feature map.

    Returns:
        A `keras.Model` instance.

        The model outputs, when called, are:
            * class_logits: [batch, H * W * anchors_per_location, 2]
                Anchor classifier logits (before softmax)
            * probs: [batch, H * W * anchors_per_location, 2]
                Anchor classifier probabilities.
            * bbox: [batch, H * W * anchors_per_location, (dy, dx, log(dh),
                                                           log(dw))]
                Deltas to be applied to anchors.
    """
    feature_map = Input(shape=(None, None, depth), name='feature_map')

    # TODO: check if stride of 2 causes alignment issues if the feature map
    # is not even.
    # Shared convolutional base of the RPN
    shared = Conv2D(512, (3, 3), padding='same', activation='relu',
                    strides=anchor_stride, name='shared')(feature_map)

    # Anchor Score. [batch, height, width, anchors per location * 2].
    x = Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
               activation='linear', name='class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    class_logits = Reshape((-1, 2), name='class_logits')(x)

    # Softmax on last dimension of BG/FG.
    probs = Activation('softmax', name='probs')(class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = Conv2D(anchors_per_location * 4, (1, 1), padding='valid',
               activation='linear', name='bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    bbox = Reshape((-1, 4), name='bbox')(x)

    return Model([feature_map], [class_logits, probs, bbox], name='RPN')
