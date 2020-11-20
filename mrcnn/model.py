"""
Mask R-CNN
The main Mask R-CNN model implementation.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""

import os
import random
import datetime
import re
import math
import logging
import multiprocessing
import numpy as np
import tensorflow as tf

from collections import OrderedDict
from mrcnn import utils
from mrcnn.rpn import RPN
from mrcnn.layers import DetectionTarget, Proposals, PyramidROIAlign, TrimZeros
from tensorflow.keras import Model
from tensorflow.keras.layers import (Activation, Add, BatchNormalization,
                                     Concatenate, Conv2D, Conv2DTranspose,
                                     Dense, Input, Lambda, Layer, MaxPooling2D,
                                     Reshape, TimeDistributed, UpSampling2D)
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion('2.1')


tf.compat.v1.disable_eager_execution()
# tf.compat.v1.experimental.output_all_intermediates(True)


############################################################
#  Utility Functions
############################################################

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ('shape: {:20}  '.format(str(array.shape)))
        if array.size:
            text += ('min: {:10.5f}  max: {:10.5f}'.format(array.min(), array.max()))
        else:
            text += ('min: {:10}  max: {:10}'.format('', ''))
        text += '  {}'.format(array.dtype)
    print(text)


############################################################
#  Proposal Layer
############################################################

def apply_box_deltas_graph(boxes, deltas):
    """Applies the given deltas to the given boxes.
    boxes: [N, (y1, x1, y2, x2)] boxes to update
    deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
    """
    # Convert to y, x, h, w
    height = boxes[:, 2] - boxes[:, 0]
    width = boxes[:, 3] - boxes[:, 1]
    center_y = boxes[:, 0] + 0.5 * height
    center_x = boxes[:, 1] + 0.5 * width
    # Apply deltas
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])
    # Convert back to y1, x1, y2, x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width
    result = tf.stack([y1, x1, y2, x2], axis=1, name='apply_box_deltas_out')
    return result


def clip_boxes_graph(boxes, window):
    """
    boxes: [N, (y1, x1, y2, x2)]
    window: [4] in the form y1, x1, y2, x2
    """
    # Split
    wy1, wx1, wy2, wx2 = tf.split(window, 4)
    y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
    # Clip
    y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
    x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
    y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
    x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
    clipped = tf.concat([y1, x1, y2, x2], axis=1, name="clipped_boxes")
    clipped.set_shape((clipped.shape[0], 4))
    return clipped


############################################################
#  Detection Layer
############################################################

class DetectionLayer(Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self, num_detections: int, min_confidence: float,
                 nms_threshold: float, bbox_std_dev: np.ndarray, **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.num_detections = num_detections
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.bbox_std_dev = bbox_std_dev

    def get_config(self):
        config = super(DetectionLayer, self).get_config()
        config['num_detections'] = self.num_detections
        config['min_confidence'] = self.min_confidence
        config['nms_threshold'] = self.nms_threshold
        config['bbox_std_dev'] = self.bbox_std_dev
        return config

    def call(self, inputs):
        input_shape = (inputs[0].shape, inputs[1].shape, inputs[2].shape,
                       inputs[3].shape)

        rois = inputs[0]
        mrcnn_class_probs = inputs[1]
        mrcnn_bbox_deltas = inputs[2]
        image_meta = inputs[3]

        # Get windows of images in normalized coordinates. Windows are the area
        # in the image that excludes the padding.
        # Use the shape of the first image in the batch to normalize the window
        # because we know that all images get resized to the same size.
        m = parse_image_meta_graph(image_meta)
        image_shape = m['image_shape'][0]
        window = norm_boxes_graph(m['window'], image_shape[:2])
        active_class_ids = m['active_class_ids']

        # Run detection refinement graph on each item in the batch
        detections = tf.map_fn(self._refine_detections,
                               (rois, mrcnn_class_probs, mrcnn_bbox_deltas,
                                window, active_class_ids),
                               fn_output_signature=tf.float32)

        # Reshape output
        # [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)]
        # in normalized coordinates
        return tf.reshape(detections,
                          self.compute_output_shape(input_shape))

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape[0]).as_list()
        return tf.TensorShape((input_shape[0], self.num_detections, 6))

    def _refine_detections(self, rois, probs, deltas, window,
                           active_class_ids):
        """Refine classified proposals and filter overlaps and return final
        detections.

        Inputs:
            rois: [N, (y1, x1, y2, x2)] in normalized coordinates
            probs: [N, num_classes]. Class probabilities.
            deltas: [N, num_classes, (dy, dx, log(dh), log(dw))].
                Class-specific bounding box deltas.
            window: (y1, x1, y2, x2) in normalized coordinates. The part of
                the image that contains the image excluding the padding.
            active_class_ids: [num_classes]. Has a value of 1 for classes
                that are allowed in the dataset of the image, and 0 for classes
                that are not allowed in the dataset.

        Returns detections shaped:
            [num_detections, (y1, x1, y2, x2, class_id, score)] where
            coordinates are normalized.
        """
        # Suppress scores for inactive classes
        probs = tf.where(tf.cast(tf.tile(tf.expand_dims(active_class_ids, 0),
                                 (probs.shape[0], 1)), tf.bool),
                         x=probs, y=tf.zeros_like(probs))
        # Class IDs per ROI
        class_ids = tf.argmax(input=probs, axis=1, output_type=tf.int32)
        # Class probability of the top class of each ROI
        indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(probs, indices)
        # Class-specific bounding box deltas
        deltas_specific = tf.gather_nd(deltas, indices)
        # Apply bounding box deltas
        # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
        refined_rois = apply_box_deltas_graph(
            rois, deltas_specific * self.bbox_std_dev)
        # Clip boxes to image window
        refined_rois = clip_boxes_graph(refined_rois, window)

        # TODO: Filter out boxes with zero area

        # Filter out background boxes
        keep = tf.where(class_ids > 0)[:, 0]
        # Filter out low confidence boxes
        if self.min_confidence:
            conf_keep = tf.where(
                class_scores >= self.min_confidence)[:, 0]
            keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
            keep = tf.sparse.to_dense(keep)[0]

        # Apply per-class NMS
        # 1. Prepare variables
        pre_nms_class_ids = tf.gather(class_ids, keep)
        pre_nms_scores = tf.gather(class_scores, keep)
        pre_nms_rois = tf.gather(refined_rois,   keep)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        def nms_keep_map(class_id):
            """Apply Non-Maximum Suppression on ROIs of the given class."""
            # Indices of ROIs of the given class
            ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # Apply NMS
            class_keep = tf.image.non_max_suppression(
                    tf.gather(pre_nms_rois, ixs),
                    tf.gather(pre_nms_scores, ixs),
                    max_output_size=self.num_detections,
                    iou_threshold=self.nms_threshold)
            # Map indices
            class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
            # Pad with -1 so returned tensors have the same shape
            gap = self.num_detections - tf.shape(class_keep)[0]
            class_keep = tf.pad(class_keep, [(0, gap)],
                                mode='CONSTANT', constant_values=-1)
            # Set shape so map_fn() can infer result shape
            class_keep.set_shape([self.num_detections])
            return class_keep

        # 2. Map over class IDs
        nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,
                             dtype=tf.int64)
        # 3. Merge results into one list, and remove -1 padding
        nms_keep = tf.reshape(nms_keep, [-1])
        nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
        # 4. Compute intersection between keep and nms_keep
        keep = tf.sets.intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]
        # Keep top detections
        roi_count = self.num_detections
        class_scores_keep = tf.gather(class_scores, keep)
        num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
        top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
        keep = tf.gather(keep, top_ids)

        # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
        # Coordinates are normalized.
        detections = tf.concat([
            tf.gather(refined_rois, keep),
            tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],
            tf.gather(class_scores, keep)[..., tf.newaxis]
            ], axis=1)

        # Pad with zeros if detections < DETECTION_MAX_INSTANCES
        gap = self.num_detections - tf.shape(detections)[0]
        detections = tf.pad(detections, [(0, gap), (0, 0)], 'CONSTANT')
        return detections


############################################################
#  Feature Pyramid Network Heads
############################################################

def fpn_classifier_graph(rois, feature_maps, image_meta, pool_size, num_classes,
                         IMAGE_META, train_bn=True, fc_layers_size=1024):
    """Builds the computation graph of the feature pyramid network classifier
    and regressor heads.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates. Zero padded.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers
    fc_layers_size: Size of the 2 FC layers

    Returns:
        logits: [batch, num_rois, NUM_CLASSES] classifier logits (before softmax).
            Zero padded.
        probs: [batch, num_rois, NUM_CLASSES] classifier probabilities. Zero padded.
        bbox_deltas: [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
            Deltas to apply to proposal boxes. Zero padded.
    """
    # first find which rois are padded
    # non_zeros: [batch, num_rois]
    _, non_zeros = tf.map_fn(trim_zeros_graph, rois,
                             fn_output_signature=(tf.float32, tf.bool))

    # ROI Pooling
    # Shape: [batch, num_rois, POOL_SIZE, POOL_SIZE, channels]
    x = PyramidROIAlign((pool_size, pool_size), IMAGE_META['image_shape'],
                        name='roi_align_classifier')([rois, image_meta] +
                                                     feature_maps)
    # Two 1024 FC layers (implemented with Conv2D for consistency)
    x = TimeDistributed(Conv2D(fc_layers_size, (pool_size, pool_size),
                               padding='valid'),
                        name='mrcnn_class_conv1')(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn1')(
        x, training=train_bn)
    x = Activation('relu')(x)
    x = TimeDistributed(Conv2D(fc_layers_size, (1, 1)),
                        name='mrcnn_class_conv2')(x)
    x = TimeDistributed(BatchNormalization(), name='mrcnn_class_bn2')(
        x, training=train_bn)
    x = Activation('relu')(x)

    shared = Lambda(lambda x: tf.squeeze(tf.squeeze(x, 3), 2),
                    name='pool_squeeze')(x)

    # Classifier head
    mrcnn_class_logits = TimeDistributed(Dense(num_classes),
                                         name='mrcnn_class_logits')(shared)
    mrcnn_probs = TimeDistributed(Activation('softmax'),
                                  name='mrcnn_class')(mrcnn_class_logits)

    # BBox head
    # [batch, num_rois, NUM_CLASSES * (dy, dx, log(dh), log(dw))]
    x = TimeDistributed(Dense(num_classes * 4, activation='linear'),
                        name='mrcnn_bbox_fc')(shared)
    # Reshape to [batch, num_rois, NUM_CLASSES, (dy, dx, log(dh), log(dw))]
    s = tf.keras.backend.int_shape(x)
    mrcnn_bbox = Reshape((-1 if s[1] is None else s[1], num_classes, 4),
                         name='mrcnn_bbox')(x)

    # finally zero out the results which are calculated from zero padded rois
    # non_zeros: [batch, num_rois, num_classes]
    non_zeros = tf.tile(tf.expand_dims(non_zeros, -1), (1, 1, num_classes))

    mrcnn_class_logits = TrimZeros((1, 1, num_classes),
                                   name='mrcnn_class_logits_trimed')([
                                        rois, mrcnn_class_logits])
    mrcnn_probs = TrimZeros((1, 1, num_classes),
                            name='mrcnn_class_trimed')([rois, mrcnn_probs])

    # non_zeros: [batch, num_rois, num_classes, 4]
    mrcnn_bbox = TrimZeros((1, 1, num_classes), (1, 1, 1, 4),
                           name='mrcnn_bbox_trimed')([rois, mrcnn_bbox])
    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox


def build_fpn_mask_graph(rois, feature_maps, image_meta, pool_size, num_classes,
                         IMAGE_META, train_bn=True):
    """Builds the computation graph of the mask head of Feature Pyramid Network.

    rois: [batch, num_rois, (y1, x1, y2, x2)] Proposal boxes in normalized
          coordinates.
    feature_maps: List of feature maps from different layers of the pyramid,
                  [P2, P3, P4, P5]. Each has a different resolution.
    image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    pool_size: The width of the square feature map generated from ROI Pooling.
    num_classes: number of classes, which determines the depth of the results
    train_bn: Boolean. Train or freeze Batch Norm layers

    Returns: Masks [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, NUM_CLASSES]
    """
    # ROI Pooling
    # Shape: [batch, num_rois, MASK_POOL_SIZE, MASK_POOL_SIZE, channels]
    x = PyramidROIAlign([pool_size, pool_size], IMAGE_META['image_shape'],
                        name='roi_align_mask')([rois, image_meta] + feature_maps)

    # Conv layers
    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                        name='mrcnn_mask_conv1')(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn1')(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                        name='mrcnn_mask_conv2')(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn2')(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                        name='mrcnn_mask_conv3')(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn3')(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2D(256, (3, 3), padding='same'),
                        name='mrcnn_mask_conv4')(x)
    x = TimeDistributed(BatchNormalization(),
                        name='mrcnn_mask_bn4')(x, training=train_bn)
    x = Activation('relu')(x)

    x = TimeDistributed(Conv2DTranspose(256, (2, 2), strides=2,
                                        activation='relu'),
                        name='mrcnn_mask_deconv')(x)
    x = TimeDistributed(Conv2D(num_classes, (1, 1), strides=1,
                               activation='sigmoid'), name='mrcnn_mask')(x)
    return x


############################################################
#  Loss Functions
############################################################

def smooth_l1_loss(y_true, y_pred):
    """Implements Smooth-L1 loss.
    y_true and y_pred are typically: [N, 4], but could be any shape.
    """
    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1.0), 'float32')
    return (less_than_one * 0.5 * diff**2) + (1 - less_than_one) * (diff - 0.5)


def rpn_class_loss_graph(rpn_match, rpn_class_logits):
    """RPN anchor classifier loss.

    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_class_logits: [batch, anchors, 2]. RPN classifier logits for BG/FG.
    """
    # Squeeze last dim to simplify
    rpn_match = tf.squeeze(rpn_match, -1)
    # Get anchor classes. Convert the -1/+1 match to 0/1 values.
    anchor_class = tf.cast(tf.equal(rpn_match, 1), tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(tf.not_equal(rpn_match, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)
    anchor_class = tf.gather_nd(anchor_class, indices)
    # Cross entropy loss
    loss = tf.keras.backend.sparse_categorical_crossentropy(
        target=anchor_class, output=rpn_class_logits, from_logits=True)
    return tf.keras.backend.switch(tf.size(loss) > 0,
                                   tf.keras.backend.mean(loss),
                                   tf.constant(0.0))


def rpn_bbox_loss_graph(target_bbox, rpn_match, rpn_bbox):
    """Return the RPN bounding box loss graph.

    target_bbox: [batch, max positive anchors, (dy, dx, log(dh), log(dw))].
        Uses 0 padding to fill in unsed bbox deltas.
    rpn_match: [batch, anchors, 1]. Anchor match type. 1=positive,
               -1=negative, 0=neutral anchor.
    rpn_bbox: [batch, anchors, (dy, dx, log(dh), log(dw))]
    """
    # Positive anchors contribute to the loss, but negative and
    # neutral anchors (match value of 0 or -1) don't.
    rpn_match = tf.squeeze(rpn_match, -1)
    indices = tf.where(tf.equal(rpn_match, 1))

    # Pick bbox deltas that contribute to the loss
    rpn_bbox = tf.gather_nd(rpn_bbox, indices)

    # Trim target bounding box deltas to the same length as rpn_bbox.
    batch_counts = tf.keras.backend.sum(tf.cast(tf.equal(rpn_match, 1),
                                                tf.int32), axis=1)
    # target_bbox = batch_pack_graph(target_bbox, batch_counts, images_per_gpu)
    target_bbox = tf.map_fn(lambda x: x[0][x[1]], (target_bbox, batch_counts),
                            fn_output_signature=np.float32)

    loss = smooth_l1_loss(target_bbox, rpn_bbox)

    return tf.keras.backend.switch(tf.size(loss) > 0,
                                   tf.keras.backend.mean(loss),
                                   tf.constant(0.0))


def mrcnn_class_loss_graph(target_bbox, target_class_ids, pred_class_logits,
                           active_class_ids):
    """Loss for the classifier head of Mask RCNN.

    target_bbox: [batch, num_rois, (y1, x1, y2, x2)]
    target_class_ids: [batch, num_rois]. Integer class IDs. Uses zero
        padding to fill in the array.
    pred_class_logits: [batch, num_rois, num_classes]. Uses zero
        padding to fill in the array.
    active_class_ids: [batch, num_classes]. Has a value of 1 for
        classes that are in the dataset of the image, and 0
        for classes that are not in the dataset.
    """

    # Specify which rois are zero padding: look for zeros in bboxes
    # (zeros in target_class_ids could still just be background class)
    non_zeros = tf.cast(tf.reduce_max(target_bbox, axis=2) > 0, 'float32')

    # During model building, Keras calls this function with
    # target_class_ids of type float32. Unclear why. Cast it
    # to int to get around it.
    target_class_ids = tf.cast(target_class_ids, 'int64')

    # Suppress predictions of classes that are not in the dataset
    active_class_ids = tf.keras.backend.tile(
        tf.expand_dims(active_class_ids, 1),
        (1, pred_class_logits.shape[1], 1))
    pred_class_logits = pred_class_logits * active_class_ids

    # Loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=pred_class_logits)

    # Erase losses of predictions of classes that are not in the active
    # classes of the image or is padding.
    loss = loss * non_zeros

    # Computer loss mean. Use only predictions that contribute
    # to the loss to get a correct mean.
    denominator = tf.reduce_sum(non_zeros)
    loss = tf.cond(tf.not_equal(denominator, 0), lambda: tf.reduce_sum(loss) /
                   denominator, lambda: tf.constant(0, 'float32'))
    return loss


def mrcnn_bbox_loss_graph(target_bbox, target_class_ids, pred_bbox):
    """Loss for Mask R-CNN bounding box refinement.

    target_bbox: [batch, num_rois, (dy, dx, log(dh), log(dw))]
    target_class_ids: [batch, num_rois]. Integer class IDs.
    pred_bbox: [batch, num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    """
    # Reshape to merge batch and roi dimensions for simplicity.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    target_bbox = tf.reshape(target_bbox, (-1, 4))
    pred_bbox = tf.reshape(pred_bbox,
                           (-1, tf.keras.backend.int_shape(pred_bbox)[2], 4))

    # Zero-padded ROIs do not contribute to the loss. Look for zeros in bboxes
    # (as zeros in target_class_ids could still just be background class).
    positive_roi_ix = tf.where(tf.reduce_max(target_bbox, axis=1, keepdims=True) > 0)[:, 0]
    positive_roi_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_roi_ix), tf.int64)
    indices = tf.stack([positive_roi_ix, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    target_bbox = tf.gather(target_bbox, positive_roi_ix)
    pred_bbox = tf.gather_nd(pred_bbox, indices)

    # Smooth-L1 Loss
    loss = tf.keras.backend.switch(tf.size(target_bbox) > 0,
                                   smooth_l1_loss(y_true=target_bbox,
                                                  y_pred=pred_bbox),
                                   tf.constant(0.0))
    loss = tf.keras.backend.mean(loss)
    return loss


def mrcnn_mask_loss_graph(target_masks, target_class_ids, pred_masks):
    """Mask binary cross-entropy loss for the masks head.

    target_masks: [batch, num_rois, height, width].
        A float32 tensor of values 0 or 1. Uses zero padding to fill array.
    target_class_ids: [batch, num_rois]. Integer class IDs. Zero padded.
    pred_masks: [batch, proposals, height, width, num_classes] float32 tensor
                with values from 0 to 1.
    """
    # Reshape for simplicity. Merge first two dimensions into one.
    target_class_ids = tf.reshape(target_class_ids, (-1,))
    mask_shape = tf.shape(target_masks)
    target_masks = tf.reshape(target_masks, (-1, mask_shape[2], mask_shape[3]))
    pred_shape = tf.shape(pred_masks)
    pred_masks = tf.reshape(pred_masks, (-1, pred_shape[2], pred_shape[3],
                                         pred_shape[4]))
    # Permute predicted masks to [N, num_classes, height, width]
    pred_masks = tf.transpose(pred_masks, [0, 3, 1, 2])

    # Only positive ROIs contribute to the loss. And only
    # the class specific mask of each ROI.
    positive_ix = tf.where(target_class_ids > 0)[:, 0]
    positive_class_ids = tf.cast(
        tf.gather(target_class_ids, positive_ix), tf.int64)
    indices = tf.stack([positive_ix, positive_class_ids], axis=1)

    # Gather the masks (predicted and true) that contribute to loss
    y_true = tf.gather(target_masks, positive_ix)
    y_pred = tf.gather_nd(pred_masks, indices)

    # Compute binary cross entropy. If no positive ROIs, then return 0.
    # shape: [batch, roi, num_classes]
    loss = tf.keras.backend.switch(tf.size(y_true) > 0,
                                   tf.keras.backend.binary_crossentropy(
                                        target=y_true, output=y_pred),
                                   tf.constant(0.0))
    loss = tf.keras.backend.mean(loss)
    return loss


############################################################
#  Data Generator
############################################################

def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = utils.resize_mask(mask, scale, padding, crop)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug
        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Convert instance mask to segmentation map
        if mask.size == 0:
            idxmask = np.zeros(image_shape[:2])
        else:
            idxmask = np.argmax(mask, axis=2) + 1
            idxmask[~np.any(mask, axis=2)] = 0 # add background
        segmap = imgaug.augmentables.segmaps.SegmentationMapsOnImage(
            idxmask.astype(np.int32), shape=image_shape)
        segmap_shape = segmap.shape
        # # Make augmenters deterministic to apply similarly to images and masks
        # det = augmentation.to_deterministic()
        # image = det.augment_image(image)
        # segmap = det.augment_segmentation_maps(segmap)
        augmented = augmentation.augment_batch_(
            imgaug.augmentables.batches.UnnormalizedBatch(images=[image],
                                                          segmentation_maps=[segmap]))
        image = augmented.images_aug[0]
        segmap = augmented.segmentation_maps_aug[0]
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert segmap.shape == segmap_shape, "Augmentation shouldn't change segmap size"
        if mask.size > 0:
            # Change mask back to bool
            mask = np.eye(mask_shape[2])[segmap.get_arr() - 1]
            mask *= np.expand_dims(segmap.get_arr() > 0, axis=2) # del background
            mask = mask.astype(np.bool)
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([config.NUM_CLASSES], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def build_detection_targets(rpn_rois, gt_class_ids, gt_boxes, gt_masks, config):
    """Generate targets for training Stage 2 classifier and mask heads.
    This is not used in normal training. It's useful for debugging or to train
    the Mask RCNN heads without using the RPN head.

    Inputs:
    rpn_rois: [N, (y1, x1, y2, x2)] proposal boxes.
    gt_class_ids: [instance count] Integer class IDs
    gt_boxes: [instance count, (y1, x1, y2, x2)]
    gt_masks: [height, width, instance count] Ground truth masks. Can be full
              size or mini-masks.

    Returns:
    rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)]
    class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    bboxes: [TRAIN_ROIS_PER_IMAGE, NUM_CLASSES, (y, x, log(h), log(w))]. Class-specific
            bbox refinements.
    masks: [TRAIN_ROIS_PER_IMAGE, height, width, NUM_CLASSES). Class specific masks cropped
           to bbox boundaries and resized to neural network output size.
    """
    assert rpn_rois.shape[0] > 0
    assert gt_class_ids.dtype == np.int32, "Expected int but got {}".format(
        gt_class_ids.dtype)
    assert gt_boxes.dtype == np.int32, "Expected int but got {}".format(
        gt_boxes.dtype)
    assert gt_masks.dtype == np.bool_, "Expected bool but got {}".format(
        gt_masks.dtype)

    # It's common to add GT Boxes to ROIs but we don't do that here because
    # according to XinLei Chen's paper, it doesn't help.

    # Trim empty padding in gt_boxes and gt_masks parts
    instance_ids = np.where(gt_class_ids > 0)[0]
    assert instance_ids.shape[0] > 0, "Image must contain instances."
    gt_class_ids = gt_class_ids[instance_ids]
    gt_boxes = gt_boxes[instance_ids]
    gt_masks = gt_masks[:, :, instance_ids]

    # Compute areas of ROIs and ground truth boxes.
    rpn_roi_area = (rpn_rois[:, 2] - rpn_rois[:, 0]) * \
        (rpn_rois[:, 3] - rpn_rois[:, 1])
    gt_box_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * \
        (gt_boxes[:, 3] - gt_boxes[:, 1])

    # Compute overlaps [rpn_rois, gt_boxes]
    overlaps = np.zeros((rpn_rois.shape[0], gt_boxes.shape[0]))
    for i in range(overlaps.shape[1]):
        gt = gt_boxes[i]
        overlaps[:, i] = utils.compute_iou(
            gt, rpn_rois, gt_box_area[i], rpn_roi_area)

    # Assign ROIs to GT boxes
    rpn_roi_iou_argmax = np.argmax(overlaps, axis=1)
    rpn_roi_iou_max = overlaps[np.arange(
        overlaps.shape[0]), rpn_roi_iou_argmax]
    # GT box assigned to each ROI
    rpn_roi_gt_boxes = gt_boxes[rpn_roi_iou_argmax]
    rpn_roi_gt_class_ids = gt_class_ids[rpn_roi_iou_argmax]

    # Positive ROIs are those with >= 0.5 IoU with a GT box.
    fg_ids = np.where(rpn_roi_iou_max > 0.5)[0]

    # Negative ROIs are those with max IoU 0.1-0.5 (hard example mining)
    # TODO: To hard example mine or not to hard example mine, that's the question
    # bg_ids = np.where((rpn_roi_iou_max >= 0.1) & (rpn_roi_iou_max < 0.5))[0]
    bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]

    # Subsample ROIs. Aim for 33% foreground.
    # FG
    fg_roi_count = int(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO)
    if fg_ids.shape[0] > fg_roi_count:
        keep_fg_ids = np.random.choice(fg_ids, fg_roi_count, replace=False)
    else:
        keep_fg_ids = fg_ids
    # BG
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep_fg_ids.shape[0]
    if bg_ids.shape[0] > remaining:
        keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
    else:
        keep_bg_ids = bg_ids
    # Combine indices of ROIs to keep
    keep = np.concatenate([keep_fg_ids, keep_bg_ids])
    # Need more?
    remaining = config.TRAIN_ROIS_PER_IMAGE - keep.shape[0]
    if remaining > 0:
        # Looks like we don't have enough samples to maintain the desired
        # balance. Reduce requirements and fill in the rest. This is
        # likely different from the Mask RCNN paper.

        # There is a small chance we have neither fg nor bg samples.
        if keep.shape[0] == 0:
            # Pick bg regions with easier IoU threshold
            bg_ids = np.where(rpn_roi_iou_max < 0.5)[0]
            assert bg_ids.shape[0] >= remaining
            keep_bg_ids = np.random.choice(bg_ids, remaining, replace=False)
            assert keep_bg_ids.shape[0] == remaining
            keep = np.concatenate([keep, keep_bg_ids])
        else:
            # Fill the rest with repeated bg rois.
            keep_extra_ids = np.random.choice(
                keep_bg_ids, remaining, replace=True)
            keep = np.concatenate([keep, keep_extra_ids])
    assert keep.shape[0] == config.TRAIN_ROIS_PER_IMAGE, \
        "keep doesn't match ROI batch size {}, {}".format(
            keep.shape[0], config.TRAIN_ROIS_PER_IMAGE)

    # Reset the gt boxes assigned to BG ROIs.
    rpn_roi_gt_boxes[keep_bg_ids, :] = 0
    rpn_roi_gt_class_ids[keep_bg_ids] = 0

    # For each kept ROI, assign a class_id, and for FG ROIs also add bbox refinement.
    rois = rpn_rois[keep]
    roi_gt_boxes = rpn_roi_gt_boxes[keep]
    roi_gt_class_ids = rpn_roi_gt_class_ids[keep]
    roi_gt_assignment = rpn_roi_iou_argmax[keep]

    # Class-aware bbox deltas. [y, x, log(h), log(w)]
    bboxes = np.zeros((config.TRAIN_ROIS_PER_IMAGE,
                       config.NUM_CLASSES, 4), dtype=np.float32)
    pos_ids = np.where(roi_gt_class_ids > 0)[0]
    bboxes[pos_ids, roi_gt_class_ids[pos_ids]] = utils.box_refinement(
        rois[pos_ids], roi_gt_boxes[pos_ids, :4])
    # Normalize bbox refinements
    bboxes /= config.BBOX_STD_DEV

    # Generate class-specific target masks
    masks = np.zeros((config.TRAIN_ROIS_PER_IMAGE, config.MASK_SHAPE[0], config.MASK_SHAPE[1], config.NUM_CLASSES),
                     dtype=np.float32)
    for i in pos_ids:
        class_id = roi_gt_class_ids[i]
        assert class_id > 0, "class id must be greater than 0"
        gt_id = roi_gt_assignment[i]
        class_mask = gt_masks[:, :, gt_id]

        if config.USE_MINI_MASK:
            # Create a mask placeholder, the size of the image
            placeholder = np.zeros(config.IMAGE_SHAPE[:2], dtype=bool)
            # GT box
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[gt_id]
            gt_w = gt_x2 - gt_x1
            gt_h = gt_y2 - gt_y1
            # Resize mini mask to size of GT box
            placeholder[gt_y1:gt_y2, gt_x1:gt_x2] = \
                np.round(utils.resize(class_mask, (gt_h, gt_w))).astype(bool)
            # Place the mini batch in the placeholder
            class_mask = placeholder

        # Pick part of the mask and resize it
        y1, x1, y2, x2 = rois[i].astype(np.int32)
        m = class_mask[y1:y2, x1:x2]
        mask = utils.resize(m, config.MASK_SHAPE)
        masks[i, :, :, class_id] = mask

    return rois, roi_gt_class_ids, bboxes, masks


def build_rpn_targets(image_shape, anchors, gt_class_ids, gt_boxes, config):
    """Given the anchors and GT boxes, compute overlaps and identify positive
    anchors and deltas to refine them to match their corresponding GT boxes.

    anchors: [num_anchors, (y1, x1, y2, x2)]
    gt_class_ids: [num_gt_boxes] Integer class IDs.
    gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

    Returns:
    rpn_match: [num_anchors] (int32) matches between anchors and GT boxes.
               1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_bbox: [num_anchors, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    """
    # Generate negative anchors for sample that doesn't have instances
    if gt_class_ids.shape[0]==0:
        rpn_match = -1 * np.ones([anchors.shape[0]], dtype=np.int32)
        rpn_bbox = generate_random_rois(image_shape, \
            config.RPN_TRAIN_ANCHORS_PER_IMAGE, gt_class_ids, gt_boxes)
        return rpn_match, rpn_bbox

    # RPN Match: 1 = positive anchor, -1 = negative anchor, 0 = neutral
    rpn_match = np.zeros([anchors.shape[0]], dtype=np.int32)
    # RPN bounding boxes: [max anchors per image, (dy, dx, log(dh), log(dw))]
    rpn_bbox = np.zeros((config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4))

    # Handle COCO crowds
    # A crowd box in COCO is a bounding box around several instances. Exclude
    # them from training. A crowd box is given a negative class ID.
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        # Filter out crowds from ground truth class IDs and boxes
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = gt_boxes[crowd_ix]
        gt_class_ids = gt_class_ids[non_crowd_ix]
        gt_boxes = gt_boxes[non_crowd_ix]
        # Compute overlaps with crowd boxes [anchors, crowds]
        crowd_overlaps = utils.compute_overlaps(anchors, crowd_boxes)
        crowd_iou_max = np.amax(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        # All anchors don't intersect a crowd
        no_crowd_bool = np.ones([anchors.shape[0]], dtype=bool)

    # Compute overlaps [num_anchors, num_gt_boxes]
    overlaps = utils.compute_overlaps(anchors, gt_boxes)

    # Match anchors to GT Boxes
    # If an anchor overlaps a GT box with IoU >= 0.7 then it's positive.
    # If an anchor overlaps a GT box with IoU < 0.3 then it's negative.
    # Neutral anchors are those that don't match the conditions above,
    # and they don't influence the loss function.
    # However, don't keep any GT box unmatched (rare, but happens). Instead,
    # match it to the closest anchor (even if its max IoU is < 0.3).
    #
    # 1. Set negative anchors first. They get overwritten below if a GT box is
    # matched to them. Skip boxes in crowd areas.
    anchor_iou_argmax = np.argmax(overlaps, axis=1)
    anchor_iou_max = overlaps[np.arange(overlaps.shape[0]), anchor_iou_argmax]
    rpn_match[(anchor_iou_max < 0.3) & (no_crowd_bool)] = -1
    # 2. Set an anchor for each GT box (regardless of IoU value).
    # If multiple anchors have the same IoU match all of them
    gt_iou_argmax = np.argwhere(overlaps == np.max(overlaps, axis=0))[:,0]
    rpn_match[gt_iou_argmax] = 1
    # 3. Set anchors with high overlap as positive.
    rpn_match[anchor_iou_max >= 0.7] = 1

    # Subsample to balance positive and negative anchors
    # Don't let positives be more than half the anchors
    ids = np.where(rpn_match == 1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE // 2)
    if extra > 0:
        # Reset the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0
    # Same for negative proposals
    ids = np.where(rpn_match == -1)[0]
    extra = len(ids) - (config.RPN_TRAIN_ANCHORS_PER_IMAGE -
                        np.sum(rpn_match == 1))
    if extra > 0:
        # Rest the extra ones to neutral
        ids = np.random.choice(ids, extra, replace=False)
        rpn_match[ids] = 0

    # For positive anchors, compute shift and scale needed to transform them
    # to match the corresponding GT boxes.
    ids = np.where(rpn_match == 1)[0]
    ix = 0  # index into rpn_bbox
    # TODO: use box_refinement() rather than duplicating the code here
    for i, a in zip(ids, anchors[ids]):
        # Closest gt box (it might have IoU < 0.7)
        gt = gt_boxes[anchor_iou_argmax[i]]

        # Convert coordinates to center plus width/height.
        # GT Box
        gt_h = gt[2] - gt[0]
        gt_w = gt[3] - gt[1]
        gt_center_y = gt[0] + 0.5 * gt_h
        gt_center_x = gt[1] + 0.5 * gt_w
        # Anchor
        a_h = a[2] - a[0]
        a_w = a[3] - a[1]
        a_center_y = a[0] + 0.5 * a_h
        a_center_x = a[1] + 0.5 * a_w

        # Compute the bbox refinement that the RPN should predict.
        rpn_bbox[ix] = [
            (gt_center_y - a_center_y) / a_h,
            (gt_center_x - a_center_x) / a_w,
            np.log(gt_h / a_h),
            np.log(gt_w / a_w),
        ]
        # Normalize
        rpn_bbox[ix] /= config.RPN_BBOX_STD_DEV
        ix += 1

    return rpn_match, rpn_bbox


def generate_random_rois(image_shape, count, gt_class_ids, gt_boxes):
    """Generates ROI proposals similar to what a region proposal network
    would generate.

    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.

    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    if gt_boxes.shape[0]==0:
        # If there are no instances in the image,
        # we don't generate GT-box-specific ROIs
        rois_per_box = 0
    else:
        # Generate random ROIs around GT boxes (90% of count)
        rois_per_box = int(0.9 * count / gt_boxes.shape[0])
        for i in range(gt_boxes.shape[0]):
            gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
            h = gt_y2 - gt_y1
            w = gt_x2 - gt_x1
            # random boundaries
            r_y1 = max(gt_y1 - h, 0)
            r_y2 = min(gt_y2 + h, image_shape[0])
            r_x1 = max(gt_x1 - w, 0)
            r_x2 = min(gt_x2 + w, image_shape[1])

            # To avoid generating boxes with zero area, we generate double what
            # we need and filter out the extra. If we get fewer valid boxes
            # than we need, we loop and try again.
            while True:
                y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
                x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
                # Filter out zero area boxes
                threshold = 1
                y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                            threshold][:rois_per_box]
                x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                            threshold][:rois_per_box]
                if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                    break

            # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
            # into x1, y1, x2, y2 order
            x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
            y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
            box_rois = np.hstack([y1, x1, y2, x2])
            rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois


class DataGenerator(tf.keras.utils.Sequence):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTarget.
    no_augmentation_sources: Optional. List of sources to exclude for
        augmentation. A source is string that identifies a dataset and is
        defined in the Dataset class.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """

    def __init__(self, dataset, config, shuffle=True, augment=False, augmentation=None,
                 random_rois=0, batch_size=1, detection_targets=False,
                 no_augmentation_sources=None):

        self.image_ids = np.copy(dataset.image_ids)
        self.dataset = dataset
        self.config = config
        self.error_count = 0

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        # self.backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
        # self.anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
        #                                               config.RPN_ANCHOR_RATIOS,
        #                                               self.backbone_shapes,
        #                                               config.BACKBONE_STRIDES,
        #                                               config.RPN_ANCHOR_STRIDE)

        self.shuffle = shuffle
        self.augment = augment
        self.augmentation = augmentation
        self.random_rois = random_rois
        self.batch_size = batch_size
        self.detection_targets = detection_targets
        self.no_augmentation_sources = no_augmentation_sources or []

    def __len__(self):
        return int(np.ceil(len(self.image_ids) / float(self.batch_size)))

    def __getitem__(self, idx):
        return self.data_generator(self.image_ids[idx * self.batch_size:(idx + 1) *
                                                  self.batch_size])

    def data_generator(self, image_ids):
        b = 0
        while b < self.batch_size and b < image_ids.shape[0]:
            try:
                # Get GT bounding boxes and masks for image.
                image_id = image_ids[b]

                # If the image source is not to be augmented pass None as augmentation
                if self.dataset.image_info[image_id]['source'] in \
                        self.no_augmentation_sources:
                    image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                        load_image_gt(self.dataset, self.config, image_id,
                                      augment=self.augment, augmentation=None,
                                      use_mini_mask=self.config.USE_MINI_MASK)
                else:
                    image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                        load_image_gt(self.dataset, self.config, image_id,
                                      augment=self.augment,
                                      augmentation=self.augmentation,
                                      use_mini_mask=self.config.USE_MINI_MASK)

                # Anchors
                anchors = utils.get_anchors(self.config.IMAGE_SHAPE, self.config)

                # RPN Targets
                rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                        gt_class_ids, gt_boxes,
                                                        self.config)

                # Mask R-CNN Targets
                if self.random_rois:
                    rpn_rois = generate_random_rois(
                        image.shape, self.random_rois, gt_class_ids, gt_boxes)
                    if self.detection_targets:
                        rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                            build_detection_targets(
                                rpn_rois, gt_class_ids, gt_boxes, gt_masks, self.config)

                # Init batch arrays
                if b == 0:
                    batch_image_meta = np.zeros(
                        (self.batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                    batch_rpn_match = np.zeros((self.batch_size, anchors.shape[0], 1),
                                               dtype=rpn_match.dtype)
                    batch_rpn_bbox = np.zeros(
                        (self.batch_size, self.config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4),
                        dtype=rpn_bbox.dtype)
                    batch_images = np.zeros(
                        (self.batch_size,) + image.shape, dtype=np.float32)
                    batch_gt_class_ids = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES), dtype=np.int32)
                    batch_gt_boxes = np.zeros(
                        (self.batch_size, self.config.MAX_GT_INSTANCES, 4),
                        dtype=np.int32)
                    batch_gt_masks = np.zeros(
                        (self.batch_size, gt_masks.shape[0], gt_masks.shape[1],
                         self.config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                    batch_anchors = np.zeros((self.batch_size,) + anchors.shape,
                                             dtype=anchors.dtype)
                    if self.random_rois:
                        batch_rpn_rois = np.zeros(
                            (self.batch_size, rpn_rois.shape[0], 4),
                            dtype=rpn_rois.dtype)
                        if self.detection_targets:
                            batch_rois = np.zeros(
                                (self.batch_size,) + rois.shape, dtype=rois.dtype)
                            batch_mrcnn_class_ids = np.zeros(
                                (self.batch_size,) + mrcnn_class_ids.shape,
                                dtype=mrcnn_class_ids.dtype)
                            batch_mrcnn_bbox = np.zeros(
                                (self.batch_size,) + mrcnn_bbox.shape,
                                dtype=mrcnn_bbox.dtype)
                            batch_mrcnn_mask = np.zeros(
                                (self.batch_size,) + mrcnn_mask.shape,
                                dtype=mrcnn_mask.dtype)

                # If more instances than fits in the array, sub-sample from them.
                if gt_boxes.shape[0] > self.config.MAX_GT_INSTANCES:
                    ids = np.random.choice(np.arange(gt_boxes.shape[0]),
                                           self.config.MAX_GT_INSTANCES,
                                           replace=False)
                    gt_class_ids = gt_class_ids[ids]
                    gt_boxes = gt_boxes[ids]
                    gt_masks = gt_masks[:, :, ids]

                # Add to batch
                batch_image_meta[b] = image_meta
                batch_rpn_match[b] = rpn_match[:, np.newaxis]
                batch_rpn_bbox[b] = rpn_bbox
                batch_images[b] = mold_image(image.astype(np.float32), self.config)
                batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
                batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
                batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
                batch_anchors[b] = anchors
                if self.random_rois:
                    batch_rpn_rois[b] = rpn_rois
                    if self.detection_targets:
                        batch_rois[b] = rois
                        batch_mrcnn_class_ids[b] = mrcnn_class_ids
                        batch_mrcnn_bbox[b] = mrcnn_bbox
                        batch_mrcnn_mask[b] = mrcnn_mask
                b += 1

                # Batch full?
                if b >= self.batch_size:
                    inputs = {
                        'input_image': batch_images,
                        'input_image_meta': batch_image_meta,
                        'input_rpn_match': batch_rpn_match,
                        'input_rpn_bbox': batch_rpn_bbox,
                        'input_gt_class_ids': batch_gt_class_ids,
                        'input_gt_boxes': batch_gt_boxes,
                        'input_gt_masks': batch_gt_masks,
                        'input_anchors': batch_anchors
                    }
                    print(batch_rpn_bbox.shape)
                    outputs = {}

                    if self.random_rois:
                        inputs['input_roi'] = batch_rpn_rois
                        if self.detection_targets:
                            inputs['batch_rois'] = batch_rois
                            # Keras requires that output and targets have the same
                            # number of dimensions
                            batch_mrcnn_class_ids = np.expand_dims(
                                batch_mrcnn_class_ids, -1)
                            outputs['mrcnn_class'] = batch_mrcnn_class_ids
                            outputs['mrcnn_bbox'] = batch_mrcnn_bbox
                            outputs['mrcnn_mask'] = batch_mrcnn_mask

                    return inputs, outputs

            except (GeneratorExit, KeyboardInterrupt):
                raise
            except Exception:
                # Log it and skip the image
                logging.exception("Error processing image {}".format(
                    self.dataset.image_info[image_id]))
                self.error_count += 1
                if self.error_count > 5:
                    raise

    def on_epoch_end(self):
        if self.shuffle is True:
            np.random.shuffle(self.image_ids)


############################################################
#  MaskRCNN Class
############################################################

class MaskRCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir if model_dir else '.'
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        image_shape = (None, None, config.IMAGE_SHAPE[2])
        input_image = Input(shape=image_shape, name='input_image')
        input_image_meta = Input(shape=[config.IMAGE_META['size']],
                                 name='input_image_meta')

        # Anchors in normalized coordinates
        input_anchors = Input(shape=(None, 4), name='input_anchors')

        if mode == 'training':
            # RPN GT
            input_rpn_match = Input(shape=[None, 1], name='input_rpn_match',
                                    dtype=tf.int32)
            input_rpn_bbox = Input(shape=[None, 4], name='input_rpn_bbox',
                                   dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = Input(shape=[None], name='input_gt_class_ids',
                                       dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = Input(shape=[None, 4], name='input_gt_boxes',
                                   dtype=tf.float32)
            # Normalize coordinates
            gt_boxes = Lambda(lambda x: norm_boxes_graph(x, (h, w)))(
                input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = Input(shape=config.MINI_MASK_SHAPE + (None,),
                                       name='input_gt_masks', dtype=bool)
            else:
                input_gt_masks = Input(shape=config.IMAGE_SHAPE[:1] + (None,),
                                       name='input_gt_masks', dtype=bool)

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        if callable(config.BACKBONE):
            _, C2, C3, C4, C5 = config.BACKBONE(input_image, stage5=True,
                                                train_bn=config.TRAIN_BN)
        elif config.BACKBONE == 'resnet50':
            model = tf.keras.applications.ResNet50V2(include_top=False,
                                                     weights=None,
                                                     input_tensor=input_image,
                                                     input_shape=image_shape,
                                                     pooling=None)
            C2 = model.get_layer('conv2_block3_preact_relu').output
            C3 = model.get_layer('conv3_block4_preact_relu').output
            C4 = model.get_layer('conv4_block6_preact_relu').output
            C5 = model.get_layer('post_relu').output
        elif config.BACKBONE == 'resnet101':
            model = tf.keras.applications.ResNet101V2(include_top=False,
                                                      weights=None,
                                                      input_tensor=input_image,
                                                      input_shape=image_shape,
                                                      pooling=None)
            C2 = model.get_layer('conv2_block3_out').output
            C3 = model.get_layer('conv3_block4_out').output
            C4 = model.get_layer('conv4_block23_out').output
            C5 = model.get_layer('conv5_block3_out').output
        else:
            return None

        # Top-down Layers
        # TODO: add assert to varify feature map sizes match what's in config
        P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
        P4 = Add(name='fpn_p4add')([
            UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5),
            Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c4p4')(C4)
        ])
        P3 = Add(name='fpn_p3add')([
            UpSampling2D(size=(2, 2), name='fpn_p4upsampled')(P4),
            Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c3p3')(C3)
        ])
        P2 = Add(name='fpn_p2add')([
            UpSampling2D(size=(2, 2), name='fpn_p3upsampled')(P3),
            Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c2p2')(C2)
        ])
        # Attach 3x3 conv to all P layers to get the final feature maps.
        P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME',
                    name='fpn_p2')(P2)
        P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME',
                    name='fpn_p3')(P3)
        P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME',
                    name='fpn_p4')(P4)
        P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3), padding='SAME',
                    name='fpn_p5')(P5)
        # P6 is used for the 5th anchor scale in RPN. Generated by
        # subsampling from P5 with stride of 2.
        P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)

        # RPN Model
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn = RPN(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS),
                  config.TOP_DOWN_PYRAMID_SIZE)
        P2_class_logits, P2_class, P2_bbox = rpn(P2)
        P3_class_logits, P3_class, P3_bbox = rpn(P3)
        P4_class_logits, P4_class, P4_bbox = rpn(P4)
        P5_class_logits, P5_class, P5_bbox = rpn(P5)
        P6_class_logits, P6_class, P6_bbox = rpn(P6)

        rpn_class_logits = Concatenate(name='rpn_class_logits',
                                       axis=1)([P2_class_logits,
                                                P3_class_logits,
                                                P4_class_logits,
                                                P5_class_logits,
                                                P6_class_logits])
        rpn_class = Concatenate(axis=1, name='rpn_class')([P2_class, P3_class,
                                                           P4_class, P5_class,
                                                           P6_class])
        rpn_bbox = Concatenate(axis=1, name='rpn_bbox')([P2_bbox, P3_bbox,
                                                         P4_bbox, P5_bbox,
                                                         P6_bbox])

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == 'training' \
            else config.POST_NMS_ROIS_INFERENCE
        rpn_rois = Proposals(proposal_count=proposal_count,
                             nms_threshold=config.RPN_NMS_THRESHOLD,
                             pre_nms_limit=config.PRE_NMS_LIMIT,
                             rpn_bbox_std_dev=config.RPN_BBOX_STD_DEV,
                             name='ROI')([rpn_class, rpn_bbox, input_anchors])

        if mode == 'training':
            # Class ID mask to mark class IDs supported by the dataset the
            # image came from.
            active_class_ids = Lambda(lambda x: x[:, 12:])(input_image_meta)

            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                   name='input_roi', dtype=np.int32)
                # Normalize coordinates
                target_rois = Lambda(lambda x: norm_boxes_graph(x, (h, w)))(
                    input_rois)
            else:
                target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask = \
                DetectionTarget(config.TRAIN_ROIS_PER_IMAGE, config.ROI_POSITIVE_RATIO,
                                config.MASK_SHAPE, config.BBOX_STD_DEV,
                                config.USE_MINI_MASK, name='proposal_targets')(
                                [target_rois, input_gt_class_ids, gt_boxes,
                                 input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(rois, [P2, P3, P4, P5], input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     config.IMAGE_META, config.TRAIN_BN,
                                     config.FPN_CLASSIF_FC_LAYERS_SIZE)

            mrcnn_mask = build_fpn_mask_graph(rois, [P2, P3, P4, P5], input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES, config.IMAGE_META,
                                              train_bn=config.TRAIN_BN)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = Lambda(lambda x: x * 1, name='output_rois')(rois)

            # Losses
            rpn_class_loss = Lambda(lambda x: rpn_class_loss_graph(*x),
                                    name='rpn_class_loss')([input_rpn_match,
                                                            rpn_class_logits])
            rpn_bbox_loss = Lambda(lambda x: rpn_bbox_loss_graph(*x),
                                   name='rpn_bbox_loss')([input_rpn_bbox,
                                                          input_rpn_match,
                                                          rpn_bbox])
            class_loss = Lambda(lambda x: mrcnn_class_loss_graph(*x),
                                name='mrcnn_class_loss')([rois,
                                                          target_class_ids,
                                                          mrcnn_class_logits,
                                                          active_class_ids])
            bbox_loss = Lambda(lambda x: mrcnn_bbox_loss_graph(*x),
                               name='mrcnn_bbox_loss')([target_bbox,
                                                        target_class_ids,
                                                        mrcnn_bbox])
            mask_loss = Lambda(lambda x: mrcnn_mask_loss_graph(*x),
                               name='mrcnn_mask_loss')([target_mask,
                                                        target_class_ids,
                                                        mrcnn_mask])

            # Model
            inputs = [input_image, input_image_meta, input_rpn_match,
                      input_rpn_bbox, input_gt_class_ids, input_gt_boxes,
                      input_gt_masks, input_anchors]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois, rpn_class_loss, rpn_bbox_loss,
                       class_loss, bbox_loss, mask_loss]
            model = Model(inputs, outputs, name='mask_rcnn')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox = \
                fpn_classifier_graph(rpn_rois, [P2, P3, P4, P5], input_image_meta,
                                     config.POOL_SIZE, config.NUM_CLASSES,
                                     config.IMAGE_META, train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id,
            #                                    score)] in
            # normalized coordinates
            detections = DetectionLayer(config.DETECTION_MAX_INSTANCES,
                                        config.DETECTION_MIN_CONFIDENCE,
                                        config.DETECTION_NMS_THRESHOLD,
                                        config.BBOX_STD_DEV,
                                        name='mrcnn_detection')(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Create masks for detections
            detection_boxes = Lambda(lambda x: x[..., :4])(detections)
            mrcnn_mask = build_fpn_mask_graph(detection_boxes,
                                              [P2, P3, P4, P5],
                                              input_image_meta,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES, config.IMAGE_META,
                                              train_bn=config.TRAIN_BN)

            model = Model([input_image, input_image_meta, input_anchors],
                          [detections, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                           rpn_rois, rpn_class, rpn_bbox], name='mask_rcnn')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            The path of the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            import errno
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find model directory under {}".format(self.model_dir))
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("mask_rcnn"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            import errno
            raise FileNotFoundError(errno.ENOENT, f'Could not find weight ' +
                                    'files in {dir_name}')
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers \
            if hasattr(keras_model, 'inner_model') else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def get_imagenet_weights(self):
        """Downloads ImageNet trained weights from Keras.
        Returns path to weights file.
        """
        from keras.utils.data_utils import get_file
        TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/'\
                                 'releases/download/v0.2/'\
                                 'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
        weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                TF_WEIGHTS_PATH_NO_TOP,
                                cache_subdir='models',
                                md5_hash='a268eb855778b3df3c7506639542a6af')
        return weights_path

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = tf.keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        loss_names = [
            "rpn_class_loss",  "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        output_names = []
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output.name in output_names:
                continue
            loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) *
                    self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)
            output_names.append(layer.output.name)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [tf.keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) /
                      tf.cast(tf.size(w), tf.float32)
                      for w in self.keras_model.trainable_weights
                      if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (tf.reduce_mean(input_tensor=layer.output, keepdims=True) *
                    self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_metric(loss, name=name, aggregation='mean')

    def set_trainable(self, layer_regex, keras_model=None, indent=0,
                      verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if layer_regex is None:
            return

        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model,
                                                           'inner_model') \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print(f'In model: {layer.name}')
                self.set_trainable(layer_regex, keras_model=layer,
                                   indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir,
                                    f'{self.config.NAME.lower()}' +
                                    f'{now:%Y%m%dT%H%M}')

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "mask_rcnn_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None, custom_callbacks=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        custom_callbacks: Optional. Add custom callbacks to be called
            with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)|(fpn\_.*)",
            # All layers
            "all": ".*",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = DataGenerator(train_dataset, self.config,
                                        shuffle=True,
                                        augmentation=augmentation,
                                        batch_size=self.config.BATCH_SIZE,
                                        no_augmentation_sources=no_augmentation_sources)
        if val_dataset:
            val_generator = DataGenerator(val_dataset, self.config,
                                          shuffle=True,
                                          batch_size=self.config.BATCH_SIZE)
        else:
            val_generator = None

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            tf.keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                           histogram_freq=0, write_graph=True,
                                           write_images=False),
            tf.keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                               verbose=0,
                                               save_weights_only=True),
            tf.keras.callbacks.TerminateOnNaN()
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log(f'\nStarting at epoch {self.epoch}. LR={learning_rate}\n')
        log(f'Checkpoint Path: {self.checkpoint_path}')
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name == 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=2, # workers
            use_multiprocessing=False, # True
        )
        self.epoch = max(self.epoch, epochs)

    def evaluate(self, val_dataset):
        """Evaluate the model. Return configured metrics.
        val_dataset: validation Dataset object.
        """
        # require training mode for loss layers
        assert self.mode == "training", "Create model in training mode."

        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.compile(self.config.LEARNING_RATE, self.config.LEARNING_MOMENTUM)
        val_generator = DataGenerator(val_dataset, self.config, shuffle=True,
                                      batch_size=self.config.BATCH_SIZE)
        return self.keras_model.evaluate_generator(
            val_generator,
            steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=2, # workers
            use_multiprocessing=False, # True
        )

    def mold_inputs(self, images, active_class_ids=None):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.
        active_class_ids: List of class_ids allowed for the given images.

        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        if active_class_ids:
            active_classes = np.zeros([self.config.NUM_CLASSES], dtype=np.int32)
            active_classes[active_class_ids] = 1
        else:
            active_classes = np.ones([self.config.NUM_CLASSES], dtype=np.int32)
        for image in images:
            # Resize image
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding, crop = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                min_scale=self.config.IMAGE_MIN_SCALE,
                max_dim=self.config.IMAGE_MAX_DIM,
                mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, molded_image.shape, window, scale,
                active_classes)
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, original_image_shape,
                          image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)] in normalized coordinates
        mrcnn_mask: [N, height, width, num_classes]
        original_image_shape: [H, W, C] Original image shape before resizing
        image_shape: [H, W, C] Shape of the image after resizing and padding
        window: [y1, x1, y2, x2] Pixel coordinates of box in the image where the real
                image is excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))

        return boxes, class_ids, scores, full_masks

    def detect(self, images, verbose=0, active_class_ids=None):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.
        active_class_ids: List of class_ids allowed for the given images.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images, active_class_ids)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape,\
                "After resizing, all images must have the same size. " + \
                "Check IMAGE_RESIZE_MODE and image sizes."

        # Anchors
        anchors = utils.get_anchors(image_shape, self.config)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def detect_molded(self, molded_images, image_metas, verbose=0):
        """Runs the detection pipeline, but expect inputs that are
        molded already. Used mostly for debugging and inspecting
        the model.

        molded_images: List of images loaded using load_image_gt()
        image_metas: image meta data, also returned by load_image_gt()

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(molded_images) == self.config.BATCH_SIZE,\
            "Number of images must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(molded_images)))
            for image in molded_images:
                log("image", image)

        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        for g in molded_images[1:]:
            assert g.shape == image_shape, "Images must have the same size"

        # Anchors
        anchors = self.get_anchors(image_shape, self.config)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            log("molded_images", molded_images)
            log("image_metas", image_metas)
            log("anchors", anchors)
        # Run object detection
        detections, _, _, mrcnn_mask, _, _, _ =\
            self.keras_model.predict([molded_images, image_metas, anchors], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(molded_images):
            window = [0, 0, image.shape[0], image.shape[1]]
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_detections(detections[i], mrcnn_mask[i],
                                       image.shape, molded_images[i].shape,
                                       window)
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
            })
        return results

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for layer in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            layer = self.find_trainable_layer(layer)
            # Include layer if it has weights
            if layer.get_weights():
                layers.append(layer)
        return layers

    def run_graph(self, images, outputs, image_metas=None):
        """Runs a sub-set of the computation graph that computes the given
        outputs.

        image_metas: If provided, the images are assumed to be already
            molded (i.e. resized, padded, and normalized)

        outputs: List of tuples (name, tensor) to compute. The tensors are
            symbolic TensorFlow tensors and the names are for easy tracking.

        Returns an ordered dict of results. Keys are the names received in the
        input and values are Numpy arrays.
        """
        model = self.keras_model

        # Organize desired outputs into an ordered dict
        outputs = OrderedDict(outputs)
        for o in outputs.values():
            assert o is not None

        # Build a Keras function to run parts of the computation graph
        inputs = model.inputs
        if model.uses_learning_phase and not isinstance(
                tf.keras.backend.learning_phase(), int):
            inputs += [tf.keras.backend.learning_phase()]
        kf = tf.keras.backend.function(model.inputs, list(outputs.values()))

        # Prepare inputs
        if image_metas is None:
            molded_images, image_metas, _ = self.mold_inputs(images)
        else:
            molded_images = images
        image_shape = molded_images[0].shape
        # Anchors
        anchors = self.get_anchors(image_shape, self.config)
        # Duplicate across the batch dimension because Keras requires it
        # TODO: can this be optimized to avoid duplicating the anchors?
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)
        model_in = [molded_images, image_metas, anchors]

        # Run inference
        if model.uses_learning_phase and not isinstance(
                tf.keras.backend.learning_phase(), int):
            model_in.append(0.)
        outputs_np = kf(model_in)

        # Pack the generated Numpy arrays into a a dict and log the results.
        outputs_np = OrderedDict([(k, v)
                                  for k, v in zip(outputs.keys(), outputs_np)])
        for k, v in outputs_np.items():
            log(k, v)
        return outputs_np


############################################################
#  Data Formatting
############################################################

def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (y1, x1, y2, x2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta


def parse_image_meta(meta):
    """Parses an array that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed values.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        'image_id': image_id,
        'original_image_shape': original_image_shape,
        'image_shape': image_shape,
        'window': window,
        'scale': scale,
        'active_class_ids': active_class_ids,
    }


def parse_image_meta_graph(meta):
    """Parses a tensor that contains image attributes to its components.
    See compose_image_meta() for more details.

    meta: [batch, meta length] where meta length depends on NUM_CLASSES

    Returns a dict of the parsed tensors.
    """
    image_id = meta[:, 0]
    original_image_shape = meta[:, 1:4]
    image_shape = meta[:, 4:7]
    window = meta[:, 7:11]  # (y1, x1, y2, x2) window of image in in pixels
    scale = meta[:, 11]
    active_class_ids = meta[:, 12:]
    return {
        'image_id': image_id,
        'original_image_shape': original_image_shape,
        'image_shape': image_shape,
        'window': window,
        'scale': scale,
        'active_class_ids': active_class_ids,
    }


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)


############################################################
#  Miscellenous Graph Functions
############################################################

def trim_zeros_graph(boxes, name='trim_zeros'):
    """Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    boxes: [N, 4] matrix of boxes.
    non_zeros: [N] a 1D boolean mask identifying the rows to keep
    """
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros, name=name)
    return boxes, non_zeros


def batch_pack_graph(x, counts, num_rows):
    """Picks different number of values from each row
    in x depending on the values in counts.
    """
    outputs = []
    for i in range(num_rows):
        outputs.append(x[i, :counts[i]])
    return tf.concat(outputs, axis=0)


def norm_boxes_graph(boxes, shape):
    """Converts boxes from pixel coordinates to normalized coordinates.
    boxes: [..., (y1, x1, y2, x2)] in pixel coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in normalized coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.divide(boxes - shift, scale)


def denorm_boxes_graph(boxes, shape):
    """Converts boxes from normalized coordinates to pixel coordinates.
    boxes: [..., (y1, x1, y2, x2)] in normalized coordinates
    shape: [..., (height, width)] in pixels

    Note: In pixel coordinates (y2, x2) is outside the box. But in normalized
    coordinates it's inside the box.

    Returns:
        [..., (y1, x1, y2, x2)] in pixel coordinates
    """
    h, w = tf.split(tf.cast(shape, tf.float32), 2)
    scale = tf.concat([h, w, h, w], axis=-1) - tf.constant(1.0)
    shift = tf.constant([0., 0., 1., 1.])
    return tf.cast(tf.round(tf.multiply(boxes, scale) + shift), tf.int32)
