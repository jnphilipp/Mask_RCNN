import numpy as np
import tensorflow as tf

from mrcnn import utils
from tensorflow.keras.layers import Layer
from typing import Optional, Tuple


############################################################
#  DetectionTarget Layer
############################################################

class DetectionTarget(Layer):
    """Subsamples proposals and generates target box refinement, class_ids,
    and masks for each.

    Inputs:
    proposals: [batch, N, (y1, x1, y2, x2)] in normalized coordinates. Might
               be zero padded if there are not enough proposals.
    gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs.
    gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
              coordinates.
    gt_masks: [batch, height, width, MAX_GT_INSTANCES] of boolean type

    Returns: Target ROIs and corresponding class IDs, bounding box shifts,
    and masks.
    rois: [batch, TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
          coordinates
    target_class_ids: [batch, TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
    target_deltas: [batch, TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw)]
    target_mask: [batch, TRAIN_ROIS_PER_IMAGE, height, width]
                 Masks cropped to bbox boundaries and resized to neural
                 network output size.

    Note: Returned arrays might be zero padded if not enough target ROIs.
    """

    def __init__(self, train_rois_per_image: int, roi_positive_ratio: float,
                 mask_shape: Tuple[int, int], bbox_std_dev: np.ndarray,
                 use_mini_mask: bool, **kwargs):
        super(DetectionTarget, self).__init__(**kwargs)
        self.train_rois_per_image = train_rois_per_image
        self.roi_positive_ratio = roi_positive_ratio
        self.mask_shape = mask_shape
        self.bbox_std_dev = bbox_std_dev
        self.use_mini_mask = use_mini_mask

    def get_config(self):
        config = super(DetectionTarget, self).get_config()
        config['train_rois_per_image'] = self.train_rois_per_image
        config['roi_positive_ratio'] = self.roi_positive_ratio
        config['mask_shape'] = self.mask_shape
        config['bbox_std_dev'] = self.bbox_std_dev
        config['use_mini_mask'] = self.use_mini_mask
        return config

    def call(self, inputs):
        input_shape = (inputs[0].shape, inputs[1].shape, inputs[2].shape,
                       inputs[3].shape)

        # Slice the batch and run a graph for each slice
        # rois, target_class_ids, target_deltas, target_masks = \
        #     tf.map_fn(self._detection_targets_graph, inputs,
        #               fn_output_signature=[tf.float32, tf.int32, tf.float32,
        #                                    tf.float32])

        rois, target_class_ids, target_deltas, target_masks = \
            tf.nest.map_structure(tf.stop_gradient,
                                  tf.map_fn(self._detection_targets_graph, inputs,
                                            fn_output_signature=[tf.float32, tf.int32,
                                                                 tf.float32,
                                                                 tf.float32]))

        output_shape = self.compute_output_shape(input_shape)

        rois.set_shape(output_shape[0])
        target_class_ids.set_shape(output_shape[1])
        target_deltas.set_shape(output_shape[2])
        target_masks.set_shape(output_shape[3])

        return rois, target_class_ids, target_deltas, target_masks

    def compute_output_shape(self, input_shape):
        """
        Returns:
        rois, class_ids, deltas, masks
        """
        batch_size = tf.TensorShape(input_shape[0]).as_list()[0]
        return (
            tf.TensorShape((batch_size, self.train_rois_per_image, 4)),
            tf.TensorShape((batch_size, self.train_rois_per_image)),
            tf.TensorShape((batch_size, self.train_rois_per_image, 4)),
            tf.TensorShape((batch_size, self.train_rois_per_image) +
                           self.mask_shape)
        )

    def compute_mask(self, inputs, mask=None):
        return [None, None, None, None]

    def _detection_targets_graph(self, inputs):
        """Generates detection targets for one image. Subsamples proposals and
        generates target class IDs, bounding box deltas, and masks for each.

        Inputs:
        proposals: [POST_NMS_ROIS_TRAINING, (y1, x1, y2, x2)] in normalized
                   coordinates. Might be zero padded if there are not enough
                   proposals.
        gt_class_ids: [MAX_GT_INSTANCES] int class IDs
        gt_boxes: [MAX_GT_INSTANCES, (y1, x1, y2, x2)] in normalized
                  coordinates.
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.

        Returns: Target ROIs and corresponding class IDs, bounding box shifts,
                 and masks.
                 * rois: [TRAIN_ROIS_PER_IMAGE, (y1, x1, y2, x2)] in normalized
                         coordinates. Zero padded.
                 * class_ids: [TRAIN_ROIS_PER_IMAGE]. Integer class IDs.
                              Zero padded.
                 * deltas: [TRAIN_ROIS_PER_IMAGE, (dy, dx, log(dh), log(dw))].
                           Zero padded.
                 * masks: [TRAIN_ROIS_PER_IMAGE, height, width]. Masks cropped
                          to bbox boundaries and resized to neural network
                          output size. Zero padded.

        Note: Returned arrays might be zero padded if not enough target ROIs.
        """

        proposals = inputs[0]
        gt_class_ids = inputs[1]
        gt_boxes = inputs[2]
        gt_masks = inputs[3]

        # Assertions
        asserts = [tf.Assert(tf.greater(tf.shape(proposals)[0], 0),
                             [proposals], name='roi_assertion')]
        with tf.control_dependencies(asserts):
            proposals = tf.identity(proposals)

        # Remove zero padding
        proposals, _ = self._trim_zeros(proposals, name='trim_proposals')
        gt_boxes, non_zeros = self._trim_zeros(gt_boxes, name='trim_gt_boxes')
        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros,
                                       name='trim_gt_class_ids')
        gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                             name='trim_gt_masks')

        # Handle COCO crowds
        # A crowd box in COCO is a bounding box around several instances.
        # Exclude them from training. A crowd box is given a negative class ID.
        crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
        non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]
        crowd_boxes = tf.gather(gt_boxes, crowd_ix)
        gt_class_ids = tf.gather(gt_class_ids, non_crowd_ix)
        gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
        gt_masks = tf.gather(gt_masks, non_crowd_ix, axis=2)

        # Compute overlaps matrix [proposals, gt_boxes]
        overlaps = self._overlaps_graph(proposals, gt_boxes)

        # Compute overlaps with crowd boxes [proposals, crowd_boxes]
        crowd_overlaps = self._overlaps_graph(proposals, crowd_boxes)
        # Tensorflow#31325 says reduce_max is allowed to produce
        # a non-empty result filled with FLOAT_MAX here (-inf)
        # so all proposals will be no_crowd_bool if crowd_boxes is empty
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        no_crowd_bool = (crowd_iou_max < 0.001)

        # Determine positive and negative ROIs
        # Tensorflow#31325 says reduce_max is allowed to produce
        # a non-empty result filled with FLOAT_MAX here (-inf)
        # so all proposals will be negative if gt_boxes is empty
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        # 1. Positive ROIs are those with >= 0.5 IoU with a GT box
        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        # 2. Negative ROIs are those with < 0.5 with every GT box. Skip crowds.
        negative_indices = tf.where(tf.logical_and(roi_iou_max < 0.5,
                                    no_crowd_bool))[:, 0]

        # Subsample ROIs. Aim for 33% positive
        # Positive ROIs
        positive_count = int(self.train_rois_per_image *
                             self.roi_positive_ratio)
        positive_indices = tf.random.shuffle(positive_indices)[:positive_count]
        positive_count = tf.shape(positive_indices)[0]
        # Negative ROIs. Add enough to maintain positive:negative ratio.
        r = 1.0 / self.roi_positive_ratio

        # But if there are too few or no positive ROIs (because of
        # too few GT instances or too inaccurate RPN), then ensure
        # that we have enough negative ROIs to avoid zero padding.
        # (However, there could still be too few negative ROIs if
        #  the RPN predicts too well.)
        negative_count = tf.cond(
            tf.greater(positive_count, 0),
            true_fn=lambda: tf.cast(r * tf.cast(positive_count, tf.float32),
                                    tf.int32) - positive_count,
            false_fn=lambda: tf.constant(self.train_rois_per_image))
        negative_indices = tf.random.shuffle(negative_indices)[:negative_count]

        # Gather selected ROIs
        positive_rois = tf.gather(proposals, positive_indices)
        negative_rois = tf.gather(proposals, negative_indices)

        # Assign positive ROIs to GT boxes.
        positive_overlaps = tf.gather(overlaps, positive_indices)
        roi_gt_box_assignment = tf.cond(
            tf.greater(tf.shape(positive_overlaps)[1], 0),
            true_fn=lambda: tf.argmax(positive_overlaps, axis=1),
            false_fn=lambda: tf.cast(tf.constant([]), tf.int64)
        )
        roi_gt_boxes = tf.gather(gt_boxes, roi_gt_box_assignment)
        roi_gt_class_ids = tf.gather(gt_class_ids, roi_gt_box_assignment)

        # Compute bbox refinement for positive ROIs
        deltas = utils.box_refinement_graph(positive_rois, roi_gt_boxes)
        deltas /= self.bbox_std_dev

        # Assign positive ROIs to GT masks
        # Permute masks to [N, height, width, 1]
        transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]),
                                          -1)
        # Pick the right mask for each ROI
        roi_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

        # Compute mask targets
        boxes = positive_rois
        if self.use_mini_mask:
            # Transform ROI coordinates from normalized image space
            # to normalized mini-mask space.
            y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_boxes, 4, axis=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = tf.concat([y1, x1, y2, x2], 1)
        box_ids = tf.range(0, tf.shape(roi_masks)[0])
        masks = tf.image.crop_and_resize(tf.cast(roi_masks, tf.float32), boxes,
                                         box_ids, self.mask_shape)
        # Remove the extra dimension from masks.
        masks = tf.squeeze(masks, axis=3)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with
        # binary cross entropy loss.
        masks = tf.round(masks)

        # Append negative ROIs and pad their class IDs, bbox deltas and masks
        # with zeros (because they are discounted above).
        # Moreover, pad remaining ROIs up to TRAIN_ROIS_PER_IMAGE  (=P) with
        # zeros, because we need a constant output shape.
        # So from now on we need to distinguish between zeros from padding
        # (to be masked) and zeros from negative ROIs (to be learned), esp.
        # in mrcnn_class_loss_graph and mrcnn_bbox_loss_graph, suppressing
        # indexes with zero bbox size but not indexes with zero class IDs.
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(self.train_rois_per_image - tf.shape(rois)[0], 0)
        rois = tf.pad(rois, [(0, P), (0, 0)])
        roi_gt_boxes = tf.pad(roi_gt_boxes, [(0, N + P), (0, 0)])
        roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, N + P)])
        deltas = tf.pad(deltas, [(0, N + P), (0, 0)])
        masks = tf.pad(masks, [[0, N + P], (0, 0), (0, 0)])

        rois = tf.cast(rois, tf.float32)
        roi_gt_boxes = tf.cast(roi_gt_boxes, tf.int32)
        deltas = tf.cast(deltas, tf.float32)
        masks = tf.cast(masks, tf.float32)

        return [rois, roi_gt_class_ids, deltas, masks]

    def _overlaps_graph(self, boxes1, boxes2):
        """Computes IoU overlaps between two sets of boxes.
        boxes1, boxes2: [N, (y1, x1, y2, x2)].
        """
        # 1. Tile boxes2 and repeat boxes1. This allows us to compare
        # every boxes1 against every boxes2 without loops.
        # TF doesn't have an equivalent to np.repeat() so simulate it
        # using tf.tile() and tf.reshape.
        b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                                [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
        b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
        # 2. Compute intersections
        b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
        b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
        y1 = tf.maximum(b1_y1, b2_y1)
        x1 = tf.maximum(b1_x1, b2_x1)
        y2 = tf.minimum(b1_y2, b2_y2)
        x2 = tf.minimum(b1_x2, b2_x2)
        intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
        # 3. Compute unions
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection
        # 4. Compute IoU and reshape to [boxes1, boxes2]
        iou = intersection / union
        overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
        return overlaps

    def _trim_zeros(self, boxes, name):
        """Often boxes are represented with matrices of shape [N, 4] and
        are padded with zeros. This removes zero boxes.

        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
        """
        non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), tf.bool)
        boxes = tf.boolean_mask(boxes, non_zeros, name=name)
        return boxes, non_zeros


############################################################
#  Proposal Layer
############################################################

class Proposals(Layer):
    """Receives anchor scores and selects a subset to pass as proposals
    to the second stage. Filtering is done based on anchor scores and
    non-max suppression to remove overlaps. It also applies bounding
    box refinement deltas to anchors.

    Inputs:
        rpn_probs: [batch, num_anchors, (bg prob, fg prob)]
        rpn_bbox: [batch, num_anchors, (dy, dx, log(dh), log(dw))]
        anchors: [batch, num_anchors, (y1, x1, y2, x2)] anchors in normalized
                    coordinates

    Returns:
        Proposals in normalized coordinates [batch, rois, (y1, x1, y2, x2)]
    """

    def __init__(self, proposal_count: int, nms_threshold: float,
                 rpn_bbox_std_dev: np.ndarray, pre_nms_limit: int, **kwargs):
        super(Proposals, self).__init__(**kwargs)
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.rpn_bbox_std_dev = rpn_bbox_std_dev
        self.pre_nms_limit = pre_nms_limit

    def call(self, inputs):
        input_shape = (inputs[0].shape, inputs[1].shape, inputs[2].shape)

        # Box Scores. Use the foreground class confidence. [Batch, num_rois, 1]
        scores = inputs[0][:, :, 1]

        # Box deltas [batch, num_rois, 4]
        deltas = inputs[1]
        deltas *= np.reshape(self.rpn_bbox_std_dev, [1, 1, 4])

        # Anchors
        anchors = inputs[2]

        # Improve performance by trimming to top anchors by score
        # and doing the rest on the smaller subset.
        pre_nms_limit = tf.minimum(self.pre_nms_limit, tf.shape(anchors)[1])
        ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True,
                         name='top_anchors').indices

        scores = tf.map_fn(lambda x: tf.gather(x[0], x[1]), (scores, ix),
                           fn_output_signature=tf.float32)
        deltas = tf.map_fn(lambda x: tf.gather(x[0], x[1]), (deltas, ix),
                           fn_output_signature=tf.float32)
        pre_nms_anchors = tf.map_fn(lambda x: tf.gather(x[0], x[1]),
                                    (anchors, ix),
                                    fn_output_signature=tf.float32)

        # Apply deltas to anchors to get refined anchors.
        # [batch, N, (y1, x1, y2, x2)]
        boxes = tf.map_fn(self._box_deltas, (pre_nms_anchors, deltas),
                          fn_output_signature=tf.float32)

        # Clip to image boundaries. Since we're in normalized coordinates,
        # clip to 0..1 range. [batch, N, (y1, x1, y2, x2)]
        boxes = tf.map_fn(self._clip_boxes, boxes)

        # Filter out small boxes
        # According to Xinlei Chen's paper, this reduces detection accuracy
        # for small objects, so we're skipping it.
        proposals = tf.map_fn(self._nms, (boxes, scores),
                              fn_output_signature=tf.float32)
        proposals.set_shape(self.compute_output_shape(input_shape))
        return proposals

    def _box_deltas(self, inputs: Tuple[tf.Tensor, tf.Tensor]):
        """Applies the given deltas to the given boxes.
        boxes: [N, (y1, x1, y2, x2)] boxes to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        """
        boxes = inputs[0]
        deltas = inputs[0]
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
        result = tf.stack([y1, x1, y2, x2], axis=1, name='box_deltas_out')
        return result

    def _clip_boxes(self, boxes: tf.Tensor) -> tf.Tensor:
        """
        boxes: [N, (y1, x1, y2, x2)]
        """
        # Split
        wy1, wx1, wy2, wx2 = tf.split(np.array([0, 0, 1, 1],
                                               dtype=np.float32), 4)
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=1)
        # Clip
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)
        clipped = tf.concat([y1, x1, y2, x2], axis=1, name='clipped_boxes')
        clipped.set_shape((clipped.shape[0], 4))
        return clipped

    def _nms(self, inputs: tf.Tensor) -> tf.Tensor:
        """Non-max suppression
        boxes
        scores
        """
        boxes = inputs[0]
        scores = inputs[1]

        indices = tf.image.non_max_suppression(boxes, scores,
                                               self.proposal_count,
                                               self.nms_threshold,
                                               name='rpn_non_max_suppression')
        proposals = tf.gather(boxes, indices)
        # Pad if needed
        padding = tf.maximum(self.proposal_count - tf.shape(proposals)[0], 0)
        return tf.pad(tensor=proposals, paddings=[(0, padding), (0, 0)])

    def get_config(self):
        config = super(Proposals, self).get_config()
        config['proposal_count'] = self.proposal_count
        config['nms_threshold'] = self.nms_threshold
        config['rpn_bbox_std_dev'] = self.rpn_bbox_std_dev
        config['pre_nms_limit'] = self.pre_nms_limit
        return config

    def compute_output_shape(self, input_shape) -> tf.TensorShape:
        input_shape = tf.TensorShape(input_shape[0]).as_list()
        return tf.TensorShape((input_shape[0], self.proposal_count, 4))


############################################################
#  PyramidROIAlign Layer
############################################################

class PyramidROIAlign(Layer):
    """Implements ROI Pooling on multiple levels of the feature pyramid.

    Params:
    - pool_shape: [pool_height, pool_width] of the output pooled regions. Usually [7, 7]

    Inputs:
    - boxes: [batch, num_boxes, (y1, x1, y2, x2)] in normalized
             coordinates. Possibly padded with zeros if not enough
             boxes to fill the array.
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - feature_maps: List of feature maps from different levels of the pyramid.
                    Each is [batch, height, width, channels]

    Output:
    Pooled regions in the shape: [batch, num_boxes, pool_height, pool_width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """

    def __init__(self, pool_shape: Tuple[int, int],
                 image_meta__image_shape: Tuple[int, int], **kwargs):
        super(PyramidROIAlign, self).__init__(**kwargs)
        self.pool_shape = tuple(pool_shape)
        self.image_meta__image_shape = image_meta__image_shape

    def call(self, inputs):
        input_shape = (inputs[0].shape, inputs[1].shape, inputs[2].shape)
        # Crop boxes [batch, num_boxes, (y1, x1, y2, x2)] in normalized coords
        boxes = inputs[0]

        # Image meta
        # Holds details about the image. See compose_image_meta()
        image_meta = inputs[1]

        # Feature Maps. List of feature maps from different level of the
        # feature pyramid. Each is [batch, height, width, channels]
        feature_maps = inputs[2:]

        # Assign each ROI to a level in the pyramid based on the ROI area.
        y1, x1, y2, x2 = tf.split(boxes, 4, axis=2)
        h = y2 - y1
        w = x2 - x1
        # Use shape of first image. Images in a batch must have the same size.
        image_shape = image_meta[:, self.image_meta__image_shape[0]:
                                 self.image_meta__image_shape[1]][0]
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        image_area = tf.cast(image_shape[0] * image_shape[1], tf.float32)
        roi_level = tf.math.log(tf.sqrt(h * w) /
                                (224.0 / tf.sqrt(image_area))) / \
            tf.math.log(2.0)
        roi_level = tf.minimum(5, tf.maximum(
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))
        roi_level = tf.squeeze(roi_level, 2)

        # Loop through levels and apply ROI pooling to each. P2 to P5.
        pooled = []
        box_to_level = []
        for i, level in enumerate(range(2, 6)):
            ix = tf.compat.v1.where(tf.equal(roi_level, level))
            level_boxes = tf.gather_nd(boxes, ix)

            # Box indices for crop_and_resize.
            box_indices = tf.cast(ix[:, 0], tf.int32)

            # Keep track of which box is mapped to which level
            box_to_level.append(ix)

            # Stop gradient propogation to ROI proposals
            level_boxes = tf.stop_gradient(level_boxes)
            box_indices = tf.stop_gradient(box_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_boxes, pool_height, pool_width, channels]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], level_boxes, box_indices, self.pool_shape,
                method="bilinear"))

        # Pack pooled features into one tensor
        pooled = tf.concat(pooled, axis=0)

        # Pack box_to_level mapping into one array and add another
        # column representing the order of pooled boxes
        box_to_level = tf.concat(box_to_level, axis=0)
        box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)
        box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range],
                                 axis=1)

        # Rearrange pooled features to match the order of the original boxes
        # Sort box_to_level by batch then box index
        # TF doesn't have a way to sort by two columns, so merge them and sort.
        sorting_tensor = box_to_level[:, 0] * 100000 + box_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(
            box_to_level)[0]).indices[::-1]
        ix = tf.gather(box_to_level[:, 2], ix)
        pooled = tf.gather(pooled, ix)

        # Re-add the batch dimension
        shape = tf.concat([tf.shape(boxes)[:2], tf.shape(pooled)[1:]], axis=0)
        pooled = tf.reshape(pooled, shape)
        pooled.set_shape(self.compute_output_shape(input_shape))
        return pooled

    def get_config(self):
        config = super(PyramidROIAlign, self).get_config()
        config['pool_shape'] = self.pool_shape
        config['image_meta__image_shape'] = self.image_meta__image_shape
        return config

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1]) + self.pool_shape + \
            (input_shape[2][3], )


############################################################
#  TrimZeros Layer
############################################################

class TrimZeros(Layer):
    def __init__(self, tile_shape: Tuple, second_tile_shape: Optional[Tuple] = None,
                 **kwargs):
        super(TrimZeros, self).__init__(**kwargs)
        self.tile_shape = tile_shape
        self.second_tile_shape = second_tile_shape

    def get_config(self):
        config = super(TrimZeros, self).get_config()
        config['tile_shape'] = self.tile_shape
        config['second_tile_shape'] = self.second_tile_shape
        return config

    def call(self, inputs):
        rois = inputs[0]
        x = inputs[1]

        non_zeros = tf.map_fn(lambda x: tf.cast(tf.reduce_sum(tf.abs(x),
                                                              axis=1),
                                                tf.bool), rois,
                              fn_output_signature=tf.bool, name='trim_rois')
        non_zeros = tf.tile(tf.expand_dims(non_zeros, -1), self.tile_shape)
        if self.second_tile_shape is not None:
            non_zeros = tf.tile(tf.expand_dims(non_zeros, -1),
                                self.second_tile_shape)
        return tf.where(non_zeros, x, tf.zeros_like(x))
