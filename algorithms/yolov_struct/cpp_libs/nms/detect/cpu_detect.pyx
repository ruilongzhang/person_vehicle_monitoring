import math

import numpy as np
cimport numpy as np

def cpu_detect(list dets, list anchors, int input_size, int num_classes, float conf_thresh):

    cdef int num_layers = len(anchors)
    cdef int num_outputs = num_classes + 5 # cx cy w h conf prob1 prob2 prob3 ...


    cdef np.int num_anchors, nx, ny

    cdef np.ndarray xv, yv, grid, anchor_grid
    cdef np.ndarray y, y_reshape
    cdef np.ndarray det, boxes, nms_boxes, output_boxes, probs, class_ids

    cdef list z = []

    cdef dict sqrt_table = {400:20, 1600:40, 6400:80} 

    for det, anchor in zip(dets, anchors):
        if not isinstance(anchor, np.ndarray):
            anchor = np.array(anchor)

        num_anchors = anchor.shape[0] // 2

        #nx = int(math.sqrt(det.shape[0] / num_anchors / num_outputs))
        nx = sqrt_table[int(det.shape[0] / num_anchors / num_outputs)]
        ny = nx

        x = det.reshape((num_anchors, ny, nx, num_outputs))

        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))
        grid = np.stack((yv, xv), 2).reshape((1, ny, nx, 2))
        anchor_grid = anchor.reshape((-1, 1, 1, 2))

        #y = 1.0 / (1 + np.exp(-x)) # sigmoid move to onnx
        y = x
        y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * (input_size / nx)
        y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * (anchor_grid)

        y_reshape = y.reshape((-1, num_outputs))

        z.append(y_reshape)

    boxes = np.concatenate(z, axis=0)

    keep = np.where(boxes[:, 4] > conf_thresh)
    nms_boxes = boxes[keep]

    class_ids = nms_boxes[:, 5:].argmax(axis=1)

    output_boxes = nms_boxes.copy()
    output_boxes[:, 0] = nms_boxes[:, 0] - 0.5 * nms_boxes[:, 2]
    output_boxes[:, 1] = nms_boxes[:, 1] - 0.5 * nms_boxes[:, 3]
    output_boxes[:, 2] = nms_boxes[:, 0] + 0.5 * nms_boxes[:, 2]
    output_boxes[:, 3] = nms_boxes[:, 1] + 0.5 * nms_boxes[:, 3]
    output_boxes[:, 5] = class_ids.astype(boxes.dtype)

    return output_boxes # cx, cy, w, h, conf, class_id
