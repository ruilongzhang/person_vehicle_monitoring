
import cv2
import sys
import time

import numpy as np

import tritonhttpclient as httpclient
sys.path.append('/root')

from person_vehicle_monitoring.tools.httpclient import InferInput
from person_vehicle_monitoring.algorithms.plate_location_rec import cpu_nms


class PlateLocation(object):
    def __init__(self, cli):
        self.cli = cli
        self.score_thresh = 0.02
        self.nms_iou_thresh = 0.4

    def ResizeImg(self, img):
        shape = img.shape[:2]
        new_shape = (320, 320)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, (dw, dh)

    def PreProcess(self, img):
        img0, ratio, pad = self.ResizeImg(img)

        img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1[:, :, 0] = img1[:, :, 0] - 104
        img1[:, :, 1] = img1[:, :, 1] - 117
        img1[:, :, 2] = img1[:, :, 2] - 123

        img2 = img1.transpose([2, 0, 1])
        img3 = np.expand_dims(img2.astype(np.float32), axis=0)
        return img3, ratio, pad

    def PostProcess(self, raw_boxes, raw_points, ratio, pad):
        (dw, dh) = pad

        boxes = raw_boxes.copy()
        points = raw_points.copy()

        boxes[:, 0] = (boxes[:, 0] - dw) / ratio
        boxes[:, 1] = (boxes[:, 1] - dh) / ratio
        boxes[:, 2] = (boxes[:, 2] - dw) / ratio
        boxes[:, 3] = (boxes[:, 3] - dh) / ratio

        points[:, 0] = (points[:, 0] - dw) / ratio
        points[:, 1] = (points[:, 1] - dh) / ratio
        points[:, 2] = (points[:, 2] - dw) / ratio
        points[:, 3] = (points[:, 3] - dh) / ratio
        points[:, 4] = (points[:, 4] - dw) / ratio
        points[:, 5] = (points[:, 5] - dh) / ratio
        points[:, 6] = (points[:, 6] - dw) / ratio
        points[:, 7] = (points[:, 7] - dh) / ratio

        return boxes, points


    def plate_location_input(self, img):
        x, ratio, pad = self.PreProcess(img)
        image = np.array(x, dtype=np.float32, order='C')

        input_name = InferInput(name="input0", shape=[1, 3, 320, 320], datatype="FP32")
        input_name.set_data_from_numpy(image)
        resp = self.cli.infer('plate_positon', [input_name])
        return resp, ratio, pad



    def plate_location_output(self, feature_maps, ratio, pad):
        plate_box = []

        feature_bbox_regression = feature_maps.as_numpy('output0')[0]  # [1, 5875, 4]
        feature_landmarks = feature_maps.as_numpy('529')[0]  # [1, 5875, 8]
        feature_scores = feature_maps.as_numpy('530')[0]  # [1, 5875, 2]

        steps = (8, 16, 32, 64)
        strides = (40, 20, 10, 5)  # 320/step
        anchor_sizes = ((10, 16, 24), (32, 48), (64, 96), (128, 192, 256))
        start_idxs = (0, 4800, 5600, 5800)
        stop_idxs = (4800, 5600, 5800, 5875)

        total_boxes = []
        total_points = []
        for i in range(4):
            step = steps[i]
            stride = strides[i]
            anchors = anchor_sizes[i]

            idx1 = start_idxs[i]
            idx2 = stop_idxs[i]
            bbox_regression = feature_bbox_regression[idx1:idx2, :].reshape(stride, stride, -1, 4)
            scores = feature_scores[idx1:idx2, :].reshape(stride, stride, -1, 2)
            landmarks = feature_landmarks[idx1:idx2, :].reshape(stride, stride, -1, 8)

            x = np.arange(stride)
            y = np.arange(stride)
            xv, yv = np.meshgrid(x, y)
            anchor_grid = np.stack((xv, yv), 2).reshape((stride, stride, 2))
            anchor_cx = (anchor_grid[:, :, 0] + 0.5) * step
            anchor_cy = (anchor_grid[:, :, 1] + 0.5) * step

            for k, anchor_sz in enumerate(anchors):
                valid_boxes = scores[:, :, k, 1] > self.score_thresh
                valid_scores = scores[:, :, k, 1][valid_boxes]

                cx = anchor_cx + 0.1 * bbox_regression[:, :, k, 0] * anchor_sz
                cy = anchor_cy + 0.1 * bbox_regression[:, :, k, 1] * anchor_sz
                sx = anchor_sz * np.exp(0.2 * bbox_regression[:, :, k, 2])
                sy = anchor_sz * np.exp(0.2 * bbox_regression[:, :, k, 3])

                # boxes
                x1 = cx - 0.5 * sx
                y1 = cy - 0.5 * sy
                x2 = cx + 0.5 * sx
                y2 = cy + 0.5 * sy

                x1 = x1[valid_boxes]
                y1 = y1[valid_boxes]
                x2 = x2[valid_boxes]
                y2 = y2[valid_boxes]

                boxes = np.stack([x1, y1, x2, y2, valid_scores], axis=1)

                # points
                pt1_x = anchor_cx + 0.1 * landmarks[:, :, k, 0] * anchor_sz
                pt1_y = anchor_cy + 0.1 * landmarks[:, :, k, 1] * anchor_sz
                pt2_x = anchor_cx + 0.1 * landmarks[:, :, k, 2] * anchor_sz
                pt2_y = anchor_cy + 0.1 * landmarks[:, :, k, 3] * anchor_sz
                pt3_x = anchor_cx + 0.1 * landmarks[:, :, k, 4] * anchor_sz
                pt3_y = anchor_cy + 0.1 * landmarks[:, :, k, 5] * anchor_sz
                pt4_x = anchor_cx + 0.1 * landmarks[:, :, k, 6] * anchor_sz
                pt4_y = anchor_cy + 0.1 * landmarks[:, :, k, 7] * anchor_sz

                pt1_x = pt1_x[valid_boxes]
                pt1_y = pt1_y[valid_boxes]
                pt2_x = pt2_x[valid_boxes]
                pt2_y = pt2_y[valid_boxes]
                pt3_x = pt3_x[valid_boxes]
                pt3_y = pt3_y[valid_boxes]
                pt4_x = pt4_x[valid_boxes]
                pt4_y = pt4_y[valid_boxes]

                points = np.stack([pt1_x, pt1_y, pt2_x, pt2_y, pt3_x, pt3_y, pt4_x, pt4_y], axis=1)
                if boxes.shape[0] > 0:
                    total_boxes.append(boxes)
                    total_points.append(points)
        if len(total_boxes) <= 0:
            return plate_box
        total_boxes = np.concatenate(total_boxes, axis=0).astype(np.float32)
        total_points = np.concatenate(total_points, axis=0).astype(np.float32)

        keep = cpu_nms.cpu_nms(total_boxes, self.nms_iou_thresh)
        nms_boxes = total_boxes[keep, :]
        nms_points = total_points[keep, :]
        boxes, points = self.PostProcess(nms_boxes, nms_points, ratio, pad)
        if len(boxes) > 0:
            plate_box = sorted(boxes, key=lambda x:x[4])[-1]

        return plate_box


    def plate_location_infer(self, img):
        feature_maps, ratio, pad = self.plate_location_input(img)
        nms_boxes = self.plate_location_output(feature_maps, ratio, pad)
        return nms_boxes



if __name__ == '__main__':

    triton_client = httpclient.InferenceServerClient(url='10.20.5.9:9911')
    mv_class = PlateLocation(triton_client)

    img_path = "/home/zrl/workplace/PyProject/person_vehicle_monitoring/test_img/1.jpg"
    img = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #  BGR
    while True:
        s = time.time()

        boxes =  mv_class.plate_location_infer(img)


        x1, y1, x2, y2, s = boxes
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, "{:.3f}".format(s), (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imwrite("blackcar1_result.jpg", img)
        print('ok')



