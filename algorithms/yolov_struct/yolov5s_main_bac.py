import sys
import pdb
import time

import cv2
import numpy as np

sys.path.append('/root')

from person_vehicle_monitoring.tools import httpclient
from person_vehicle_monitoring.algorithms.yolov_struct import YOLOV5S_PARAM
from person_vehicle_monitoring.algorithms.yolov_struct.cpp_libs.detect.cpu_detect import cpu_detect
from person_vehicle_monitoring.algorithms.yolov_struct.cpp_libs.nms.cpu_nms import cpu_nms


class Yolov5TRT(object):

    def __init__(self):
        self.input_size = 640
        self.conf_thresh = 0.4
        self.iou_thresh = 0.5

        self.anchors = YOLOV5S_PARAM['anchors']
        self.num_layers = len(self.anchors)
        self.num_classes = YOLOV5S_PARAM['nc']

    def inference(self, img, cli):
        img0, ratio, pad = self.pre_process(img)
        img1 = img0[:, :, ::-1].transpose(2, 0, 1) / 255.0
        img1 = img1[np.newaxis, :]
        img1 = np.array(img1, dtype=np.float32, order='C')

        input_name = httpclient.InferInput(name="images", shape=[1, 3, 640, 640], datatype="FP32")
        input_name.set_data_from_numpy(img1)
        outputs = cli.infer('yolov5s', [input_name])
        resp = outputs.get_response()['outputs']
        detects = [np.array(resp[0]['data']).astype('float32'), np.array(resp[1]['data']).astype('float32'),
                   np.array(resp[2]['data']).astype('float32')]

        boxes = cpu_detect(detects, self.anchors, self.input_size, self.num_classes, self.conf_thresh)
        nms_boxes = cpu_nms(boxes, self.num_classes, self.iou_thresh)

        raw_boxes = self.post_process(nms_boxes, ratio, pad)

        return raw_boxes

    def pre_process(self, img, color=(114, 114, 114)):
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = [self.input_size, self.input_size]
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

    def post_process(self, boxes, ratio, pad):
        raw_boxes = boxes.copy()
        raw_boxes[:, 0] = (boxes[:, 0] - pad[0]) / ratio[0]
        raw_boxes[:, 1] = (boxes[:, 1] - pad[1]) / ratio[1]
        raw_boxes[:, 2] = (boxes[:, 2] - pad[0]) / ratio[0]
        raw_boxes[:, 3] = (boxes[:, 3] - pad[1]) / ratio[1]
        return raw_boxes


if __name__ == "__main__":
    triton_client = httpclient.InferenceServerClient(url='10.20.5.9:9911')
    yolov5 = Yolov5TRT()
    img = cv2.imread("/models/xu.jpg")
    t1 = time.time()
    for i in range(100):
        dets = yolov5.inference(img, triton_client)
        t2 = time.time()
        print('infer={}ms'.format((t2 - t1) * 1000 / 100))
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for det in dets:
            x1 = int(det[0])
            y1 = int(det[1])
            x2 = int(det[2])
            y2 = int(det[3])
            conf = det[4]
            cls = int(det[5])

            cv2.rectangle(img, (x1, y1), (x2, y2), colors[cls], 1)
        cv2.imwrite("/models/xu_test.jpg", img)
