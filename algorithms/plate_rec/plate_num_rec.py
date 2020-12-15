# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
from tritonhttpclient import InferenceServerClient, InferInput

from person_vehicle_monitoring.algorithms.plate_location_rec.plate_location import PlateLocation
from person_vehicle_monitoring.algorithms.vehicle_plate_color.vehicle_plate_color_main import vehicle_plate_color_input, \
    vehicle_plate_color_output
from person_vehicle_monitoring.config import TRITON_HTTP_SERVER_URL
from person_vehicle_monitoring.tools import httpclient
from person_vehicle_monitoring.tools.utils import cut_img


class LPR():
    res: object

    def __init__(self, cli):
        self.res = cli

        self.chars = ("京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "皖", "闽", "赣", "鲁", "豫",
                      "鄂", "湘", "粤", "桂",
                      "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "0", "1", "2", "3", "4", "5",
                      "6",
                      "7", "8", "9", "A",
                      "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S",
                      "T",
                      "U", "V", "W", "X",
                      "Y", "Z", "港", "学", "使", "警", "澳", "挂", "军", "北", "南", "广", "沈", "兰", "成", "济",
                      "海",
                      "民", "航", "空"
                      )

    def decode_ctc(self, y_pred):
        results = ""
        confidence = 0.0
        y_pred = y_pred.T
        table_pred = y_pred
        res = table_pred.argmax(axis=1)
        for i, one in enumerate(res):
            if one < len(self.chars) and (i == 0 or (one != res[i - 1])):
                results += self.chars[one]
                confidence += table_pred[i][one]
        confidence /= len(results)
        return results, confidence

    def segmentation_free_recognition(self, src):
        temp = cv2.resize(src, (160, 40))
        temp = temp.transpose(1, 0, 2)
        blob = (1 / 255.0) * temp.astype(np.float32).transpose([2, 0, 1])
        input_name = InferInput(name="data", shape=[3, 160, 40], datatype="FP32")
        input_name.set_data_from_numpy(blob)
        resp = self.res.infer('plate-recognition', [input_name])
        output = np.array(resp.get_response()["outputs"][0]["data"]).reshape(84, 20, 1)
        y_pred = output[:, 2:, :]
        y_pred = np.squeeze(y_pred)
        return self.decode_ctc(y_pred)

    def plate_recognition(self, image):
        res, confidence = self.segmentation_free_recognition(image)
        return res


class Plate_model(object):
    def __init__(self, cli):
        self.PR = LPR(cli)
        self.plate_location = PlateLocation(cli)

    def HyperLPR_plate_recognition(self, Input_BGR):
        plate_img, plate_no = np.array([]), ''
        plate_box = self.plate_location.plate_location_infer(Input_BGR)
        if len(plate_box) > 0:
            plate_img = cut_img(Input_BGR, [0,0, plate_box[:4]])
            plate_no = self.PR.plate_recognition(plate_img)

        return plate_img, plate_no

    async def plate_reg(self, image):
        plate_color, vehicle_plate, plate_no = 'blue', [], ''

        plate_img, plate_no = self.HyperLPR_plate_recognition(image)
        if plate_img.size > 0:
            plate_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            resp = vehicle_plate_color_input(plate_img, self.PR.res)
            plate_color = vehicle_plate_color_output(resp)
        return plate_color, plate_no


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='10.20.5.9:9911')
    plate = Plate_model(triton_client)
    img_list = ['./2123.jpg', './1019.jpg', './74.jpg']
    # for jpg in img_list:
    #    img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
    #    res = plate.plate_reg(img)
    #    print(jpg, res)
    for jpg in range(100):
        jpg = "/home/zrl/workplace/PyProject/person_vehicle_monitoring/test_img/7.jpg"
        img = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)
        res = plate.plate_reg(img)
        print(jpg, res)
