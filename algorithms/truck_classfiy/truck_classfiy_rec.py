import os
import time

import cv2
import sys
from pathlib import Path

import pickle
import numpy as np
import torch
import tritonhttpclient as httpclient

from person_vehicle_monitoring.algorithms.plate_location_rec import PriorBox, decode, decode_landm
from person_vehicle_monitoring.algorithms.truck_classfiy import TRUCK_LABEL

sys.path.append('/root')

from person_vehicle_monitoring.algorithms.mvehicle_multi import *
from person_vehicle_monitoring.tools.httpclient import InferInput



class TruckClassfiy(object):
    def __init__(self, cli):
        self.cli = cli

    def truck_classfiy_input(self, img):
        image = cv2.resize(img, (224, 224))
        image = image[np.newaxis, :] / 255.
        image = image.astype("float32")
        image = np.transpose(image, (0, 3, 1, 2))
        input = httpclient.InferInput('conv1', [1, 3, 224, 224], "FP32")

        input.set_data_from_numpy(image)
        resp = self.cli.infer(model_name='truck_classfiy', inputs=[input])
        return resp

    def truck_classfiy_output(self, resp):
        resp = resp.as_numpy('fc')
        ocr_res = TRUCK_LABEL[np.argmax(resp)]

        return ocr_res

    async def truck_classfiy_infer(self, img):
        resp = self.truck_classfiy_input(img)
        result = self.truck_classfiy_output(resp)
        return result



if __name__ == '__main__':

    triton_client = httpclient.InferenceServerClient(url='10.20.5.9:9911')
    mv_class = TruckClassfiy(triton_client)

    img_path = "/home/zrl/workplace/PyProject/person_vehicle_monitoring/test_img/160568409138081_0.jpg"
    image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #  BGR
    while True:
        s = time.time()
        resp = mv_class.truck_classfiy_input(image)
        print(time.time() -s)
        result = mv_class.truck_classfiy_output(resp)

        print(result)
