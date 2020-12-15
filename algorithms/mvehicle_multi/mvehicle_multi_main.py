import os

import cv2
import sys
from pathlib import Path

import pickle
import numpy as np
import tritonhttpclient as httpclient

sys.path.append('/root')

from person_vehicle_monitoring.algorithms.mvehicle_multi import *
from person_vehicle_monitoring.tools.httpclient import InferInput


class RCF_model(object):

    def __init__(self, cli):
        self.cli = cli

    def detect_mvehicle(self, proba):

        idxs = np.argsort(proba)[::-1]
        mtype_all = []
        mcolor_all = []
        mtowards_all = []
        for label_index in idxs:
            if CLASSES_LABEL[label_index] in TYPE_ENUM:
                mtype_all.append(CLASSES_LABEL[label_index])
            elif CLASSES_LABEL[label_index] in COLOR_ENUM:
                mcolor_all.append(CLASSES_LABEL[label_index])
            elif CLASSES_LABEL[label_index] in TOWARDS_ENUM:
                mtowards_all.append(CLASSES_LABEL[label_index])
        mtype = mtype_all[0]
        mtowards = mtowards_all[0]
        if mtype in TYPE_MAP_ENUM.keys():
            mtype = TYPE_MAP_ENUM[mtype_all[0]]
        if mtowards in TOWARDS_MAP_ENUM.keys():
            mtowards = TOWARDS_MAP_ENUM[mtowards]

        result = {
            "mvehicle_type": mtype,
            "mvehicle_color": mcolor_all[0],
            "mvehicle_towards": mtowards,
        }

        return result

    def mvehicle_input(self, image):
        image = cv2.resize(image, (144, 144))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)
        image = np.transpose(image, (0, 3, 1, 2))
        image = np.array(image, dtype=np.float32, order='C')

        input_name = InferInput(name="input", shape=[1, 3, 144, 144], datatype="FP32")
        input_name.set_data_from_numpy(image)
        resp = self.cli.infer('mvehicle-multi', [input_name])
        return resp

    def mvehicle_output(self, resp):
        prob = resp.get_response()["outputs"][0]["data"]
        result = self.detect_mvehicle(prob)
        return result

    async def mvehicle_infer(self, image):
        resp = self.mvehicle_input(image)
        result = self.mvehicle_output(resp)
        return result


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='10.20.5.9:9911')
    mv_class = RCF_model(triton_client)

    img_path = "/home/zrl/workplace/PyProject/person_vehicle_monitoring/test_img/7.jpg"
    image = cv2.imread(img_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #  BGR
    resp = mv_class.mvehicle_input(image)
    result = mv_class.mvehicle_output(resp)
    print(result)
