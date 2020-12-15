import sys

import cv2
import numpy as np

from person_vehicle_monitoring.algorithms.seatbelt import SEATBELT_LABEL

sys.path.append('/root')

from person_vehicle_monitoring.tools import httpclient


class Seatbelt(object):

    def __init__(self, cli):
        self.cli = cli

    def softmax(self, x):
        x_row_max = x.max(axis=-1)
        x_row_max = x_row_max.reshape(list(x.shape)[:-1] + [1])
        x = x - x_row_max
        x_exp = np.exp(x)
        x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1] + [1])
        softmax = x_exp / x_exp_row_sum
        return softmax

    def seatbelt_input(self, image):
        pil_img = cv2.resize(image, (256, 256))
        pil_img = pil_img[16:240, 16:240]  # 裁剪【y1,y2：x1,x2】
        np_img = np.array(pil_img, dtype=np.float32) / 255.0
        np_img = np_img.transpose([2, 0, 1])
        input_image = np.array(np_img, dtype=np.float32, order='C')
        input_image = np.expand_dims(input_image, axis=0)
        input_name = httpclient.InferInput(name="input_0", shape=[1, 3, 224, 224], datatype="FP32")
        input_name.set_data_from_numpy(input_image)
        output = self.cli.infer('seatbelt', inputs=[input_name])
        return output

    def seatbelt_output(self, resp):
        resp = resp.as_numpy('output_0')
        prob = self.softmax(resp)
        max_idx = np.argmax(prob)
        predict = SEATBELT_LABEL['seatbelt'][max_idx]
        return predict

    async def seatbelt_infer(self, img):
        resp = self.seatbelt_input(img)
        result = self.seatbelt_output(resp)
        return result


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='10.20.5.9:9911')
    img = cv2.imread('/models/1-1.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resp = seatbelt_input(img, triton_client)
    resp = resp.as_numpy('output_0')
    result = seatbelt_output(resp)
    print(result)
