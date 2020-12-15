import cv2
import sys

import numpy as np
import tritonhttpclient as httpclient

sys.path.append('/root')

from person_vehicle_monitoring.algorithms.distraction_driver import CLASS_LABEL


class DistractionDriver(object):
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


    def distraction_input(self, image):
        pil_img = cv2.resize(image, (256, 256))
        pil_img = pil_img[16:240, 16:240]
        np_img = np.array(pil_img, dtype=np.float32) / 255.0

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        np_img[:, :, 0] = (np_img[:, :, 0] - mean[0]) / std[0]
        np_img[:, :, 1] = (np_img[:, :, 1] - mean[1]) / std[1]
        np_img[:, :, 2] = (np_img[:, :, 2] - mean[2]) / std[2]
        np_img = np_img.transpose([2, 0, 1])
        input_image = np.array(np_img, dtype=np.float32, order='C')
        input_image = np.expand_dims(input_image, axis=0)
        input_name = httpclient.InferInput(name="input_0", shape=[1, 3, 224, 224], datatype="FP32")
        input_name.set_data_from_numpy(input_image)
        output = self.cli .infer('distraction', inputs=[input_name])
        return output


    def distraction_output(self, resp):
        prob = self.softmax(resp)
        max_idx = np.argmax(prob)
        predict = CLASS_LABEL['distraction'][max_idx]
        return predict


    async def distraction_infer(self, img):
        resp = self.distraction_input(img)
        resp = resp.as_numpy('output_0')
        result = self.distraction_output(resp)
        return result


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='10.20.5.9:9911')
    img = cv2.imread('/models/2-2.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    import time

    s = time.time()
    resp = distraction_input(img, triton_client)
    resp = resp.as_numpy('output_0')
    result = distraction_output(resp)

    print(time.time() - s)
    print(result)
