import cv2
import numpy as np

import tritonhttpclient as httpclient

from person_vehicle_monitoring.algorithms.vehicle_plate_color import COLOR_MAP


def vehicle_plate_color_input(image, cli):
    image = cv2.resize(image, (224, 224))
    image = image[np.newaxis, :] / 255.
    image = image.astype("float32")
    input = httpclient.InferInput('Placeholder', [1, 224, 224, 3], "FP32")
    input.set_data_from_numpy(image)
    async_resp = cli.infer(model_name='plate_color', inputs=[input])
    return async_resp


def vehicle_plate_color_output(resp):
    resp = resp.as_numpy('logit_output_pool_and_classify')
    ocr_res = COLOR_MAP[np.argmax(resp)]
    return ocr_res


if __name__ == '__main__':
    triton_client = httpclient.InferenceServerClient(url='10.20.5.9:9911')
    img = cv2.imread('/models/3-3.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    user_data = vehicle_plate_color_input(img, triton_client)
    user_data = user_data.get_result()
    result = vehicle_plate_color_output(user_data)
    print(result)

