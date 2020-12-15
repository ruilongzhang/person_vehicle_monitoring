import time

import cv2

from person_vehicle_monitoring.algorithms.decoration_objs.decoration_objs_main import decoration_objs_output, \
    decoration_objs_input
from person_vehicle_monitoring.algorithms.distraction_driver.distraction_driver_main import distraction_output, \
    distraction_input
from person_vehicle_monitoring.algorithms.mvehicle_multi.mvehicle_multi_main import mvehicle_input, mvehicle_output
from person_vehicle_monitoring.algorithms.seatbelt.seatbelt_main import seatbelt_input, seatbelt_output
from person_vehicle_monitoring.algorithms.vehicle_plate_color.vehicle_plate_color_main import \
    vehicle_plate_color_output, vehicle_plate_color_input
from person_vehicle_monitoring.algorithms.yolov_struct.yolov_struct_main import detect
from person_vehicle_monitoring.config import TRITON_HTTP_SERVER_URL
from person_vehicle_monitoring.core.base import filter_vehicle
from person_vehicle_monitoring.tools.utils import retry
from tritonclient.utils import InferenceServerException
import tritonclient.grpc as grpcclient

try:
    triton_client = grpcclient.InferenceServerClient(url=TRITON_HTTP_SERVER_URL, verbose=False)
except Exception as e:
    print("context creation failed: " + str(e))


#  TODO  依据每个算法推理时间 调整判断位置
def mvehicle_decoration_objs(img, results, info, next_input):
    for k, v in results:
        event_result = event_loop(v)
        if k == 'mvehicle':
            mvehicle_result = mvehicle_output(event_result)
            info.update(mvehicle=mvehicle_result['mvehicle'])
        elif k == 'decoration_objs':
            decoration_objs_result = decoration_objs_output(event_result)
            info.update(decoration_objs=decoration_objs_result['decoration_objs'])
            plate_location = decoration_objs_result['plate_location']
            if plate_location:
                resp3 = vehicle_plate_color_input(img, v)
                next_input.update(vehicle_plate_color=resp3)
            windows = decoration_objs_result['windows']
            if windows:
                resp_1 = seatbelt_input(img, v)
                resp_2 = distraction_input(img, v)
                next_input.update(distraction=resp_2, seatbelt=resp_1)
    return next_input, info


def event_loop(user_data, time_sleep=0.01, time_out=10):
    while (len(user_data) == 0) and time_out > 0:
        time_out = time_out - 1
        time.sleep(time_sleep)
    if type(user_data[0]) == InferenceServerException:
        return []
    return user_data


#  TODO  依据每个算法推理时间 调整判断位置
def distraction_seatbelt_vehicle_plate_color(results, output):
    for k, v in results:
        event_result = event_loop(v)
        if k == 'seatbelt':
            seatbelt_result = seatbelt_output(event_result)
            output.update(seatbelt=seatbelt_result)
        elif k == 'distraction':
            resp_4 = distraction_output(v)
            output.update(distraction=resp_4)
        elif k == 'vehicle_plate_color':
            resp_4 = vehicle_plate_color_output(v)
            output.update(vehicle_plate_color=resp_4)
    return output


def cut_img(img, structure):
    x1, y1, x2, y2 = structure
    short_box = img[y1:y2, x1:x2]
    return short_box


def run(data):
    img = data['img']
    equipment_id = data['equipment_id']
    structures = detect(img, triton_client)
    filter_structures = filter_vehicle(structures, equipment_id, max_area=True)
    for structure in filter_structures:
        short_box = cut_img(img, structure)
        #  TODO  截图 目前支持 bachsize = 1
        resp_1 = mvehicle_input(short_box, triton_client)
        resp_2 = decoration_objs_input(short_box, triton_client)
        results = {'mvehicle': resp_1, 'decoration_objs': resp_2}
        info, next_input = {}, {}
        output, next_input = mvehicle_decoration_objs(img, results, info, next_input)
        algorithms_result = distraction_seatbelt_vehicle_plate_color(next_input, output)
        return algorithms_result


if __name__ == '__main__':
    img = cv2.imread('')
    run({'img': img, 'equipment_id': 1})
