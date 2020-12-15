import sys
import time
import asyncio
from pathlib import Path

import cv2
from utils import *
import copy
import numpy as np
import nest_asyncio



sys.path.append('/app/workdir')
sys.path.append('/app')
from utils import *

from person_vehicle_monitoring.core import CLIENT, CLIENT_DECORATE
from person_vehicle_monitoring.tools.utils import cut_image, cut_img
from person_vehicle_monitoring.core.base import filter_vehicle
from person_vehicle_monitoring.algorithms.seatbelt.seatbelt_main import Seatbelt
from person_vehicle_monitoring.algorithms.plate_rec.plate_num_rec import Plate_model
#from person_vehicle_monitoring.algorithms.yolov_struct.yolov5s_main import Yolov5TRT
from person_vehicle_monitoring.algorithms.mvehicle_multi.mvehicle_multi_main import RCF_model
from person_vehicle_monitoring.algorithms.decoration_objs.decoration_objs_main import YoloDecorate
from person_vehicle_monitoring.algorithms.distraction_driver.distraction_driver_main import DistractionDriver
from person_vehicle_monitoring.algorithms.truck_classfiy.truck_classfiy_rec import TruckClassfiy


from person_vehicle_monitoring.tools import httpclient

decorate = YoloDecorate(CLIENT_DECORATE)
# LABEL_BIN_PATH = Path().cwd().parent.joinpath('algorithms/mvehicle_multi')
LABEL_BIN_PATH = Path('/app/workdir/person_vehicle_monitoring/algorithms/mvehicle_multi')
plate = Plate_model(CLIENT)
mv_class = RCF_model(CLIENT)
truck_classfiy = TruckClassfiy
distraction_driver = DistractionDriver(CLIENT)
seatbelt = Seatbelt(CLIENT)

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

nest_asyncio.apply()



def run(data):
    s = time.time()
    direction = 'head'
    seatbelt_result, distraction_result = 'have', 'no'
    equipment_id = data.get('sbbh', "")
    print(f'get equipment_id {equipment_id}')
    structures = data['structure']
    structures = eval(structures)
    img = data['img']
    del data['img']


    flag = False
    truck_type = ''
    short_box =img

    #  BGR
    vehicle_plate_task = asyncio.ensure_future(plate.plate_reg(short_box))
    # Rgb
    decorate_img = cv2.cvtColor(short_box, cv2.COLOR_BGR2RGB)
    decorate_task = asyncio.ensure_future(decorate.decorate_infer(decorate_img))

    #  BGR
    mvehicle_task = asyncio.ensure_future(mv_class.mvehicle_infer(short_box))
    loop.run_until_complete(asyncio.wait([vehicle_plate_task, mvehicle_task, decorate_task]))
    mvehicle_info = mvehicle_task.result()

    decorate_info = decorate_task.result()
    plate_color, vehicle_plate = vehicle_plate_task.result()
    windows = decorate_info.get('window', [])
    persons = decorate_info.get('persons', [])
    vehicle_type = mvehicle_info.get('mvehicle_type')

    if (direction != 'back' and len(windows) > 0) or (direction != 'back' and len(persons) > 0):
        print(f'begin seatbelt_task and distraction_task !')
        if len(windows) > 0:
            window = sorted(windows, key=lambda x:x[4])[-1]
            window_img = cut_img(decorate_img, [0,0,window[:4]])
            h, w, c = window_img.shape
            person_img = window_img[:, w // 2:w]
        else:
            person = sorted(persons, key=lambda x:x[4][2])[-1]
            person_img = cut_img(decorate_img, [0,0,person[:4]])

        tasks = []
        if vehicle_type in ["HeavyTruck", "LargeTruck", "MiddleTruck"]:
            flag = True
            truck_classfiy_task = asyncio.ensure_future(truck_classfiy.truck_classfiy_infer(short_box))
            tasks.append(truck_classfiy_task)



        seatbelt_task = asyncio.ensure_future(seatbelt.seatbelt_infer(person_img))
        distraction_task = asyncio.ensure_future(distraction_driver.distraction_infer(person_img))
        tasks.append(seatbelt_task)
        tasks.append(distraction_task)
        loop.run_until_complete(asyncio.wait(tasks))
        seatbelt_result = seatbelt_task.result()
        distraction_result = distraction_task.result()
        if flag:
            truck_type = truck_classfiy_task.result()


    new_data = copy.deepcopy(data)
    new_data['car_color'] = mvehicle_info.get('mvehicle_color')
    car_img_id = uploadfile(short_box)
    new_data["shortCutFile"] = car_img_id.decode()
    new_data['car_plate_no'] = vehicle_plate
    new_data['car_plate_color'] = plate_color
    new_data['truck_type'] = truck_type
    new_data['visor'] = decorate_info.get('visor', [])
    new_data['accessories'] = decorate_info.get('accessories', [])
    new_data['saftey'] = [1] if seatbelt_result == 'not_have' else []
    new_data['carType'] = mvehicle_info.get('mvehicle_type')
    new_data['distractedDriving'] = 'yes' if distraction_result == 'calling' else 'no'


#  TODO  ensure yolov5s output vehicle type
#  TODO  ensure
if __name__ == '__main__':
    img = cv2.imread('/app/workdir/person_vehicle_monitoring/test_img/xu.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    while True:
        s = time.time()
        info = run({'img': img, 'equipment_id': 1})
        print(time.time() - s)
        print(info)