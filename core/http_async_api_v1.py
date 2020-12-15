import sys
import time
import asyncio
from pathlib import Path

import cv2
from utils import *
import numpy as np

sys.path.append('/app/workdir')
sys.path.append('/app')
from utils import *

from person_vehicle_monitoring.core import CLIENT, CLIENT_DECORATE
from person_vehicle_monitoring.tools.utils import cut_image, cut_img
from person_vehicle_monitoring.core.base import filter_vehicle
from person_vehicle_monitoring.algorithms.seatbelt.seatbelt_main import Seatbelt
from person_vehicle_monitoring.algorithms.plate_rec.plate_num_rec import Plate_model
from person_vehicle_monitoring.algorithms.yolov_struct.yolov5s_main import Yolov5TRT
from person_vehicle_monitoring.algorithms.mvehicle_multi.mvehicle_multi_main import RCF_model
from person_vehicle_monitoring.algorithms.decoration_objs.decoration_objs_main import YoloDecorate
from person_vehicle_monitoring.algorithms.distraction_driver.distraction_driver_main import DistractionDriver


from person_vehicle_monitoring.tools import httpclient

decorate = YoloDecorate(CLIENT_DECORATE)
# LABEL_BIN_PATH = Path().cwd().parent.joinpath('algorithms/mvehicle_multi')
LABEL_BIN_PATH = Path('/app/workdir/person_vehicle_monitoring/algorithms/mvehicle_multi')
plate = Plate_model(CLIENT)
mv_class = RCF_model(CLIENT, LABEL_BIN_PATH)
distraction_driver = DistractionDriver(CLIENT)
seatbelt = Seatbelt(CLIENT)
loop = asyncio.get_event_loop()

# person_img = cv2.imread('/app/workdir/person_vehicle_monitoring/test_img/ren.jpg')
# person_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)

# jpg = "/app/workdir/person_vehicle_monitoring/test_img/7.jpg"
# car_box = cv2.imdecode(np.fromfile(jpg, dtype=np.uint8), -1)


def run(data):
    from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED




    structures = data['structure']
    structures = eval(structures)

    with ThreadPoolExecutor(max_workers=len(structures)) as t:
        all_tasks = [t.submit(decorate, structure)for structure in structures]
        wait(fs, timeout=None, return_when=ALL_COMPLETED)

    img = data['img']
    del data['img'

    del data['structure']
    # TODO input color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # TODO debug=True
    filter_structures, direction = filter_vehicle(structures, equipment_id=None)
    event_list = []
    for structure in filter_structures:
        short_box = cut_image(img, structure[2])
        # TODO debug=True
        #short_box = cv2.imread(f'/app/workdir/person_vehicle_monitoring/test_img/4.jpg')
        
        
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
        #direction = mvehicle_info.get('mvehicle_towards', 'back')
        if (direction != 'back' and len(windows) > 0) or (direction != 'back' and len(persons) > 0):
#             short_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if len(windows) > 0:
                
                window = sorted(windows, key=lambda x:x[4])[-1]
                window_img = cut_img(decorate_img, [0,0,window[:4]])
                window_img = cv2.cvtColor(window_img, cv2.COLOR_RGB2BGR)
                h, w, c = window_img.shape
                person_img = window_img[:, w // 2:w]
            else:
                person = sorted(persons, key=lambda x:x[4][2])[-1]
                person_img = cut_img(decorate_img, [0,0,window[:4]])
            seatbelt_task = asyncio.ensure_future(seatbelt.seatbelt_infer(person_img))
            distraction_task = asyncio.ensure_future(distraction_driver.distraction_infer(person_img))
            loop.run_until_complete(asyncio.wait([seatbelt_task, distraction_task]))
            seatbelt_result = seatbelt_task.result()
            distraction_result = distraction_task.result()

        x, y, w, h = structure[2]
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
#         short_box_img = cv2.cvtColor(short_box, cv2.COLOR_RGB2BGR)
        data['car_color'] = mvehicle_info.get('mvehicle_color')
        car_img_id = uploadfile(short_box)
        data["shortCutFile"] = car_img_id.decode()
        data['car_plate_no'] = vehicle_plate
        data['car_plate_color'] = plate_color
        data['visor'] = decorate_info.get('visor', [])
        data['accessories'] = decorate_info.get('accessories', [])
        data['saftey'] = [1] if seatbelt_result == 'not_have' else []
        data['LeftTopX'] = xmin
        data['LeftTopY'] = ymax
        data['RightBtmX'] = xmax
        data['RightBtmY'] = ymin
        data['carType'] = mvehicle_info.get('mvehicle_type')
        data['distractedDriving'] = 'yes' if distraction_result == 'calling' else 'no'

        
        event_list.append(data)
    print(data)
    return event_list

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
