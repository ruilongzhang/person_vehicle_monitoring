
import os
import sys

import cv2
import numpy as np

sys.path.append("/app/workdir")

from person_vehicle_monitoring.tools import httpclient
from person_vehicle_monitoring.algorithms.decoration_objs import *
from cpu_nms import cpu_nms

class YoloDecorate(object):
    INPUT_W = 608
    INPUT_H = 608
    BOX_SIZE = 7
    MAX_BOX_COUNT = 1000
    NUM_CLASS = 13

    def __init__(self, cli):
        self.client = cli
        self.pad = None
        self.ratio = None

    def convert_buffer_to_numpy(self, dets: np.ndarray) -> np.ndarray:
        count = min(int(dets[0]), YoloDecorate.MAX_BOX_COUNT)
        dets = dets[1:1 + YoloDecorate.BOX_SIZE * count]
        dets_np = np.array(dets, dtype=np.float32).reshape((-1, YoloDecorate.BOX_SIZE))
        dets_np = dets_np[:count, :]
        # cx,cy,w,h -> x1,y1,x2,y2
        x = dets_np[:, 0].copy()
        y = dets_np[:, 1].copy()
        w = dets_np[:, 2].copy()
        h = dets_np[:, 3].copy()
        dets_np[:, 0] = x - w / 2
        dets_np[:, 1] = y - h / 2
        dets_np[:, 2] = x + w / 2
        dets_np[:, 3] = y + h / 2
        return dets_np

    def nms(self, dets: np.ndarray, nms_thresh: float = 0.4, box_conf_thresh: float = 0.5) -> np.ndarray:
        keep = dets[:, 4] > box_conf_thresh
        dets = dets[keep, :]

        nms_dets = []
        for i in range(YoloDecorate.NUM_CLASS):
            cls = dets[:, 5].astype(np.int32) == i
            dets_cls = dets[cls, :]
            if dets_cls.size == 0:
                continue
            keep = cpu_nms(dets_cls, nms_thresh)  # fast speed nms
            # keep = py_cpu_nms(dets_cls, nms_thresh)    # slow
            nms_dets_cls = dets_cls[keep, :]
            nms_dets.append(nms_dets_cls)

        if len(nms_dets) > 0:
            nms_dets = np.concatenate(nms_dets)
        else:
            nms_dets = np.array([[]], dtype=np.float32)
        return nms_dets
    
    def processImage(self, img: np.ndarray) -> np.ndarray:
        r_w = 1.0 * YoloDecorate.INPUT_W / img.shape[1]
        r_h = 1.0 * YoloDecorate.INPUT_H / img.shape[0]
        shape = img.shape[:2]  # current shape [height, width]
        if r_h > r_w:
            w = int(YoloDecorate.INPUT_W)
            h = int(r_w * img.shape[0])
            x = 0
            y = int((YoloDecorate.INPUT_H - h) / 2)
        else:
            w = int(r_h * img.shape[1])
            h = int(YoloDecorate.INPUT_H)
            x = int((YoloDecorate.INPUT_W - w) / 2)
            y = 0
            
        self.pad = (x, y)
        self.ratio = w/img.shape[1], h/img.shape[0]
        re = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        out = 128 * np.ones((YoloDecorate.INPUT_H, YoloDecorate.INPUT_W, 3), dtype=np.uint8, order='C')
        out[y:y + h, x:x + w, :] = re
        return out


    def decoration_input(self, image):
        img0 = self.processImage(image)
        img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        img1 = img1.transpose([2, 0, 1]) / 255.0
        input_image = np.array(img1, dtype=np.float32, order='C')
        #input_image = np.expand_dims(input_image, axis=0)
        input_name = httpclient.InferInput(name="data", shape=[3, 608, 608], datatype="FP32")
        input_name.set_data_from_numpy(input_image)
        output = self.client.infer('yolov3', inputs=[input_name])
        return output, img0

    def decoration_output(self, resp):
        dets_list = resp.get_response()["outputs"][0]["data"]
        dets_np = self.convert_buffer_to_numpy(dets_list)
        nms_dets_np = self.nms(dets_np, nms_thresh=0.4, box_conf_thresh=0.5)
        return nms_dets_np

    async def decorate_infer(self, img):
        window=[]
        visor=[]
        accessories=[]
        person=[]
        resp, img0 = self.decoration_input(img)
        result = self.decoration_output(resp)
        boxes = self.post_process(result)
        if boxes.size > 0:
           
            window = [box.tolist() for box in boxes if int(box[5]) == 1]
            visor = [box.tolist() for box in boxes if int(box[5]) == 12]
            accessories = [box.tolist() for box in boxes if int(box[5]) in [8,9,10]]
            person = [box.tolist() for box in boxes if int(box[5]) == 4]
           

        return dict(window=window, visor=visor, accessories=accessories, person=person)
    def infer(self, img):
        resp, img0 = self.decoration_input(img)
        result = self.decoration_output(resp)
        boxes = self.post_process(result)
     
        window = [box.tolist() for box in boxes if int(box[5]) == 1]
        visor = [box.tolist() for box in boxes if int(box[5]) == 12]
        accessories = [box.tolist() for box in boxes if int(box[5]) in [8,9,10]]
        import pdb
        pdb.set_trace()
        return boxes


    def post_process(self, boxes):
        raw_boxes =np.array([])
        if boxes.size > 0:
            raw_boxes = boxes.copy()
            raw_boxes[:, 0] = (boxes[:, 0] - self.pad[0]) / self.ratio[0]
            raw_boxes[:, 1] = (boxes[:, 1] - self.pad[1]) / self.ratio[1]
            raw_boxes[:, 2] = (boxes[:, 2] - self.pad[0]) / self.ratio[0]
            raw_boxes[:, 3] = (boxes[:, 3] - self.pad[1]) / self.ratio[1]
        
   
        return raw_boxes

if __name__ == "__main__":
    triton_client = httpclient.InferenceServerClient(url='10.20.5.9:9922')
    decorate = YoloDecorate(triton_client)
    import pdb
    pdb.set_trace()

    
    image = cv2.imread(f'/app/workdir/person_vehicle_monitoring/test_img/6011_1_20201118_063721296_津AT5638_P1.jpg')
    nms_dets_np = decorate.infer(image)
    for box in nms_dets_np:
        x1 = int(box[0])
        y1 = int(box[1])
        x2 = int(box[2])
        y2 = int(box[3])
        class_id = int(box[5])
        score = box[4]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(image, str(class_id), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imwrite("_1.jpg", image)
