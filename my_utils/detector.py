import numpy as np
import torch

from my_utils.health_bar_utils import HealthBarUtils
import time

import sys
# Add folder path to Python path
folder_path = "./yolov9"
sys.path.append(folder_path)

from yolov9.models.common import DetectMultiBackend
from yolov9.utils.augmentations import letterbox
from yolov9.utils.general import non_max_suppression, scale_boxes
from yolov9.utils.plots import Annotator, colors


class Detector:

    def __init__(
        self,
        weight_path = './weights/best.pt',
        device = 'mps',
        dnn = False,
        half = False,
        imgsz = (960, 960),

    ):
        
        self.model = DetectMultiBackend(weight_path, device=device, dnn=dnn, fp16=half)

        self.imgsz = imgsz
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.object_counts_template = {
            object_name: 0 for object_name in self.model.names.values()
        }

        self.health_bar_utils = HealthBarUtils()

    def detect_objects(self, img):

        prev_time = time.time()
        im0 = np.ascontiguousarray(img.copy())  # BGR
        im = letterbox(im0, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        print("--- detector time 1 %s seconds ---" % (time.time() - prev_time))

        prev_time = time.time()
        pred = self.model(im, augment=False, visualize=False)

        pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
        pred = non_max_suppression(pred)

        print("--- detector time 2 %s seconds ---" % (time.time() - prev_time))

        prev_time = time.time()
        annotator = Annotator(im0, line_width=1)

        red_minions = list()
        blue_minions = list()
        red_health_bars = list()
        blue_health_bars = list()
        cur_object_counts = self.object_counts_template.copy()
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    xyxy = [tensor.item() for tensor in xyxy]
                    c = int(cls)  # integer class
                    cur_object_counts[self.names[c]] += 1
                    annotator.box_label(xyxy, self.names[c], color=colors(c, True))
                    if 'red' in self.names[c]: 
                        if 'minion' in self.names[c]:
                            red_minions.append((xyxy, self.names[c], conf))
                        else:
                            health_level = self.health_bar_utils.get_health_level(health_bar_img=img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])], bar_color='red')
                            red_health_bars.append((xyxy, health_level, conf))
                    else:
                        if 'minion' in self.names[c]:
                            blue_minions.append((xyxy, self.names[c], conf))
                        else:
                            health_level = self.health_bar_utils.get_health_level(health_bar_img=img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])], bar_color='blue')
                            blue_health_bars.append((xyxy, health_level, conf))
                        
        print("--- detector time 3 %s seconds ---" % (time.time() - prev_time))

        return red_minions, blue_minions, red_health_bars, blue_health_bars, cur_object_counts, annotator.result()
