import cv2
import numpy as np
import torch
import time
import mss
import pyautogui
import sys
from pynput.mouse import Button, Controller
mouse = Controller()

# Add folder path to Python path
folder_path = "./yolov9"
sys.path.append(folder_path)

from yolov9.models.common import DetectMultiBackend
from yolov9.utils.augmentations import letterbox
from yolov9.utils.general import non_max_suppression, scale_boxes, xyxy2xywh
from yolov9.utils.plots import Annotator, colors

W, H = 1920, 1080
H, W = 2234//2, 3456//2
# monitor_dimensions = {
#         'left': 0,
#         'top': 0,
#         'width': W,
#         'height': H
#     }

red_health_bar_img = cv2.cvtColor(cv2.imread('templates/red_health_bar.png', cv2.IMREAD_UNCHANGED), cv2.COLOR_RGBA2RGB)
sct = mss.mss()

weights = './weights/best.pt'
device = 'mps'
dnn = False
data = './datasets/coco.yaml'
half = False
model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)

imgsz=(960, 960)
stride, names, pt = model.stride, model.names, model.pt
object_counts = {
    object_name: 0 for object_name in model.names.values()
}
conf_thres, iou_thres, classes, agnostic_nms, max_det = 0.25, 0.45, None, False, 1000
line_thickness = 0.5


def detect_objects(img):

    im0 = np.ascontiguousarray(img)  # BGR
    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous


    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    prev_time = time.time()
    pred = model(im, augment=False, visualize=False)
    print("--- inference %s seconds ---" % (time.time() - prev_time))

    pred = pred[0][1] if isinstance(pred[0], list) else pred[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    cur_object_counts = object_counts.copy()
    for det in pred:
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                if 'health' not in names[c]: continue
                cur_object_counts[names[c]] += 1
                label = f'{names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))
                if names[c] == 'blue melee minion':
                    prev_time = time.time()
                    # pyautogui.click(xyxy[0], xyxy[1], button='right')
                    mouse.position = (xyxy[0], xyxy[1])
                    mouse.click(Button.right)
                    print("Mouse clicked at:", xyxy[0], xyxy[1])
                    print("--- click time %s seconds ---" % (time.time() - prev_time))


    # Stream results
    im0 = annotator.result()

    return im0, cur_object_counts

x = 0
while True:
    x += 1
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Capture screen
    prev_time = time.time()
    # img = np.array(ImageGrab.grab(bbox=(0, 0, 1920, 1080)))[:,:,:3]  # Adjust the bbox according to your screen resolution
    img = np.array(sct.grab(sct.monitors[1]))
    print(f'\nframe {x}:')
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    img = cv2.resize(img, (W, H))
    original_img = img.copy()
    print("--- screenshot %s seconds ---" % (time.time() - prev_time))

    # Object detection
    prev_time = time.time()
    img, cur_object_counts = detect_objects(img)
    print("--- detect_objects %s seconds ---" % (time.time() - prev_time))

    # Write object counts on top right corner
    text_y = 40
    for label, count in cur_object_counts.items():
        cv2.putText(img, f'{label}: {count}', (W - 400, text_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        text_y += 30

    # Show the output
    cv2.imshow("Object Detection", img)

    if x%1 == 0: 
        cv2.imwrite(f"frames2/{x}.jpg", img)
        cv2.imwrite(f"frames2/{x}_original.jpg", original_img)


cv2.destroyAllWindows()
