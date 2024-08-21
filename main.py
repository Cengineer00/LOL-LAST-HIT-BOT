import cv2

from my_utils.loaders.frame_loader import FrameLoader
from my_utils.modules.detector import Detector
from my_utils.modules.algorithms import ObjectAssociator
from my_utils.modules.last_hitter import LastHitter
from my_utils.visualizers.annotator import AIAnnotator

monitor_idx = 1
visualize = True
minion_color_to_last_hit = 'red'
H, W = 2234//2, 3456//2

frame_loader = FrameLoader(H=H, W=W, monitor_idx=monitor_idx)
detector = Detector()
object_associator = ObjectAssociator()
last_hitter = LastHitter(H=H, W=W)
annotator = AIAnnotator(W=W)

for frame_no, frame in enumerate(frame_loader):
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    detection_result = detector(frame)
    matched_minions = object_associator(detection_result, minion_type=minion_color_to_last_hit, img=frame)
    hit_pixel = last_hitter(minions=matched_minions, frame_no=frame_no)
    if visualize: 
        ai_visualized_img = annotator(hit_pixel=hit_pixel, minions=matched_minions, img=frame, object_counts=detection_result['cur_object_counts'], frame_no=frame_no)
        cv2.imwrite(f"frames/{frame_no}.jpg", ai_visualized_img)
        # Show the output
        cv2.imshow("Last Hit AI Result", ai_visualized_img)

cv2.destroyAllWindows()