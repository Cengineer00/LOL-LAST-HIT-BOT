import cv2
import argparse

from my_utils.loaders.frame_loader import FrameLoader
from my_utils.modules.detector import Detector
from my_utils.modules.algorithms import ObjectAssociator
from my_utils.modules.last_hitter import LastHitter
from my_utils.visualizers.annotator import AIAnnotator


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor_idx', type=int, default=1, required=False, help="The index of the monitor to capture screenshots from (0 for primary monitor)")
    parser.add_argument('--visualize', action='store_false', help="Set to `True` to enable visualization, `False` to disable.")
    parser.add_argument('--minion_color', type=str, default='red', required=False, help="Specify the color of minions to target (e.g., `blue`, `red`).")
    parser.add_argument('--H', type=int, default=2234//2, required=False, help='Half the height of the screen resolution.')
    parser.add_argument('--W', type=int, default=3456//2, required=False, help='Half the width of the screen resolution.')
    args = parser.parse_args()

    monitor_idx = args.monitor_idx
    visualize = args.visualize
    minion_color_to_last_hit = args.minion_color
    H, W = args.H, args.W

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
            # Show the output
            cv2.imshow("Last Hit AI Result", ai_visualized_img)

    cv2.destroyAllWindows()