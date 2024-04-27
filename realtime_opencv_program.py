import cv2
import numpy as np
import time
import mss

from my_utils.detector import Detector
from my_utils.algorithms import ObjectAssociator
from my_utils.last_hitter import LastHitter


H, W = 2234//2, 3456//2

sct = mss.mss()

if __name__ == '__main__':

    print('LOADING DETECTOR..')
    detector = Detector(imgsz=(960,960))
    print('LOADING ASSOCIATOR..')
    object_associator = ObjectAssociator()
    print('LOADING LAST HITTER..')
    last_hitter = LastHitter(W=W/2, H=H/2)

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
        img = cv2.resize(img, (W, H))
        print(f'\nframe {x}:')
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        original_img = img.copy()
        print("--- screenshot %s seconds ---" % (time.time() - prev_time))

        # Object detection
        prev_time = time.time()
        red_minions, blue_minions, red_health_bars, blue_health_bars, cur_object_counts, debug_img = detector.detect_objects(img)
        print("--- detect_objects %s seconds ---" % (time.time() - prev_time))

        # Match health bars with minions and create minions
        prev_time = time.time()
        matched_blue_minions = object_associator.match_bars_to_minions(blue_minions, blue_health_bars)
        matched_red_minions = object_associator.match_bars_to_minions(red_minions, red_health_bars)
        print("--- match minions %s seconds ---" % (time.time() - prev_time))

        # HIT THE LOW HEALTH MINION
        prev_time = time.time()
        hit_pixel = last_hitter.hit(minions=matched_red_minions)
        print("--- hit to the minion %s seconds ---" % (time.time() - prev_time))
        if hit_pixel is not None:
            cv2.circle(img, hit_pixel, radius=5, color=(0,0,255), thickness=10)
            
        # Annotate image
        # TODO: Create a class for this
        img = object_associator.annotate_minions(img, matched_blue_minions, color=(255,255,0))
        img = object_associator.annotate_minions(img, matched_red_minions)
        # Write object counts on top right corner
        text_y = 40
        for label, count in cur_object_counts.items():
            cv2.putText(img, f'{label}: {count}', (W - 400, text_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            text_y += 30

        # Show the output
        cv2.imshow("Object Detection", img)

        if x%1 == 0:
            cv2.imwrite(f"frames3/{x}.jpg", img)
            # cv2.imwrite(f"frames3/{x}_original.jpg", original_img)
            cv2.imwrite(f"frames3/{x}_debug.jpg", debug_img)

    cv2.destroyAllWindows()
