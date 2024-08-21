import numpy as np
import mss
import cv2


class FrameLoader:
    
    def __init__(self, H = 2234//2, W = 3456//2, monitor_idx = 1) -> None:
        self.sct = mss.mss()
        self.monitor_idx = monitor_idx
        self.H, self.W = H, W
        pass

    def __iter__(self):
        return self
    
    def __next__(self):
        
        img = np.array(self.sct.grab(self.sct.monitors[self.monitor_idx]))
        img = cv2.resize(img, (self.W, self.H))
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        return img