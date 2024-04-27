from my_utils.frame_loader import FrameLoader
from my_utils.detector import Detector
from my_utils.algorithms import ObjectAssociator
from my_utils.last_hitter import LastHitter

frame_loader = FrameLoader()
detector = Detector()
object_associator = ObjectAssociator()
last_hitter = LastHitter()

for frame in frame_loader:
    objects = detector(frame)
    associated_objects = object_associator(objects)
    last_hitter(associated_objects)
    