from typing import Any
from scipy.optimize import linear_sum_assignment
from my_utils.data_structures.minion import Minion
import numpy as np
import sys

sys.path.append(f'./yolov9')
from yolov9.utils.plots import Annotator, colors


class ObjectAssociator:

    def __init__(self, intersection_threshold = 0.5):
        self.intersection_threshold = intersection_threshold

    def __call__(self, minions, health_bars):
        self.match_bars_to_minions(minions, health_bars)

    def match_bars_to_minions(self, minions, health_bars):

        if len(minions) == 0 or len(health_bars) == 0: return list()

        minion_boxes = np.array([i[0] for i in minions])
        health_bar_boxes = np.array([i[0] for i in health_bars])
        hbars_vs_minions_ious = self._get_intersected_area(health_bar_boxes, minion_boxes)
        
        row_ind, col_ind = linear_sum_assignment(hbars_vs_minions_ious, maximize = True)

        matched_minions = list()
        for hbar_idx, minion_idx in zip(row_ind, col_ind):
            if hbars_vs_minions_ious[hbar_idx, minion_idx] > self.intersection_threshold:
                xyxy = health_bar_boxes[hbar_idx]
                health_level = self.health_bar_utils.get_health_level(health_bar_img=img[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])], bar_color='red')
                matched_minions.append(Minion(
                    minion_type=minions[minion_idx][1], 
                    health_level=health_bars[hbar_idx][1], 
                    bbox=minions[minion_idx][0],
                    confidence=minions[minion_idx][2],
                ))

        return matched_minions

    def annotate_minions(self, img, minions: list[Minion], line_thickness = 1, color = (0, 0, 255)):
        annotator = Annotator(img, line_width=line_thickness)

        for minion in minions:
            label = f'{minion.minion_type} {minion.confidence:.2f}, health: {100*minion.health_level:.1f}%'
            annotator.box_label(minion.bbox, label, color=color)

        # Stream results
        return annotator.result()
    

    def _get_intersected_area(self, health_bar_boxes, minion_boxes):
    
        # Compute intersection coordinates
        x1_intersection = np.maximum(health_bar_boxes[:, 0][:, np.newaxis], minion_boxes[:, 0])
        y1_intersection = np.maximum(health_bar_boxes[:, 1][:, np.newaxis], minion_boxes[:, 1])
        x2_intersection = np.minimum(health_bar_boxes[:, 2][:, np.newaxis], minion_boxes[:, 2])
        y2_intersection = np.minimum(health_bar_boxes[:, 3][:, np.newaxis], minion_boxes[:, 3])

        # Compute intersection area
        intersection_area = np.maximum(x2_intersection - x1_intersection, 0) * np.maximum(y2_intersection - y1_intersection, 0)

        # Compute health bar and minion box areas
        health_bar_areas = (health_bar_boxes[:, 2] - health_bar_boxes[:, 0]) * (health_bar_boxes[:, 3] - health_bar_boxes[:, 1])

        # Compute IoU
        intersection_ratio = np.where(health_bar_areas[:, np.newaxis] > 0, intersection_area / health_bar_areas[:, np.newaxis], 0)

        # Filter IoU based on threshold
        hbars_vs_minions_ious = np.where(intersection_ratio >= self.intersection_threshold, intersection_ratio, 0)

        return hbars_vs_minions_ious