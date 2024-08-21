import cv2

from my_utils.visualizers.last_hit_wave import Wave
from my_utils.data_structures.minion import Minion

from yolov9.utils.plots import Annotator

class AIAnnotator:
    
    def __init__(self, W = 3456//2) -> None:
        self.W = W
        self.waves = list()

    def __call__(self, hit_pixel, minions, img, object_counts, frame_no):
        # Check if there is a new click at this frame
        if hit_pixel is not None:
            new_wave = Wave(hit_pixel, frame_no, max_frames=5, max_radius=75)
            self.waves.append(new_wave)
        
        # Update and apply all active waves
        for wave in self.waves:
            if wave.active:
                img = wave.update(img, frame_no)
        
        # Remove inactive waves
        self.waves = [wave for wave in self.waves if wave.active]
            
        # Annotate image
        img = self.annotate_minions(img, minions)

        # Write object counts on top right corner
        text_y = 40
        for label, count in object_counts.items():
            cv2.putText(img, f'{label}: {count}', (self.W - 400, text_y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
            text_y += 30

        return img

    def annotate_minions(self, img, minions: list[Minion], line_thickness = 1, color = (0, 0, 255)):
        annotator = Annotator(img, line_width=line_thickness)

        for minion in minions:
            label = f'{minion.minion_type} {minion.confidence:.2f}, health: {100*minion.health_level:.1f}%'
            annotator.box_label(minion.bbox, label, color=color)

        # Stream results
        return annotator.result()