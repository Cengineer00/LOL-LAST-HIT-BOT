import numpy as np
from pynput.mouse import Button, Controller

from my_utils.data_structures.minion import Minion


class LastHitter:

    def __init__(self, W=3456/2, H=2234/2, fps=5):
        self.minion_total_healths = {
            'red caster minion': 300,
            'blue caster minion': 300,
            'red melee minion': 500,
            'blue melee minion': 500,
            'red siege minion': 1500,
            'blue siege minion': 1500,
        }

        self.hero_position = np.array([W, H])
        self.mouse = Controller()

        self.fps = fps

        self.last_hitted_time = -99999

        self._update_AD()
        self._update_AS()

    def __call__(self, minions, frame_no, click=True):
        return self.hit(minions, frame_no, click)

    def hit(self, minions: list[Minion], frame_no: int, click=True):

        # If attack speed is not enough to hit again
        if (frame_no - self.last_hitted_time)/self.fps < (1/self.hero_attack_speed):
            return None
        
        min_distance_to_hero = 999999
        hit_pixel = None
        for i, minion in enumerate(minions):
            if minion.health_level*self.minion_total_healths[minion.minion_type] <= self.hero_AD:
                middle_of_minion = np.array([int((minion.bbox[0]+minion.bbox[2])/2), int((minion.bbox[1]+minion.bbox[3])/2)])
                distance_to_hero = np.linalg.norm(self.hero_position - middle_of_minion)

                if min_distance_to_hero > distance_to_hero:
                    min_distance_to_hero = distance_to_hero
                    hit_pixel = middle_of_minion
                    self.last_hitted_time = frame_no

        if hit_pixel is not None and click:
            print(f'clicked: {hit_pixel}')
            self.mouse.position = (hit_pixel[0], hit_pixel[1])
            self.mouse.click(Button.right)
            
        return hit_pixel
    
    def _update_AD(self):
        # TODO: read from static position at the image, maybe every n second
        self.hero_AD = 80

    def _update_AS(self):
        # TODO: read from static position at the image, maybe every n second
        self.hero_attack_speed = 1