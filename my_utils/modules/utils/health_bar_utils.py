import cv2
import numpy as np

class HealthBarUtils:

    def __init__(
            self, 
            blue_bar_template_path = 'templates/health_bar_level_template.jpg',
            red_bar_template_path = 'templates/red_health_bar_level_template.jpg',
            matching_threshold = 0.8,
            save_for_testing = False,
    ) -> None:

        self.blue_health_level_identifier = cv2.imread(blue_bar_template_path)

        # TODO: save a template for red health bar too.
        if red_bar_template_path is None: self.red_health_level_identifier = cv2.imread(blue_bar_template_path)
        else: self.red_health_level_identifier = cv2.imread(red_bar_template_path)

        self.matching_threshold = matching_threshold
        self.save_for_testing = save_for_testing

    def get_health_level(self, health_bar_img: np.array, bar_color: str = 'blue') -> float:

        if bar_color == 'blue': 
            try: template_matching_result = cv2.matchTemplate(health_bar_img, self.blue_health_level_identifier, cv2.TM_CCOEFF_NORMED)
            except: 
                print('template error')
                return 1.
        elif bar_color == 'red': 
            try: template_matching_result = cv2.matchTemplate(health_bar_img, self.red_health_level_identifier, cv2.TM_CCOEFF_NORMED)
            except: 
                print('template error')
                return 1.

        _, max_val, _, max_loc = cv2.minMaxLoc(template_matching_result)

        if max_val > self.matching_threshold:
            health_level = (max_loc[0]/health_bar_img.shape[1])
        else:
            health_level = 1.

        if self.save_for_testing:
            print(f'health level = {health_level:.3f}%')
            if max_val > self.matching_threshold: cv2.circle(health_bar_img, max_loc, radius=1, color=(0,255,255), thickness=2)
            cv2.imwrite('./templates/health_bar_matched_template.jpg', health_bar_img)

        return health_level