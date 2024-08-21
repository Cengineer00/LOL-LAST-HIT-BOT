import numpy as np
import cv2

# Function to generate a wave effect centered at (x, y)
def create_wave(frame, center, frame_num, max_frames, max_radius):
    wave_frame = np.copy(frame)
    
    # Calculate current radius of the wave
    current_radius = int(max_radius * (frame_num / max_frames))
    
    # Draw circles to represent the wave
    for r in range(0, current_radius, 5):
        cv2.circle(wave_frame, center, r, (0, 255, 255), -1)
    
    # Fade effect: decrease the opacity as the wave expands
    alpha = 1 - (frame_num / max_frames)
    cv2.addWeighted(wave_frame, alpha, frame, 1 - alpha, 0, wave_frame)
    
    return wave_frame

# Class to manage individual waves
class Wave:
    def __init__(self, position, start_frame, max_frames=5, max_radius=15):
        self.position = position
        self.start_frame = start_frame
        self.max_frames = max_frames
        self.max_radius = max_radius
        self.active = True

    def update(self, frame, current_frame):
        frame_num = current_frame - self.start_frame
        if frame_num < self.max_frames:
            return create_wave(frame, self.position, frame_num, self.max_frames, self.max_radius)
        else:
            self.active = False
            return frame