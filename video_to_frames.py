import cv2
import os


def extract_frame_numbers(folder):
    """Extract frame numbers from filenames in a folder."""
    frame_numbers = set()
    for filename in os.listdir(folder):
        if filename.startswith('frame_') and (filename.endswith('.txt') or filename.endswith('.jpg')):
            frame_numbers.add(int(filename.split('_')[1].split('.')[0]))
    return frame_numbers

def video_to_frames(video_path, output_folder, annotations_path):

    frame_numbers = extract_frame_numbers(annotations_path)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = 0

    # Loop through each frame in the video
    while success:
        # Write the current frame to a file in the output folder
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
        if frame_count in frame_numbers: cv2.imwrite(frame_filename, image)  # Save frame as JPEG file
        success, image = vidcap.read()  # Read next frame
        frame_count += 1

    # Release the video capture object
    vidcap.release()

# Example usage:
video_path = "dataset files/my_gameplay/my_gameplay.mov"
output_folder = "dataset files/my_gameplay/data"
annotations_path = 'dataset files/my_gameplay/obj_train_data'
video_to_frames(video_path, output_folder, annotations_path)