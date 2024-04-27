import os
import shutil

def extract_frame_numbers(folder):
    """Extract frame numbers from filenames in a folder."""
    frame_numbers = set()
    for filename in os.listdir(folder):
        if filename.startswith('frame_') and (filename.endswith('.txt') or filename.endswith('.jpg')):
            frame_numbers.add(int(filename.split('_')[1].split('.')[0]))
    return frame_numbers

def main():
    folder_path = 'middle_midlane_updated'
    folder1 = f"{folder_path}/obj_train_data"
    folder2 = f"{folder_path}/data"
    output_folder1 = f"{folder_path}/annotation_files"
    output_folder2 = f"{folder_path}/image_files"
    output_name = f'{folder_path}'

    # Extract frame numbers from filenames in each folder
    frame_numbers1 = extract_frame_numbers(folder1)
    frame_numbers2 = extract_frame_numbers(folder2)

    # Find common frame numbers
    common_frame_numbers = frame_numbers1.intersection(frame_numbers2)

    # Create output folders if they don't exist
    os.makedirs(output_folder1, exist_ok=True)
    os.makedirs(output_folder2, exist_ok=True)

    # Copy files with common frame numbers to output folders
    for frame_number in common_frame_numbers:
        shutil.copy(os.path.join(folder1, f"frame_{frame_number:06d}.txt"),
                    os.path.join(output_folder1, f"{output_name}_{frame_number:06d}.txt"))
        shutil.copy(os.path.join(folder2, f"frame_{frame_number:06d}.jpg"),
                    os.path.join(output_folder2, f"{output_name}_{frame_number:06d}.jpg"))

if __name__ == "__main__":
    main()
