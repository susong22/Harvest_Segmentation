import os
import random
import shutil
import re

"""
Path Example:

Depth  : harvest_{date}_depth_{number}.png
RGB    : harvest_{date}_left_{number}.png
Labels : harvest_{date}_left_{number}.json
"""

def create_directory_structure(base_path, subset):
    """Create the directory structure for the dataset split."""
    
    dirs = {
        'rgb'   : os.path.join(base_path, subset, 'rgb'),
        'depth' : os.path.join(base_path, subset, 'depth'),
        'labels': os.path.join(base_path, subset, 'labels')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
        
    return dirs


def get_frame_number(filename):
    """Extract frame number from filename."""
    
    match = re.search(r'(\d{6})\.', filename)
    
    return match.group(1) if match else None


def split_dataset(source_path, dest_path, train_ratio=0.8):
    """Split the dataset into training and testing sets."""
    
    # Get all RGB image files
    rgb_files = [f for f in os.listdir(os.path.join(source_path, 'rgb')) 
                 if f.endswith('.png') and 'left' in f]
    
    # Extract frame numbers
    frame_numbers = [get_frame_number(f) for f in rgb_files]
    
    # Randomly shuffle frame numbers
    random.shuffle(frame_numbers)
    
    # Calculate split point
    split_idx = int(len(frame_numbers) * train_ratio)
    train_frames = frame_numbers[:split_idx]
    test_frames = frame_numbers[split_idx:]
    
    # Create directory structure
    train_dirs = create_directory_structure(dest_path, 'train')
    test_dirs = create_directory_structure(dest_path, 'test')
    
    
    def copy_frame_files(frame_num, source_dirs, dest_dirs):
        # Find matching files
        rgb_file = next(f for f in os.listdir(source_dirs['rgb']) 
                       if frame_num in f and 'left' in f)
        depth_file = next(f for f in os.listdir(source_dirs['depth']) 
                         if frame_num in f and 'depth' in f)
        label_file = next(f for f in os.listdir(source_dirs['labels']) 
                         if frame_num in f and 'left' in f)
        
        # Copy files
        shutil.copy2(os.path.join(source_dirs['rgb'], rgb_file), 
                    os.path.join(dest_dirs['rgb'], rgb_file))
        shutil.copy2(os.path.join(source_dirs['depth'], depth_file), 
                    os.path.join(dest_dirs['depth'], depth_file))
        shutil.copy2(os.path.join(source_dirs['labels'], label_file), 
                    os.path.join(dest_dirs['labels'], label_file))
    
    # Source directories
    source_dirs = {
        'rgb': os.path.join(source_path, 'rgb'),
        'depth': os.path.join(source_path, 'depth'),
        'labels': os.path.join(source_path, 'labels')
    }
    
    # Copy files for train and test sets
    print("Copying training files...")
    for frame in train_frames:
        copy_frame_files(frame, source_dirs, train_dirs)
    
    print("Copying testing files...")
    for frame in test_frames:
        copy_frame_files(frame, source_dirs, test_dirs)


if __name__ == '__main__':
    #################### EDIT THIS ####################
    source_path = '/home/song/ESANet_0104/ESANet/datasets/harvest_raw'
    dest_path = '/home/song/ESANet_0104/ESANet/datasets/harvest'
    ###################################################
    
    split_dataset(source_path, dest_path, train_ratio=0.8)