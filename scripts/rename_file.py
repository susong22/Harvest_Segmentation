import os
import re

def rename_files(base_path):
    # Define subdirectories
    splits = ['train', 'test']
    subdirs = ['rgb', 'depth', 'labels']
    
    for split in splits:
        for subdir in subdirs:
            dir_path = os.path.join(base_path, split, subdir)
            if not os.path.exists(dir_path):
                continue
                
            # Get all files in directory
            files = os.listdir(dir_path)
            
            for filename in files:
                # Extract frame number using regex
                match = re.search(r'(\d{6})\.(png|json)$', filename)
                if not match:
                    continue
                    
                frame_num = match.group(1)
                extension = match.group(2)
                
                # Create new filename
                new_filename = f"{frame_num}.{extension}"
                
                # Full paths
                old_path = os.path.join(dir_path, filename)
                new_path = os.path.join(dir_path, new_filename)
                
                # Rename file
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")



if __name__ == "__main__":
    #################### EDIT THIS ####################
    dataset_path = "/home/song/ESANet_0104/ESANet/datasets/harvest"
    ##################################################
    
    rename_files(dataset_path)