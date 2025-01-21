import os

def rename_files_in_folder(folder_path, file_extension):
    """
    지정된 폴더 안의 파일들을 순서대로 이름 변경
    :param folder_path: 처리할 폴더 경로
    :param file_extension: 변경할 파일 확장자 (예: '.png', '.json')
    """
    files = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    files.sort()  # 기존 정렬 기준 유지 (안정적 순서)
    
    for idx, file_name in enumerate(files, start=1):
        new_name = f"{idx:06d}{file_extension}"  # 6자리 숫자, 예: 000001.png
        src_path = os.path.join(folder_path, file_name)
        dest_path = os.path.join(folder_path, new_name)
        os.rename(src_path, dest_path)
        print(f"Renamed: {file_name} -> {new_name}")

def rename_files(base_path):
    """
    train 및 test 폴더 내의 파일 이름을 순서대로 변경
    :param base_path: 데이터셋의 최상위 경로
    """
    for split in ['train', 'test']:
        for folder in ['rgb', 'depth', 'labels']:
            folder_path = os.path.join(base_path, split, folder)
            if os.path.exists(folder_path):
                print(f"Renaming files in {folder_path}...")
                if folder == 'labels':
                    rename_files_in_folder(folder_path, '.json')
                else:
                    rename_files_in_folder(folder_path, '.png')

if __name__ == '__main__':
    #################### EDIT THIS ####################
    base_path = '/home/agmo/ESANet_0116/ESANet/datasets/harvest'
    ###################################################
    rename_files(base_path)
    print("All files have been renamed.")
