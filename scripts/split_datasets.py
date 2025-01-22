import os
import shutil
import random
import re

def split_datasets(source_path, dest_path, train_ratio=0.8, seed=None):
    if seed is not None:
        random.seed(seed)  # 재현 가능한 랜덤성을 위해 시드 설정

    # 폴더 경로 설정
    rgb_path = os.path.join(source_path, 'rgb')
    depth_path = os.path.join(source_path, 'depth')
    labels_path = os.path.join(source_path, 'labels')

    # train/test 폴더 생성
    train_path = os.path.join(dest_path, 'train')
    test_path = os.path.join(dest_path, 'test')

    for folder in ['rgb', 'depth', 'labels']:
        os.makedirs(os.path.join(train_path, folder), exist_ok=True)
        os.makedirs(os.path.join(test_path, folder), exist_ok=True)

     # 파일 이름 패턴 설정 (rotary 또는 harvest로 시작)
    depth_pattern = r'(rotary_\d+|harvest_\d+)_depth(\d+)'  # rotary_123 또는 harvest_123로 시작
    rgb_pattern = r'(rotary_\d+|harvest_\d+)_left(\d+)'     # rotary_123 또는 harvest_123로 시작

    # 파일 매칭
    depth_files = [f for f in os.listdir(depth_path) if f.endswith('.png')]
    rgb_files = [f for f in os.listdir(rgb_path) if f.endswith('.png')]
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.json')]

    # 파일 ID 매핑
    depth_ids = {re.match(depth_pattern, f).groups() for f in depth_files if re.match(depth_pattern, f)}
    rgb_ids = {re.match(rgb_pattern, f).groups() for f in rgb_files if re.match(rgb_pattern, f)}
    matched_ids = depth_ids & rgb_ids  # depth와 rgb의 공통 ID

    # 파일 세트로 매칭
    matched_files = []
    for harvest_or_rotary_id, index in matched_ids:
        depth_file = f"{harvest_or_rotary_id}_depth{index}.png"
        rgb_file = f"{harvest_or_rotary_id}_left{index}.png"
        label_file = f"{harvest_or_rotary_id}_left{index}.json"
        if label_file in label_files:  # labels에도 존재해야 함
            matched_files.append((depth_file, rgb_file, label_file))

    # 데이터 섞기 (random성을 높이기 위해 random.sample 사용)
    shuffled_files = random.sample(matched_files, len(matched_files))

    # train/test 나누기
    split_index = int(len(shuffled_files) * train_ratio)
    train_files = shuffled_files[:split_index]
    test_files = shuffled_files[split_index:]

    # 파일 복사 함수
    def copy_files(file_list, dest_folder):
        for depth_file, rgb_file, label_file in file_list:
            for src_folder, dest_subfolder, file_name in [
                (depth_path, 'depth', depth_file),
                (rgb_path, 'rgb', rgb_file),
                (labels_path, 'labels', label_file),
            ]:
                src_file = os.path.join(src_folder, file_name)
                dest_file = os.path.join(dest_folder, dest_subfolder, file_name)
                if os.path.exists(src_file):
                    shutil.copy(src_file, dest_file)

    # 파일 복사
    copy_files(train_files, train_path)
    copy_files(test_files, test_path)

    print(f"Train/Test split 완료:")
    print(f"Train: {len(train_files)}개, Test: {len(test_files)}개")

if __name__ == '__main__':
    #################### EDIT THIS ####################
    source_path = '/home/agmo/ESANet_0116/ESANet/datasets/harvest_raw'
    dest_path = '/home/agmo/ESANet_0116/ESANet/datasets/harvest'
    ###################################################
    split_datasets(source_path, dest_path, seed=42)