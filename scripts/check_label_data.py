import cv2
import numpy as np

# 라벨 파일 경로
label_path = "/home/agmo/ESANet_0116/ESANet/datasets/harvest/"

# 라벨 이미지 불러오기 (grayscale 모드로 불러옴)
label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

# 고유한 픽셀 값 확인
unique_values = np.unique(label)
print("Unique pixel values in label:", unique_values)
