from src.datasets.harvest.pytorch_dataset import Harvest
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import torch

# 데이터셋 경로 설정
data_dir = "/home/song/ESANet_0104/ESANet/datasets/harvest"

# 데이터셋 초기화
train_dataset = Harvest(data_dir=data_dir, split='train')
test_dataset = Harvest(data_dir=data_dir, split='test')

# 데이터 길이 출력
print(f"Train dataset length: {len(train_dataset)}")
print(f"Test dataset length: {len(test_dataset)}")

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# DataLoader 데이터 확인
for batch_idx, batch in enumerate(train_loader):
    images = batch['image']
    labels = batch['label']
    depths = batch['depth']
    
    print(f"Batch {batch_idx}:")
    print(f" - Images shape: {images.shape}")
    print(f" - Labels shape: {labels.shape}")
    print(f" - Depths shape: {depths.shape}")
    
    # 샘플 시각화
    if batch_idx == 0:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(images[0].permute(1, 2, 0).numpy().astype("uint8"))
        plt.title("Image")
        plt.subplot(1, 3, 2)
        plt.imshow(labels[0].numpy(), cmap="gray")
        plt.title("Label")
        plt.subplot(1, 3, 3)
        plt.imshow(depths[0][0].numpy(), cmap="gray")  # 깊이는 1채널이므로 [0]으로 접근
        plt.title("Depth")
        plt.show()
        break



# from src.datasets.harvest.pytorch_dataset import Harvest
# from torch.utils.data import DataLoader
# import matplotlib.pyplot as plt
# import numpy as np

# # 데이터셋 경로 설정
# data_dir = "/home/song/ESANet_0104/ESANet/datasets/harvest"

# # 데이터셋 초기화
# train_dataset = Harvest(data_dir=data_dir, split='train')
# test_dataset = Harvest(data_dir=data_dir, split='test')

# # 데이터 길이 출력
# print(f"Train dataset length: {len(train_dataset)}")
# print(f"Test dataset length: {len(test_dataset)}")

# # Helper function: 라벨 시각화
# def visualize_label(label):
#     """
#     라벨 데이터를 시각화합니다.
#     Args:
#         label (numpy array): 정수값으로 된 라벨 데이터 (0: background, 1: field).
#     """
#     # 클래스별 색상 정의
#     colors = {
#         0: [128, 0, 128],  # Background (보라색)
#         1: [255, 255, 0],  # Field (노란색)
#     }

#     # 라벨 데이터를 RGB로 변환
#     label_color = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
#     for class_id, color in colors.items():
#         label_color[label == class_id] = color

#     return label_color

# # 첫 번째 데이터 확인 및 시각화
# sample = train_dataset[0]
# image = sample['image'].permute(1, 2, 0).numpy()  # (C, H, W) -> (H, W, C)
# label = sample['label'].numpy()

# # 정규화된 이미지를 복원
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# image = (image * std) + mean
# image = np.clip(image, 0, 1)

# # 라벨 시각화
# label_colored = visualize_label(label)

# # 시각화
# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title("Input Image")
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(label_colored)
# plt.title("Label (Purple: Background, Yellow: Field)")
# plt.axis('off')

# plt.show()
