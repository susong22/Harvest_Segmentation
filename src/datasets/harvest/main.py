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

