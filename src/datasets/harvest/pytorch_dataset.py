import os
import json
import numpy as np
import cv2
import torch

from ..dataset_base import DatasetBase
from .harvest import HarvestBase

import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Harvest(HarvestBase, DatasetBase):
    def __init__(self,
                 data_dir=None,
                 split='train', 
                 depth_mode='raw',
                 with_input_orig=False):
        super(Harvest, self).__init__()
        
        self._n_classes = 2
        self._split = split
        self._depth_mode = depth_mode
        self._with_input_orig = with_input_orig
        self._cameras = ['camera1']
        
        if data_dir is not None:
            self._data_dir = os.path.expanduser(data_dir)
            # Load filenames
            split_dir = os.path.join(self._data_dir, split)
            self._filenames = [f.split('.')[0] for f in os.listdir(os.path.join(split_dir, self.RGB_DIR))]
        
        # Set class info
        self._class_names = self.CLASS_NAMES
        self._class_colors = np.array(self.CLASS_COLORS, dtype='uint8')
        
        # Set depth stats - adjust these based on your depth data
        self._depth_mean = 0.0  # calculate this from your dataset
        self._depth_std = 1.0   # calculate this from your dataset
        
        self._target_height = 480
        self._target_width = 640
        
        
        # Data augmentation 설정
        self.augmentation = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomBrightnessContrast(p=0.5),
            A.Rotate(limit=15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ], additional_targets={'depth': 'image', 'label': 'mask'})

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def depth_mean(self):
        return self._depth_mean

    @property
    def depth_mode(self):
        return self._depth_mode

    @property
    def depth_std(self):
        return self._depth_std

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def n_classes_without_void(self):
        return self._n_classes - 1

    @property
    def source_path(self):
        return self._data_dir

    @property
    def split(self):
        return self._split

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def load_image(self, idx):
        fp = os.path.join(self._data_dir, self._split, self.RGB_DIR, f'{self._filenames[idx]}.png')
        img = cv2.imread(fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # 크기 조정
        img = cv2.resize(img, (640, 480))
        return img

    def load_depth(self, idx): 
        fp = os.path.join(self._data_dir, self._split, self.DEPTH_DIR, f'{self._filenames[idx]}.png')
        depth = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        # Ensure depth is a single channel image
        if len(depth.shape) == 3 and depth.shape[2] == 4:
            depth = depth[:, :, 0]
        # 크기 조정
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
        return depth

    def load_label(self, idx):
        json_fp = os.path.join(self._data_dir, self._split, self.LABELS_DIR, f'{self._filenames[idx]}.json')
        with open(json_fp) as f:
            data = json.load(f)
        # Create binary mask from labelme JSON
        height = data['imageHeight']
        width = data['imageWidth'] 
        mask = np.zeros((height, width), dtype=np.uint8)
        for shape in data['shapes']:
            points = np.array(shape['points'])
            cv2.fillPoly(mask, [points.astype(np.int32)], 1)
        # 크기 조정
        mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_NEAREST)
        return mask

            
        return mask

    def __len__(self):
        return len(self._filenames)
    
    def __getitem__(self, idx):
        # 데이터 로드
        image = self.load_image(idx)  # (H, W, C)
        depth = self.load_depth(idx)  # (H, W)
        label = self.load_label(idx)  # 원본 크기의 라벨
        
        # 라벨이 전부 0인 경우, 다른 샘플로 대체
        if np.all(label == 0):
            new_idx = (idx + 1) % len(self)
            return self.__getitem__(new_idx)
        
        # Data Augmentation 적용
        augmented = self.augmentation(image=image, depth=depth, label=label)
        image = augmented['image']
        depth = augmented['depth']
        label = augmented['label']
        
        # 원본 라벨 저장
        label_orig = label.copy()

        # label_down 생성 (다양한 축소 버전)
        label_down = {}
        for rate in [8, 16, 32]:  # 축소 비율
            down_width = self._target_width // rate
            down_height = self._target_height // rate
            label_down[rate] = cv2.resize(label, (down_width, down_height), interpolation=cv2.INTER_NEAREST)

        # 텐서로 변환
        image_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        depth_tensor = torch.tensor(depth, dtype=torch.float32).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)
        label_orig_tensor = torch.tensor(label_orig, dtype=torch.long)
        label_down_tensors = {rate: torch.tensor(label_down[rate], dtype=torch.long) for rate in label_down}

        # 딕셔너리 반환
        return {
            'image': image_tensor,
            'depth': depth_tensor,
            'label': label_tensor,
            'label_orig': label_orig_tensor,
            'label_down': label_down_tensors
        }
