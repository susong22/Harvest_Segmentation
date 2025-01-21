import os
import json
import numpy as np
import cv2

from ..dataset_base import DatasetBase
from .harvest import HarvestBase

class Harvest(HarvestBase, DatasetBase):
    def __init__(self,
                 data_dir=None,
                 n_classes=3,
                 split='train', 
                 depth_mode='raw',
                 with_input_orig=False):
        super(Harvest, self).__init__()
        # assert n_classes in self.N_CLASSES
        
        self._n_classes = 3
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
        fp = os.path.join(self._data_dir, self._split, self.RGB_DIR, f'{self._filenames[idx]}.jpg')
        img = cv2.imread(fp)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_depth(self, idx): 
        fp = os.path.join(self._data_dir, self._split, self.DEPTH_DIR, f'{self._filenames[idx]}.jpg')
        depth = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        
        # Ensure depth is a single channel image
        if len(depth.shape) == 3 and depth.shape[2] == 4:
            depth = depth[:, :, 0]
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        return depth

    def load_label(self, idx):
        json_fp = os.path.join(self._data_dir, self._split, self.LABELS_DIR, f'{self._filenames[idx]}.json')
        with open(json_fp) as f:
            data = json.load(f)

        # 기본값은 2 (나머지 영역)
        height, width = 1200, 1920
        mask = np.full((height, width), 0, dtype=np.uint8)  # 기본값 2로 초기화

         # 라벨에 따라 값을 설정 ('field' = 1, 'machine' = 2)
        for shape in data['shapes']:
            label = shape['label']
            points = np.array(shape['points'])

            if label == 'field':
                cv2.fillPoly(mask, [points.astype(np.int32)], 1)  # field = 1
            elif label == 'machine':
                cv2.fillPoly(mask, [points.astype(np.int32)], 2)  # machine = 2

        # 크기 조정 (interpolation 확인)
        target_height, target_width = 480, 640  # 예제 크기
        mask_resized = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        # # 디버깅: 라벨 값 확인
        # print(f"Original Mask Unique Values: {np.unique(mask)}")
        # print(f"Resized Mask Unique Values: {np.unique(mask_resized)}")

        return mask_resized


    def __len__(self):
        return len(self._filenames)