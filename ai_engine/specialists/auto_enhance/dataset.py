import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class HDRDataset(Dataset):
    """
    Dataset để nạp 8 cặp ảnh sạch from data/pairs/{before,after}/
    Trả về (full_before, proxy_before_256, full_after).
    """
    def __init__(self, data_dir="data/pairs", transform=None, is_train=True):
        self.before_dir = os.path.join(data_dir, "before")
        self.after_dir = os.path.join(data_dir, "after")
        self.transform = transform
        self.is_train = is_train
        
        # Lấy danh sách ảnh trong before_dir
        if os.path.exists(self.before_dir):
            self.filenames = [f for f in os.listdir(self.before_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        else:
            self.filenames = []
            
        self.filenames.sort()
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        
        before_path = os.path.join(self.before_dir, filename)
        after_path = os.path.join(self.after_dir, filename)
        
        # Đọc ảnh (BGR từ OpenCV)
        before_img = cv2.imread(before_path)
        after_img = cv2.imread(after_path)
        
        if before_img is None or after_img is None:
            raise FileNotFoundError(f"Không tìm thấy file hoặc lỗi đọc: {filename}")
            
        # Chuyển đổi sang RGB
        before_img = cv2.cvtColor(before_img, cv2.COLOR_BGR2RGB)
        after_img = cv2.cvtColor(after_img, cv2.COLOR_BGR2RGB)
        
        # Tạo Proxy 256x256 từ before
        proxy_img = cv2.resize(before_img, (256, 256), interpolation=cv2.INTER_AREA)
        
        # Augmentation đơn giản khi train
        if self.is_train and np.random.rand() > 0.5:
            # Lật ngang
            before_img = cv2.flip(before_img, 1)
            after_img = cv2.flip(after_img, 1)
            proxy_img = cv2.flip(proxy_img, 1)
            
        # Chuẩn hóa về [0, 1] float32
        before_tensor = torch.from_numpy(before_img.transpose(2, 0, 1).copy()).float() / 255.0
        after_tensor = torch.from_numpy(after_img.transpose(2, 0, 1).copy()).float() / 255.0
        proxy_tensor = torch.from_numpy(proxy_img.transpose(2, 0, 1).copy()).float() / 255.0
        
        return before_tensor, proxy_tensor, after_tensor
