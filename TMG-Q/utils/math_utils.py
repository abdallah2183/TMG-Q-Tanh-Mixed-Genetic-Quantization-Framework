import cv2
import numpy as np
import torch
from config import DEVICE

def process_frame(frame):
    """
    تحويل صورة البيئة الخام إلى تنسور يفهمه الـ VAE
    Input: (H, W, 3) RGB uint8
    Output: (1, 1, 64, 64) Float Tensor normalized [0, 1]
    """
    # 1. Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # 2. Resize to 64x64 (standard for small VAEs)
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    
    # 3. Normalize (0-255 -> 0.0-1.0)
    normalized = resized.astype(np.float32) / 255.0
    
    # 4. To Tensor & Add Batch/Channel Dims (B, C, H, W)
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    
    return tensor.to(DEVICE)
