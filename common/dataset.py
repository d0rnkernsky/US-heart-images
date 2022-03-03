import torch
from torch.utils.data import Dataset
import cv2 as cv
import os
from natsort import natsorted

class USImagesDataset(Dataset):
    """
    Heart ultrasound images dataset
    """

    def __init__(self, img_path, mask_path=None, image_transform=None, label_transform=None):
        """
        Args:
            archive_path (string): Path to the archive with image files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images = natsorted(list(map(lambda f: os.path.join(img_path, f), os.listdir(img_path))))
        if mask_path is not None:
            self.masks = natsorted(list(map(lambda f: os.path.join(mask_path, f), os.listdir(mask_path))))
        else:
            self.masks = None
            
        self.image_transform = image_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img = cv.imread(self.images[idx], cv.IMREAD_GRAYSCALE)
        if self.image_transform is not None:
            img = self.image_transform(img)
        
        if self.masks is None:
            return img
        
        mask = cv.imread(self.masks[idx], cv.IMREAD_GRAYSCALE)
        if self.label_transform is not None:
            mask = self.label_transform(mask)
            
        return img, mask
