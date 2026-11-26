import os
from torch.utils.data import Dataset
import cv2

class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.img_dir, self.images[idx]))
        return self.transform(image), True

class CustomImageDatasetBuffer(Dataset):
    def __init__(self, img_dir, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.images = [self.transform(cv2.imread(os.path.join(self.img_dir, name))) for name in os.listdir(img_dir)]
        print("Dataset loaded in memory")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], True