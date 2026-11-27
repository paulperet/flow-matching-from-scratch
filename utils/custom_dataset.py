import os
from torch.utils.data import Dataset
import cv2
import random
import multiprocessing

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

class CustomImageDatasetBufferSingle(Dataset):
    def __init__(self, img_dir, paths, transform):
        self.img_dir = img_dir
        self.transform = transform
        self.cpu_count = os.cpu_count()
        print(f"Running data processing on {self.cpu_count} cores")
        with multiprocessing.Pool(self.cpu_count) as pool:
            self.images = pool.map(self._process_item, paths)
        print("Dataset loaded in memory")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], True

    def _process_item(self, name):
        return self.transform(cv2.imread(os.path.join(self.img_dir, name)))

def custom_dataset_split(img_dir, transform, val_size=1000, shuffle=True):
    image_paths = os.listdir(img_dir)
    if shuffle:
        random.shuffle(image_paths)
    train_set = CustomImageDatasetBufferSingle(img_dir, image_paths[val_size:], transform)
    val_set = CustomImageDatasetBufferSingle(img_dir, image_paths[:val_size], transform)
    return train_set, val_set