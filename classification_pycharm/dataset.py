import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from data_preprocessing import apply_clahe_to_image


# define the data set class
class MyDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, target_transform=None):
        self.file_paths = file_paths
        self.labels = labels
        #  different transformers to process images in different ways
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        image = Image.open(img_path).convert('RGB')

        # process the image using the load_ben_color function and then apply CLAHE
        # image = load_ben_color(img_path, sigmaX=10)
        # image_pil = Image.fromarray(image)
        # image = apply_clahe_to_image(image_pil)

        # only use CLAHE
        image = apply_clahe_to_image(image)
        label = self.labels[idx]
        image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), img_path


# create a new Dataset class to process the enhanced images and labels
class AugmentedDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

