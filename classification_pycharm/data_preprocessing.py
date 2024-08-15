<<<<<<< HEAD
import os

import cv2
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
from utils import apply_random_transform_and_save
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Read the training set label and image path
def load_data(labels_csv, images_dir, validation_labels_csv, validation_images_dir):
    labels_df = pd.read_csv(labels_csv)
    file_paths = [os.path.join(images_dir, fname) for fname in labels_df['image']]
    # read the label
    labels = labels_df['myopic_maculopathy_grade'].values

    # Read the validation set label and image path
    validation_labels_df = pd.read_csv(validation_labels_csv)
    validation_file_paths = [os.path.join(validation_images_dir, fname) for fname in validation_labels_df['image']]
    # read the label
    validation_labels = validation_labels_df['myopic_maculopathy_grade'].values

    return file_paths, labels, validation_file_paths, validation_labels


# this is pre-training data augmentation
def augment_data(file_paths, labels):
    enhanced_file_paths = []
    enhanced_labels = []

    # define data enhancement transformations
    target_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # is it inconsistent with reality?
        transforms.RandomRotation(10),
        transforms.Resize(224),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
    ])

    # define data enhancement transformations using albumentaions library
    transform_train = A.Compose([
        # p is probability of using certain transformation
        A.Flip(always_apply=False, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.Resize(height=224, width=224, p=1.0),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), p=1.0),
        # to tensor
        ToTensorV2(),
    ], p=1)

    for path, label in zip(file_paths, labels):
        enhanced_file_paths.append(path)
        enhanced_labels.append(label)
        # if label in [0]:  # add class 0 sample
        #     for _ in range(1):  # times of repetition
        #         transformed_path = apply_random_transform_and_save(path, transform_train)
        #         enhanced_file_paths.append(transformed_path)
        #         enhanced_labels.append(label)
        # if label in [1]:  # add class 1 sample
        #     for _ in range(1):  # times of repetition
        #         transformed_path = apply_random_transform_and_save(path, transform_train)
        #         enhanced_file_paths.append(transformed_path)
        #         enhanced_labels.append(label)
        if label in [2]:  # add class 2 sample
            for _ in range(1):  # times of repetition
                transformed_path = apply_random_transform_and_save(path, transform_train)
                enhanced_file_paths.append(transformed_path)
                enhanced_labels.append(label)
        if label in [3]:  # add class 3 sample
            for _ in range(5):  # times of repetition
                transformed_path = apply_random_transform_and_save(path, transform_train)
                enhanced_file_paths.append(transformed_path)
                enhanced_labels.append(label)
        if label in [4]:  # add class 4 sample
            for _ in range(7):  # times of repetition
                transformed_path = apply_random_transform_and_save(path, transform_train)
                enhanced_file_paths.append(transformed_path)
                enhanced_labels.append(label)

    return enhanced_file_paths, enhanced_labels


# data preprocessing functions
# use CLAHE to preprocess images
def apply_clahe_to_image(image):
    image_np = np.array(image)
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    # separate LAB channels
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    # combine channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_np_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    # convert NumPy array to PIL images
    image_clahe = Image.fromarray(image_np_clahe)

    return image_clahe


# use load_ben_color (this is not used as it is not useful)
def load_ben_color(path):
    # read images
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # sharpened the image
    # the input image is processed by Gaussian blur, elevates the edges of the original image
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (6, 6), 8), -4, 128)

    return image
=======
import os

import cv2
import numpy as np
import pandas as pd
from collections import Counter
from PIL import Image
from utils import apply_random_transform_and_save
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


# Read the training set label and image path
def load_data(labels_csv, images_dir, validation_labels_csv, validation_images_dir):
    labels_df = pd.read_csv(labels_csv)
    file_paths = [os.path.join(images_dir, fname) for fname in labels_df['image']]
    # read the label
    labels = labels_df['myopic_maculopathy_grade'].values

    # Read the validation set label and image path
    validation_labels_df = pd.read_csv(validation_labels_csv)
    validation_file_paths = [os.path.join(validation_images_dir, fname) for fname in validation_labels_df['image']]
    # read the label
    validation_labels = validation_labels_df['myopic_maculopathy_grade'].values

    return file_paths, labels, validation_file_paths, validation_labels


# this is pre-training data augmentation
def augment_data(file_paths, labels):
    enhanced_file_paths = []
    enhanced_labels = []

    # define data enhancement transformations
    target_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # is it inconsistent with reality?
        transforms.RandomRotation(10),
        transforms.Resize(224),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
    ])

    # define data enhancement transformations using albumentaions library
    transform_train = A.Compose([
        # p is probability of using certain transformation
        A.Flip(always_apply=False, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.Resize(height=224, width=224, p=1.0),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), p=1.0),
        # to tensor
        ToTensorV2(),
    ], p=1)

    for path, label in zip(file_paths, labels):
        enhanced_file_paths.append(path)
        enhanced_labels.append(label)
        # if label in [0]:  # add class 0 sample
        #     for _ in range(1):  # times of repetition
        #         transformed_path = apply_random_transform_and_save(path, transform_train)
        #         enhanced_file_paths.append(transformed_path)
        #         enhanced_labels.append(label)
        # if label in [1]:  # add class 1 sample
        #     for _ in range(1):  # times of repetition
        #         transformed_path = apply_random_transform_and_save(path, transform_train)
        #         enhanced_file_paths.append(transformed_path)
        #         enhanced_labels.append(label)
        if label in [2]:  # add class 2 sample
            for _ in range(1):  # times of repetition
                transformed_path = apply_random_transform_and_save(path, transform_train)
                enhanced_file_paths.append(transformed_path)
                enhanced_labels.append(label)
        if label in [3]:  # add class 3 sample
            for _ in range(5):  # times of repetition
                transformed_path = apply_random_transform_and_save(path, transform_train)
                enhanced_file_paths.append(transformed_path)
                enhanced_labels.append(label)
        if label in [4]:  # add class 4 sample
            for _ in range(7):  # times of repetition
                transformed_path = apply_random_transform_and_save(path, transform_train)
                enhanced_file_paths.append(transformed_path)
                enhanced_labels.append(label)

    return enhanced_file_paths, enhanced_labels


# data preprocessing functions
# use CLAHE to preprocess images
def apply_clahe_to_image(image):
    image_np = np.array(image)
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    # separate LAB channels
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    # combine channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    image_np_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    # convert NumPy array to PIL images
    image_clahe = Image.fromarray(image_np_clahe)

    return image_clahe


# use load_ben_color (this is not used as it is not useful)
def load_ben_color(path):
    # read images
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    # sharpened the image
    # the input image is processed by Gaussian blur, elevates the edges of the original image
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (6, 6), 8), -4, 128)

    return image
>>>>>>> db3e9e707d5fd9ad3c06f9b58016989e2e363368
