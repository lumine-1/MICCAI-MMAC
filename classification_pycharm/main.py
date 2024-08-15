import os

import torch
import optuna
import functools
import numpy as np
import torch.nn as nn

import utils
from utils import set_seed
from dataset import MyDataset
from collections import Counter
from train_model import train_model, augmented_train_model, train_only
from torch.utils.data import DataLoader, ConcatDataset
import torchvision.transforms as transforms
from data_preprocessing import augment_data, load_data
# from model import ViTModel, CModel, GoogleModel
from model import ConvNeXtModel, GoogleModel, CModel, ConvNeXtFModel


def prepare_dataloaders(file_paths, labels, validation_file_paths, validation_labels):
    # preprocessing transformation of data loader
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # create training and validation datasets
    train_dataset = MyDataset(file_paths, labels, transform=transform)
    val_dataset = MyDataset(validation_file_paths, validation_labels, transform=transform)

    # create a data loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    return train_loader, val_loader



def combined_data(file_paths, labels, validation_file_paths, validation_labels):
    # preprocessing transformation of data loader

    # if labels are NumPy arrays, convert to lists
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if isinstance(validation_labels, np.ndarray):
        validation_labels = validation_labels.tolist()

    # merge file paths and labels
    combined_file_paths = file_paths + validation_file_paths
    combined_labels = labels + validation_labels

    return combined_file_paths, combined_labels


# create and choose different optimisers and criterion here
def prepare_training(my_model):
    # Define loss functions and optimizers
    criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.Adam(my_model.parameters(), lr=0.0001, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, my_model.parameters()), lr=0.00001, weight_decay=1e-4)

    return criterion, optimizer


def main():
    # set random number
    set_seed(42)

    # read data (remote server)
    # task = 'classification'
    # base_dir = ''
    # labels_csv = os.path.join(base_dir, task, 'Training', 'Groundtruths',
    #                           'MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv')
    # images_dir = os.path.join(base_dir, task, 'Training', 'Images')
    # validation_labels_csv = os.path.join(base_dir, task, 'Validation', 'Groundtruths',
    #                                      'MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv')
    # validation_images_dir = os.path.join(base_dir, task, 'Validation', 'Images')

    # read data (local machine)
    labels_csv = '../classification/Training/Groundtruths/MMAC2023_Myopic_Maculopathy_Classification_Training_Labels.csv'
    images_dir = '../classification/Training/Images'
    validation_labels_csv = '../classification/Validation/Groundtruths/MMAC2023_Myopic_Maculopathy_Classification_Validation_Labels.csv'
    validation_images_dir = '../classification/Validation/Images'

    # load data and augment data
    train_file_paths, train_labels, validation_file_paths, validation_labels = load_data(
        labels_csv, images_dir, validation_labels_csv, validation_images_dir)
    train_file_paths, train_labels = augment_data(train_file_paths, train_labels)


    # dataloader
    train_loader, val_loader = prepare_dataloaders(train_file_paths, train_labels, validation_file_paths, validation_labels)

    # count the number of categories in the training set
    train_class_counts = Counter(train_labels)
    print("The number of categories in the training set:", train_class_counts)

    # initialise the model
    my_model = ConvNeXtModel()

    # test convnext_v2 model here, models can be downloaded directly from huggingface.com
    # (this research work do not contain these models as they are not the best choice)
    # my_model = convnext_v2.convnextv2_huge()
    # my_model.head = nn.Linear(in_features=my_model.head.in_features, out_features=5)

    # parameter freezing (avoid update some parameters when doing back propagation)
    # for name, param in my_model.named_parameters():
    #     if not ('stages.3.2' in name or 'head' in name):
    #         param.requires_grad = False
    #
    # load pre-training weights into the model
    # change which weights to load (need to download first)
    # pretrained_weights = torch.load('convnextv2_huge_22k_384_ema.pt')
    # if 'model' in pretrained_weights:
    #     pretrained_weights = {k.replace('model.', ''): v for k, v in pretrained_weights['model'].items()}
    # state_dict = {k: v for k, v in pretrained_weights.items() if not k.startswith('head.')}
    # my_model.load_state_dict(state_dict, strict=False)

    # prepare to train
    criterion, optimizer = prepare_training(my_model)

    # train the model
    # two training types
    train_model(my_model, train_loader, val_loader, criterion, optimizer, train_file_paths, train_labels)
    # augmented_train_model(my_model, train_file_paths, train_labels, validation_file_paths, validation_labels, criterion, optimizer)

    # nested train or optuna
    # nested_test.nested_train(my_model,paths, labels, criterion, optimizer)
    # nested_test.optuna_train(paths, labels, criterion)


if __name__ == "__main__":
    main()
