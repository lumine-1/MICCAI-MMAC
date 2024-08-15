<<<<<<< HEAD
import copy
import functools

import torch
import utils
import optuna
import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold, ParameterSampler
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision.transforms import transforms
from tqdm import tqdm
from model import ConvNeXtModel, GoogleModel
from dataset import MyDataset, AugmentedDataset
from optuna.integration import PyTorchLightningPruningCallback

# this is the third way of training models, using corss validation to test the model
# use 4-fold corss validation here
# also use
def nested_train(model, paths, labels, criterion, optimizer, num_epochs=10, k_outer=4, k_inner=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # divide the dataset, use StratifiedKFold to make sure the data distribution is similar
    outer_kfold = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=42)
    inner_kfold = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=42)

    dataset = MyDataset(paths, labels, transform=transform)

    # set hyperparameters
    # (due to the computation power, not able to tune too many parameters)
    param = {
        'lr': np.logspace(-6, -2, 100),
        'epoch': [5, 10, 15, 20, 25]
    }
    n_iter = 10
    param_sampler = ParameterSampler(param, n_iter=n_iter)

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(range(len(dataset)), labels)):
        outer_train_dataset = Subset(dataset, outer_train_idx)
        outer_test_dataset = Subset(dataset, outer_test_idx)

        best_avg = -np.inf
        best_params = []

        for params in param_sampler:
            current_avg_scores = []
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kfold.split(range(len(outer_train_dataset)), [labels[i] for i in outer_train_idx])):

                # no data augmentation
                # inner_val = DataLoader(Subset(outer_train_dataset, inner_val_idx), batch_size=16)
                # inner_train = DataLoader(Subset(outer_train_dataset, inner_train_idx), batch_size=16, shuffle=True)

                # define the label that needs to be enhanced and how many times it should be enhanced
                training_set = Subset(outer_train_dataset, inner_train_idx)
                augmented_images, augmented_labels = augment_dataset(training_set, train_transform)
                augmented_dataset = AugmentedDataset(augmented_images, augmented_labels)
                inner_train = DataLoader(augmented_dataset, batch_size=16, shuffle=True)
                inner_val = DataLoader(Subset(outer_train_dataset, inner_val_idx), batch_size=16)

                model = ConvNeXtModel().to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)

                for epoch in range(params['epoch']):
                    train_epoch(model, inner_train, criterion, optimizer, device)

                all_labels, all_preds = validate_epoch(model, inner_val, criterion, device)

                cm = confusion_matrix(all_labels, all_preds)
                qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
                macro_f1 = f1_score(all_labels, all_preds, average='macro')
                classes = np.unique(all_labels)
                macro_specificity = utils.calculate_specificity(np.array(all_labels), np.array(all_preds), classes)
                avg = (macro_f1 + macro_specificity + qwk) / 3
                current_avg_scores.append(avg)
                print(params)
                print(f"Confusion Matrix:\n{cm}")
                print(f"Average Score: {avg:.4f}")
                if avg < 0.78:
                    print("too small, break")
                    break

            # calculates the average performance under the current parameter configuration
            current_params_avg = np.mean(current_avg_scores)
            print(f"Total Average Score: {current_params_avg:.4f}")

            # save the best result
            if current_params_avg > best_avg:
                best_avg = current_params_avg
                best_params = params

        print("Best Parameters:", best_params)

        # Evaluation on the outer test set
        test_loader = torch.utils.data.DataLoader(outer_test_dataset, batch_size=16, shuffle=False)
        model = ConvNeXtModel().to(device)  # use new ConvNeXtModel
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)

        # retrain the model to use the best parameters
        for epoch in range(best_params['epoch']):  # use best_params
            train_epoch(model, DataLoader(outer_train_dataset, batch_size=16, shuffle=True), criterion, optimizer,
                        device)
        all_labels, all_preds = validate_epoch(model, test_loader, criterion, device)

        # print the result
        test_cm = confusion_matrix(all_labels, all_preds)
        test_qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        test_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        test_macro_specificity = utils.calculate_specificity(np.array(all_labels), np.array(all_preds),
                                                             np.unique(all_labels))
        test_avg = (test_macro_f1 + test_macro_specificity + test_qwk) / 3

        print("////////////////////////////////////////////////////////////////////////")
        print(f"Test Metrics with Best Parameters:")
        print(f"Confusion Matrix:\n{test_cm}")
        print(f"Quadratic Weighted Kappa: {test_qwk:.4f}")
        print(f"Macro F1-Score: {test_macro_f1:.4f}")
        print(f"Macro Specificity: {test_macro_specificity:.4f}")
        print(f"Average Score: {test_avg:.4f}")
        print("////////////////////////////////////////////////////////////////////////")

# simplify the train function
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

# simplify the train function
def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, paths in tqdm(data_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds


# also define preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# also define preprocessing transform for training set
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])


def augment_dataset(dataset, transform):
    augmented_images = []
    augmented_labels = []

    for i in range(len(dataset)):
        img, label, path = dataset[i]

        # add original image
        augmented_images.append(img)
        augmented_labels.append(label)

        # if label in [0]:
        #     for _ in range(1):
        #         augmented_img = transform(img)
        #         augmented_images.append(augmented_img)
        #         augmented_labels.append(label)
        # if label in [1]:
        #     for _ in range(1):
        #         augmented_img = transform(img)
        #         augmented_images.append(augmented_img)
        #         augmented_labels.append(label)
        if label in [2]:
            for _ in range(1):
                augmented_img = transform(img)
                augmented_images.append(augmented_img)
                augmented_labels.append(label)
        if label in [3]:
            for _ in range(5):
                augmented_img = transform(img)
                augmented_images.append(augmented_img)
                augmented_labels.append(label)
        if label in [4]:
            for _ in range(7):
                augmented_img = transform(img)
                augmented_images.append(augmented_img)
                augmented_labels.append(label)

    return augmented_images, augmented_labels

# this functino is to create an optuna parameter search work
# very time-consuming (not give the best result)
def objective(trial, labels, outer_train_dataset, outer_train_idx):
    # define the hyperparameter space
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 1, 50)
    criterion = nn.CrossEntropyLoss()
    scores = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inner_kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            inner_kfold.split(range(len(outer_train_dataset)), [labels[i] for i in outer_train_idx])):

        # use divided data
        training_set = Subset(outer_train_dataset, inner_train_idx)
        augmented_images, augmented_labels = augment_dataset(training_set, train_transform)
        augmented_dataset = AugmentedDataset(augmented_images, augmented_labels)
        inner_train = DataLoader(augmented_dataset, batch_size=16, shuffle=True)
        inner_val = DataLoader(Subset(outer_train_dataset, inner_val_idx), batch_size=16)

        # use a new model (avoid using the same model for different trails)
        model = ConvNeXtModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(epochs):
            train_epoch(model, inner_train, criterion, optimizer, device)
        all_labels, all_preds = validate_epoch(model, inner_val, criterion, device)

        # evaluate
        cm = confusion_matrix(all_labels, all_preds)
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        classes = np.unique(all_labels)
        macro_specificity = utils.calculate_specificity(np.array(all_labels), np.array(all_preds), classes)
        avg = (macro_f1 + macro_specificity + qwk) / 3
        scores.append(avg)
        print(lr)
        print(epochs)
        print(f"Confusion Matrix:\n{cm}")
        print(f"Average Score: {avg:.4f}")

    average_score = sum(scores) / len(scores)
    print(f"Final Score: {average_score:.4f}")
    return average_score


# train using optuna (this is too time-consuming, did not give the best answer)
def optuna_train(paths, labels, criterion):
    # create dataset instance
    dataset = MyDataset(paths, labels, transform=transform)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # process for training and validation
    outer_kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(range(len(dataset)), labels)):
        outer_train_dataset = Subset(dataset, outer_train_idx)
        outer_test_dataset = Subset(dataset, outer_test_idx)

        augmented_images, augmented_labels = augment_dataset(outer_train_dataset, train_transform)
        augmented_dataset = AugmentedDataset(augmented_images, augmented_labels)
        outer_train = DataLoader(augmented_dataset, batch_size=16, shuffle=True)
        outer_test = DataLoader(outer_test_dataset, batch_size=16)

        # partial_objective = functools.partial(objective, labels=labels, outer_train_dataset=outer_train_dataset,
        #                                       outer_train_idx=outer_train_idx)
        # we need the result to be larger
        # study = optuna.create_study(direction='maximize')
        # study.optimize(partial_objective, n_trials=12)
        # utils.visualise_optuna(study)
        #
        # choose 3 best trail (because I want to test not only the best result, to prove it can )
        # sorted_trials = sorted(study.trials, key=lambda trial: trial.value, reverse=True)
        # best_three_trials = sorted_trials[:3]
        # print(best_three_trials)
        #

        # this is used to test the best settings found, can be deleted
        # best_params = study.best_trial.params
        best_params = {'lr': 2.7795834117065912e-05, 'epochs': 38}

        model = ConvNeXtModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)

        for epoch in range(best_params['epochs']):
            train_epoch(model, outer_train, criterion, optimizer, device)
        all_labels, all_preds = validate_epoch(model, outer_test, criterion, device)

        # evaluate
        cm = confusion_matrix(all_labels, all_preds)
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        classes = np.unique(all_labels)
        macro_specificity = utils.calculate_specificity(np.array(all_labels), np.array(all_preds), classes)
        avg = (macro_f1 + macro_specificity + qwk) / 3
        print(best_params['lr'])
        print(best_params['epochs'])
        print(f"Confusion Matrix:\n{cm}")
        print(f"Average Score: {avg:.4f}")
        print("ok, good, finish!!")
        break



=======
import copy
import functools

import torch
import utils
import optuna
import numpy as np
from sklearn.model_selection import ParameterGrid, StratifiedKFold, ParameterSampler
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision.transforms import transforms
from tqdm import tqdm
from model import ConvNeXtModel, GoogleModel
from dataset import MyDataset, AugmentedDataset
from optuna.integration import PyTorchLightningPruningCallback

# this is the third way of training models, using corss validation to test the model
# use 4-fold corss validation here
# also use
def nested_train(model, paths, labels, criterion, optimizer, num_epochs=10, k_outer=4, k_inner=4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # divide the dataset, use StratifiedKFold to make sure the data distribution is similar
    outer_kfold = StratifiedKFold(n_splits=k_outer, shuffle=True, random_state=42)
    inner_kfold = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=42)

    dataset = MyDataset(paths, labels, transform=transform)

    # set hyperparameters
    # (due to the computation power, not able to tune too many parameters)
    param = {
        'lr': np.logspace(-6, -2, 100),
        'epoch': [5, 10, 15, 20, 25]
    }
    n_iter = 10
    param_sampler = ParameterSampler(param, n_iter=n_iter)

    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(range(len(dataset)), labels)):
        outer_train_dataset = Subset(dataset, outer_train_idx)
        outer_test_dataset = Subset(dataset, outer_test_idx)

        best_avg = -np.inf
        best_params = []

        for params in param_sampler:
            current_avg_scores = []
            for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(inner_kfold.split(range(len(outer_train_dataset)), [labels[i] for i in outer_train_idx])):

                # no data augmentation
                # inner_val = DataLoader(Subset(outer_train_dataset, inner_val_idx), batch_size=16)
                # inner_train = DataLoader(Subset(outer_train_dataset, inner_train_idx), batch_size=16, shuffle=True)

                # define the label that needs to be enhanced and how many times it should be enhanced
                training_set = Subset(outer_train_dataset, inner_train_idx)
                augmented_images, augmented_labels = augment_dataset(training_set, train_transform)
                augmented_dataset = AugmentedDataset(augmented_images, augmented_labels)
                inner_train = DataLoader(augmented_dataset, batch_size=16, shuffle=True)
                inner_val = DataLoader(Subset(outer_train_dataset, inner_val_idx), batch_size=16)

                model = ConvNeXtModel().to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=params['lr'], weight_decay=1e-4)

                for epoch in range(params['epoch']):
                    train_epoch(model, inner_train, criterion, optimizer, device)

                all_labels, all_preds = validate_epoch(model, inner_val, criterion, device)

                cm = confusion_matrix(all_labels, all_preds)
                qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
                macro_f1 = f1_score(all_labels, all_preds, average='macro')
                classes = np.unique(all_labels)
                macro_specificity = utils.calculate_specificity(np.array(all_labels), np.array(all_preds), classes)
                avg = (macro_f1 + macro_specificity + qwk) / 3
                current_avg_scores.append(avg)
                print(params)
                print(f"Confusion Matrix:\n{cm}")
                print(f"Average Score: {avg:.4f}")
                if avg < 0.78:
                    print("too small, break")
                    break

            # calculates the average performance under the current parameter configuration
            current_params_avg = np.mean(current_avg_scores)
            print(f"Total Average Score: {current_params_avg:.4f}")

            # save the best result
            if current_params_avg > best_avg:
                best_avg = current_params_avg
                best_params = params

        print("Best Parameters:", best_params)

        # Evaluation on the outer test set
        test_loader = torch.utils.data.DataLoader(outer_test_dataset, batch_size=16, shuffle=False)
        model = ConvNeXtModel().to(device)  # use new ConvNeXtModel
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)

        # retrain the model to use the best parameters
        for epoch in range(best_params['epoch']):  # use best_params
            train_epoch(model, DataLoader(outer_train_dataset, batch_size=16, shuffle=True), criterion, optimizer,
                        device)
        all_labels, all_preds = validate_epoch(model, test_loader, criterion, device)

        # print the result
        test_cm = confusion_matrix(all_labels, all_preds)
        test_qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        test_macro_f1 = f1_score(all_labels, all_preds, average='macro')
        test_macro_specificity = utils.calculate_specificity(np.array(all_labels), np.array(all_preds),
                                                             np.unique(all_labels))
        test_avg = (test_macro_f1 + test_macro_specificity + test_qwk) / 3

        print("////////////////////////////////////////////////////////////////////////")
        print(f"Test Metrics with Best Parameters:")
        print(f"Confusion Matrix:\n{test_cm}")
        print(f"Quadratic Weighted Kappa: {test_qwk:.4f}")
        print(f"Macro F1-Score: {test_macro_f1:.4f}")
        print(f"Macro Specificity: {test_macro_specificity:.4f}")
        print(f"Average Score: {test_avg:.4f}")
        print("////////////////////////////////////////////////////////////////////////")

# simplify the train function
def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

# simplify the train function
def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, paths in tqdm(data_loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return all_labels, all_preds


# also define preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# also define preprocessing transform for training set
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.Resize(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])


def augment_dataset(dataset, transform):
    augmented_images = []
    augmented_labels = []

    for i in range(len(dataset)):
        img, label, path = dataset[i]

        # add original image
        augmented_images.append(img)
        augmented_labels.append(label)

        # if label in [0]:
        #     for _ in range(1):
        #         augmented_img = transform(img)
        #         augmented_images.append(augmented_img)
        #         augmented_labels.append(label)
        # if label in [1]:
        #     for _ in range(1):
        #         augmented_img = transform(img)
        #         augmented_images.append(augmented_img)
        #         augmented_labels.append(label)
        if label in [2]:
            for _ in range(1):
                augmented_img = transform(img)
                augmented_images.append(augmented_img)
                augmented_labels.append(label)
        if label in [3]:
            for _ in range(5):
                augmented_img = transform(img)
                augmented_images.append(augmented_img)
                augmented_labels.append(label)
        if label in [4]:
            for _ in range(7):
                augmented_img = transform(img)
                augmented_images.append(augmented_img)
                augmented_labels.append(label)

    return augmented_images, augmented_labels

# this functino is to create an optuna parameter search work
# very time-consuming (not give the best result)
def objective(trial, labels, outer_train_dataset, outer_train_idx):
    # define the hyperparameter space
    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    epochs = trial.suggest_int('epochs', 1, 50)
    criterion = nn.CrossEntropyLoss()
    scores = []

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    inner_kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            inner_kfold.split(range(len(outer_train_dataset)), [labels[i] for i in outer_train_idx])):

        # use divided data
        training_set = Subset(outer_train_dataset, inner_train_idx)
        augmented_images, augmented_labels = augment_dataset(training_set, train_transform)
        augmented_dataset = AugmentedDataset(augmented_images, augmented_labels)
        inner_train = DataLoader(augmented_dataset, batch_size=16, shuffle=True)
        inner_val = DataLoader(Subset(outer_train_dataset, inner_val_idx), batch_size=16)

        # use a new model (avoid using the same model for different trails)
        model = ConvNeXtModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in range(epochs):
            train_epoch(model, inner_train, criterion, optimizer, device)
        all_labels, all_preds = validate_epoch(model, inner_val, criterion, device)

        # evaluate
        cm = confusion_matrix(all_labels, all_preds)
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        classes = np.unique(all_labels)
        macro_specificity = utils.calculate_specificity(np.array(all_labels), np.array(all_preds), classes)
        avg = (macro_f1 + macro_specificity + qwk) / 3
        scores.append(avg)
        print(lr)
        print(epochs)
        print(f"Confusion Matrix:\n{cm}")
        print(f"Average Score: {avg:.4f}")

    average_score = sum(scores) / len(scores)
    print(f"Final Score: {average_score:.4f}")
    return average_score


# train using optuna (this is too time-consuming, did not give the best answer)
def optuna_train(paths, labels, criterion):
    # create dataset instance
    dataset = MyDataset(paths, labels, transform=transform)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # process for training and validation
    outer_kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    for outer_fold, (outer_train_idx, outer_test_idx) in enumerate(outer_kfold.split(range(len(dataset)), labels)):
        outer_train_dataset = Subset(dataset, outer_train_idx)
        outer_test_dataset = Subset(dataset, outer_test_idx)

        augmented_images, augmented_labels = augment_dataset(outer_train_dataset, train_transform)
        augmented_dataset = AugmentedDataset(augmented_images, augmented_labels)
        outer_train = DataLoader(augmented_dataset, batch_size=16, shuffle=True)
        outer_test = DataLoader(outer_test_dataset, batch_size=16)

        # partial_objective = functools.partial(objective, labels=labels, outer_train_dataset=outer_train_dataset,
        #                                       outer_train_idx=outer_train_idx)
        # we need the result to be larger
        # study = optuna.create_study(direction='maximize')
        # study.optimize(partial_objective, n_trials=12)
        # utils.visualise_optuna(study)
        #
        # choose 3 best trail (because I want to test not only the best result, to prove it can )
        # sorted_trials = sorted(study.trials, key=lambda trial: trial.value, reverse=True)
        # best_three_trials = sorted_trials[:3]
        # print(best_three_trials)
        #

        # this is used to test the best settings found, can be deleted
        # best_params = study.best_trial.params
        best_params = {'lr': 2.7795834117065912e-05, 'epochs': 38}

        model = ConvNeXtModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)

        for epoch in range(best_params['epochs']):
            train_epoch(model, outer_train, criterion, optimizer, device)
        all_labels, all_preds = validate_epoch(model, outer_test, criterion, device)

        # evaluate
        cm = confusion_matrix(all_labels, all_preds)
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        classes = np.unique(all_labels)
        macro_specificity = utils.calculate_specificity(np.array(all_labels), np.array(all_preds), classes)
        avg = (macro_f1 + macro_specificity + qwk) / 3
        print(best_params['lr'])
        print(best_params['epochs'])
        print(f"Confusion Matrix:\n{cm}")
        print(f"Average Score: {avg:.4f}")
        print("ok, good, finish!!")
        break



>>>>>>> db3e9e707d5fd9ad3c06f9b58016989e2e363368
