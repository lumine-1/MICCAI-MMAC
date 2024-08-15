import torch
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm
import xgboost as xgb
from dataset import MyDataset
from utils import calculate_specificity
from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score


# this is used when using in-training data augmentation
# (define a new one here as the prepare_train function can not be reuse in main.py)
def prepare_train_dataloaders(file_paths, labels):
    # preprocessing transformation of data loader
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # create training and validation datasets
    train_dataset = MyDataset(file_paths, labels, transform=transform)
    # create a data loader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    return train_loader

# This is the first function for training
#
def train_model(model, train_loader, val_loader, criterion, optimizer, train_file_paths, train_labels, num_epochs=50):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_avg = 0

    scheduler = ExponentialLR(optimizer, gamma=0.98)

    for epoch in range(num_epochs):

        # used when utilising in-training data augmentation
        # train_loader = prepare_train_dataloaders(train_file_paths, train_labels)
        model.train()
        running_loss = 0.0

        # create process bar using tqdm library
        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Training]', unit='batch')

        for inputs, labels, paths in train_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_progress_bar.set_postfix(train_loss=(running_loss / (train_progress_bar.n + 1)))

        # update the learning rate at the end of each epoch
        scheduler.step()

        # evaluate
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Validation]', unit='batch')

        with torch.no_grad():
            for inputs, labels, paths in val_progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_predictions += torch.sum(preds == labels.data)
                val_progress_bar.set_postfix(val_loss=(running_loss / (val_progress_bar.n + 1)))

        val_loss = running_loss / len(val_loader)
        val_acc = correct_predictions.double() / len(val_loader.dataset)

        # TTA (can use TTA and majority vote to enhance the model)
        # (Not that useful)
        # with torch.no_grad():
        #     for inputs, labels in val_progress_bar:
        #         inputs, labels = inputs.to(device), labels.to(device)
        #         batch_predictions = []
        #         # the original image and its enhanced version are predicted
        #
        # change how many time to predict here
        #         for i in range(5):
        #             if i > 0:
        #                 inputs_augmented = augmentation(inputs)
        #             else:
        #                 inputs_augmented = inputs
        #
        #             outputs = model(inputs_augmented)
        #             _, preds = torch.max(outputs, 1)
        #             batch_predictions.append(preds.cpu())
        #
        #         # majority vote is taken on each image to get the final prediction
        #         final_preds = [
        #             majority_vote([batch_predictions[0][j], batch_predictions[1][j], batch_predictions[2][j]]) for j in
        #             range(len(labels))]
        #
        #         # evaluate
        #         final_preds_tensor = torch.tensor(final_preds, device=device)
        #         loss = criterion(outputs, labels)
        #         running_loss += loss.item()
        #         correct_predictions += torch.sum(final_preds_tensor == labels.data)
        #         val_progress_bar.set_postfix(val_loss=(running_loss / (val_progress_bar.n + 1)))
        # val_loss = running_loss / len(val_loader)
        # val_acc = correct_predictions.double() / len(val_loader.dataset)
        ## TTA end

        # calculate the confusion matrix at the end of each epoch
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels, paths in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # evaluate
        cm = confusion_matrix(all_labels, all_preds)
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        macro_specificity = calculate_specificity(np.array(all_labels), np.array(all_preds), np.unique(all_labels))
        avg = (macro_f1 + macro_specificity + qwk) / 3
        if avg > best_avg:
            best_avg = avg
            best_epoch = epoch + 1

        print(f'Epoch {epoch + 1}:')
        print(f'{cm}')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        print(f'Quadratic Weighted Kappa: {qwk:.4f}')
        print(f'Macro F1-Score: {macro_f1:.4f}')
        print(f'Macro Specificity: {macro_specificity:.4f}')
        print(f'AVG:  {avg:.4f}')

    # save the weights
    model_path = 'model_weights.pth'
    torch.save(model.state_dict(), model_path)

    print('Training complete')


# this is the second training function, using in-training data augmentation
# less stable but may have better result than pre-training data augmentation
# define transformers in the function (can not use the prepare_train function in the main.py)
def augmented_train_model(model, training_file_paths, training_labels, validation_file_paths,
                          validation_labels, criterion, optimizer, num_epochs=50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_avg = 0

    # define data enhancement transformations
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(10),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transform = transforms.Compose([
        transforms.Resize(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = MyDataset(validation_file_paths, validation_labels, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=16)

    n = -1
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        # if n % 5 == 4:
        #     train_dataset = MyDataset(training_file_paths, training_labels, transform=train_transform)
        #     n = -1
        # else:
        #     train_dataset = MyDataset(training_file_paths, training_labels, transform=transform)
        #     n += 1
        train_dataset = MyDataset(training_file_paths, training_labels, transform=train_transform)

        # update the train_loader (use new data)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Training]', unit='batch')

        for inputs, labels, paths in train_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_progress_bar.set_postfix(train_loss=(running_loss / (train_progress_bar.n + 1)))

        # evaluate
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        val_progress_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Validation]', unit='batch')

        # calculate at the end of each epoch
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels, paths in val_progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                # store the results
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                correct_predictions += torch.sum(preds == labels.data)
                val_progress_bar.set_postfix(val_loss=(running_loss / (val_progress_bar.n + 1)))
        val_loss = running_loss / len(val_loader)
        val_acc = correct_predictions.double() / len(val_loader.dataset)

        # calculate evaluation matrix
        cm = confusion_matrix(all_labels, all_preds)
        qwk = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        macro_specificity = calculate_specificity(np.array(all_labels), np.array(all_preds), np.unique(all_labels))
        avg = (macro_f1 + macro_specificity + qwk) / 3

        # use best_avg to record the bast result
        if avg > best_avg:
            best_avg = avg
            model_path = f'model_weights_{best_avg}.pth'
            torch.save(model.state_dict(), model_path)

        # print the result after each epoch
        print(f'Epoch {epoch + 1}:')
        print(f'{cm}')
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}')
        print(f'Quadratic Weighted Kappa: {qwk:.4f}')
        print(f'Macro F1-Score: {macro_f1:.4f}')
        print(f'Macro Specificity: {macro_specificity:.4f}')
        print(f'Avg:  {avg:.4f}')
        print(f'Best:  {best_avg:.4f}')


# try to use TTA, model predict many times, use the majority vote to get the final result
def majority_vote(votes):
    # count the number of times each prediction occurs
    from collections import Counter
    vote_counts = Counter(votes)

    # find the most frequent predictions
    max_votes = max(vote_counts.values())
    winners = [vote for vote, count in vote_counts.items() if count == max_votes]

    # if there are multiple predictions that occur the same number of times and the most, one is chosen at random
    import random
    final_prediction = random.choice(winners)

    return final_prediction


# define data enhancement transformations
augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    # is it inconsistent with reality?
    transforms.RandomRotation(5),
    transforms.Resize(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
])

# train the model without evaluating (already know the best parameter settings)
def train_only(model, train_loader, criterion, optimizer, num_epochs=24):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs} [Training]', unit='batch')

        for inputs, labels, paths in train_progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_progress_bar.set_postfix(train_loss=(running_loss / (train_progress_bar.n + 1)))

        # only for test (see the result before the process terminate)
        # if epoch == 21:
        #     model_path = 'model_weights_22.pth'
        #     torch.save(model.state_dict(), model_path)

    model_path = 'model_weights_24.pth'  # the path and saved file name
    torch.save(model.state_dict(), model_path)

    print('Training complete')


# when parameters aare frozen, update the optimiser to not update them
def update_optimizer(optimizer, model):
    optimizer.param_groups.clear()  # clear existing parameter groups
    optimizer.add_param_group({'params': filter(lambda p: p.requires_grad, model.parameters())})
