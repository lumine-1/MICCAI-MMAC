import os
import uuid
import torch
import random
import numpy as np
from PIL import Image
import optuna.visualization as ov
import torchvision.transforms.functional as F

# calculate specificity for the evaluation process
def calculate_specificity(y_true, y_pred, classes):
    specificity = []
    for a in classes:
        tn = np.sum((y_true != a) & (y_pred != a))
        fp = np.sum((y_true != a) & (y_pred == a))
        specificity.append(tn / (tn + fp))
    return np.mean(specificity)

# create a new image and save it in the same folder
# (It can not reuse, each time running the code, it will create new images)
def apply_random_transform_and_save(image_path, transform):
    # read and convert the images
    image = Image.open(image_path).convert('RGB')
    augmented = transform(image=np.array(image))
    transformed_image = augmented['image']
    transformed_image = F.to_pil_image(transformed_image)
    # transformed_image = transform(image)
    # generate a unique file name and save the image
    unique_filename = str(uuid.uuid4()) + '.jpg'
    transformed_image_path = os.path.join(os.path.dirname(image_path), unique_filename)
    transformed_image.save(transformed_image_path)

    return transformed_image_path


# set random seed to make the training repeatable
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



