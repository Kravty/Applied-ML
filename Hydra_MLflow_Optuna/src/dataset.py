# standard lib imports
from pathlib import Path
from typing import List, Tuple

# external lib imports
# import cv2
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


def prepare_dataset(cfg: DictConfig) -> Tuple[Dataset[List[torch.Tensor]], Dataset[List[torch.Tensor]],
                                              Dataset[List[torch.Tensor]]]:
    """
    Function to prepare torch dataset object for dataset specified in config.

    Args:
        cfg: configuration file loaded by Hydra

    Returns:
        train/dev/test datasets
    """

    # call appropriate prepare_dataset function based on configuration
    if cfg.dataset.name == "pets_facial_expression":
        train_set, dev_set, test_set = prepare_pets_facial_expression_dataset(cfg)
    elif cfg.dataset.name == "people_facial_expression":
        train_set, dev_set, test_set = prepare_people_facial_expression_dataset(cfg)
    else:
        raise KeyError(f"Incorrect dataset name: {cfg.dataset}. "
                       f"Valid names: pets_facial_expression, people_facial_expression")

    return train_set, dev_set, test_set


def prepare_pets_facial_expression_dataset(cfg: DictConfig) -> Tuple[Dataset[List[torch.Tensor]],
                                                                     Dataset[List[torch.Tensor]],
                                                                     Dataset[List[torch.Tensor]]]:
    """
    Function to prepare torch dataset object from pets facial expression dataset.

    Args:
        cfg: configuration file loaded by Hydra

    Returns:
        train/dev/test datasets
    """

    # instantiate torchvision.transforms.Compose with train and test transforms
    train_transforms = instantiate(cfg.augmentations.train_augmentations)
    test_transforms = instantiate(cfg.augmentations.test_augmentations)
    # since the dataset is split and stored in right the format for torchvision ImageFolder implementation is easy
    train_set = datasets.ImageFolder(cfg.dataset.train_data, transform=train_transforms)
    dev_set = datasets.ImageFolder(cfg.dataset.dev_data, transform=test_transforms)
    test_set = datasets.ImageFolder(cfg.dataset.test_data, transform=test_transforms)

    return train_set, dev_set, test_set


# custom dataset for facial expression dataset
class PeopleFacialExpressionDataset(Dataset):
    def __init__(self, X: List[str], y: List[int], transform: transforms.Compose = None) -> None:
        """
        Init custom torch Dataset object for training.

        Args:
            X: list of filepaths to images
            y: list of labels
            transform: Compose of torchvision transforms
        """

        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.X[index]
        image = Image.open(img_path)
        # image = cv2.imread(img_path)
        # # change color channels from cv2 BGR to RGB
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image)

        label = self.y[index]

        return image, label

    def __len__(self):
        return len(self.y)


def prepare_people_facial_expression_dataset(cfg: DictConfig) -> Tuple[Dataset[List[torch.Tensor]],
                                                                       Dataset[List[torch.Tensor]],
                                                                       Dataset[List[torch.Tensor]]]:
    """
    Function to prepare torch dataset object from pets facial expression dataset.

    Args:
        cfg: configuration file loaded by Hydra

    Returns:
        train/dev/test datasets
    """

    label_names = ("Anger", "Contempt", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprised")
    # create dictionary to convert label names into numeric labels
    label_dict = {name: idx for idx, name in enumerate(label_names)}

    # facial expression dataset is not in the correct format for ImageFolder, so we will create a custom dataset
    root_dir = Path(cfg.dataset.root_dir)
    images_directory = root_dir.joinpath(cfg.dataset.data_dir)
    # images directory has enumerated directories. Each directory contains images of person in a few different emotions
    directories = list(images_directory.iterdir())
    # dataset is not split into train/dev/test sets
    # check if train/dev/test proportions sum to 1
    assert cfg.dataset.train_dev_test_split.train + cfg.dataset.train_dev_test_split.dev + \
           cfg.dataset.train_dev_test_split.test == 1
    # first split train set and remaining data
    dev_test_set_size = cfg.dataset.train_dev_test_split.dev + cfg.dataset.train_dev_test_split.test
    train_data, dev_test_data = train_test_split(directories, test_size=dev_test_set_size, random_state=cfg.params.seed)
    # split remaining data into dev and test sets
    dev_test_ratio = cfg.dataset.train_dev_test_split.test / dev_test_set_size
    dev_data, test_data = train_test_split(dev_test_data, test_size=dev_test_ratio, random_state=cfg.params.seed)

    X_train, y_train = [], []
    X_dev, y_dev = [], []
    X_test, y_test = [], []
    for image_directories, X, y in zip([train_data, dev_data, test_data],
                                       [X_train, X_dev, X_test],
                                       [y_train, y_dev, y_test]):
        for image_directory in image_directories:
            image_filepaths = list(image_directory.iterdir())
            for image_filepath in image_filepaths:
                X.append(image_filepath)
                # split emotion name from image_filepath (remove extension suffix and dir path)
                label_name = image_filepath.stem
                label = label_dict[label_name]  # convert to numeric label
                y.append(label)

    # instantiate torchvision.transforms.Compose with train and test transforms
    train_transforms = instantiate(cfg.augmentations.train_augmentations)
    test_transforms = instantiate(cfg.augmentations.test_augmentations)

    # create train/dev/test sets as Dataset objects
    train_set = PeopleFacialExpressionDataset(X=X_train, y=y_train, transform=train_transforms)
    dev_set = PeopleFacialExpressionDataset(X=X_dev, y=y_dev, transform=test_transforms)
    test_set = PeopleFacialExpressionDataset(X=X_test, y=y_test, transform=test_transforms)

    return train_set, dev_set, test_set
