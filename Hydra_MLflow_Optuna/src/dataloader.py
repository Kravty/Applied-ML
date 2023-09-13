# standard lib imports
from typing import List, Tuple

# external lib imports
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader, Dataset


def get_data_loader(cfg: DictConfig, train_set: Dataset[List[torch.Tensor]], dev_set: Dataset[List[torch.Tensor]],
                    test_set: Dataset[List[torch.Tensor]]
                    ) -> Tuple[DataLoader[List[torch.Tensor]], DataLoader[List[torch.Tensor]],
                               DataLoader[List[torch.Tensor]]]:
    """
    Function creates DataLoader object to iterate over data samples in mini batches.

    Args:
        cfg: configuration file loaded by Hydra
        train_set: train_set Dataset object
        dev_set: dev_set Dataset object
        test_set: test_set Dataset object

    Returns:
        train/dev/test DataLoader objects to iterate over data samples in batches
    """

    train_loader = DataLoader(train_set, batch_size=cfg.params.batch_size, shuffle=True, drop_last=False)
    dev_loader = DataLoader(dev_set, batch_size=cfg.params.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=cfg.params.batch_size, shuffle=False, drop_last=False)

    return train_loader, dev_loader, test_loader
