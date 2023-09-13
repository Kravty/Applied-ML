import os

# external lib imports
import hydra
from omegaconf import DictConfig
import torch

# internal src files imports
from src.dataset import prepare_dataset
from src.dataloader import get_data_loader
from src.models import get_model
from src.train import fit_model


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> float:
    """

    Args:
        cfg: configuration file loaded by Hydra

    Returns:
        Test accuracy (required for Optuna hyperparameter automatic tuning)

    """
    device = torch.device("cuda") if cfg.params.device == "cuda" else "cpu"
    # prepare dataset
    train_set, dev_set, test_set = prepare_dataset(cfg=cfg)
    train_loader, dev_loader, test_loader = get_data_loader(cfg=cfg, train_set=train_set, dev_set=dev_set,
                                                            test_set=test_set)
    # get pretrained model and fine tune it
    model = get_model(cfg=cfg, device=device)
    test_accuracy = fit_model(cfg=cfg, pretrained_model=model, train_loader=train_loader, dev_loader=dev_loader,
                              test_loader=test_loader, device=device)

    return test_accuracy


if __name__ == "__main__":
    main()
