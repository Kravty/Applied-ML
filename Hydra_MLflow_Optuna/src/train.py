# standard lib imports
import copy
from typing import List, Tuple

# external lib imports
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# internal src files imports
from .tracking import MLflowLogger


def train(cfg: DictConfig, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer,
          train_loader: DataLoader[List[torch.Tensor]], device: torch.device) -> np.ndarray:
    """
    Function for model training using train data.

    Args:
        cfg: configuration file loaded by Hydra
        model: model to be trained
        criterion: criterion used to calculate loss
        optimizer: optimizer object
        train_loader: DataLoader object with train data
        device: torch.device (cuda or cpu)
    Returns:
        train loss of the model
    """
    model.train()
    losses = []
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        # use mixed precision to speed up training
        with torch.autocast(device_type=cfg.params.device):
            outputs = model(X)
            loss = criterion(outputs, y)
            losses.append(loss.item())
            optimizer.zero_grad()  # Zero the gradients accumulated by PyTorch
            # Backward and optimize
            loss.backward()
            optimizer.step()

    # Taking mean value of previous losses as loss per epoch
    loss = np.mean(losses)

    return loss


def test(model: nn.Module, criterion: nn.Module, test_loader: DataLoader[List[torch.Tensor]], device: torch.device
         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Function for model validate or test model on unseen data using.

    Args:
        model: model to be trained
        criterion: criterion used to calculate loss
        test_loader: DataLoader object with validation/test data
        device: torch.device (cuda or cpu)

    Returns:
        loss value as well as preds and labels to calculate metrics in main training loop
    """

    model.eval()
    losses = []
    preds, gt = [], []  # will be used for accuracy and confusion matrix

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)
            losses.append(loss.item())
            # save predictions and ground truths for evaluation
            softmax_outputs = torch.nn.functional.softmax(input=outputs, dim=1)
            softmax_vals, indices = softmax_outputs.max(1)
            y = y.detach().cpu().numpy()
            pred = indices.detach().cpu().numpy()
            if len(preds) == 0:
                preds.append(pred)
                gt.append(y)
            else:
                preds[0] = np.append(preds[0], pred, axis=0)
                gt[0] = np.append(gt[0], y, axis=0)

    preds = np.concatenate(np.array(preds), axis=0)
    gt = np.concatenate(np.array(gt), axis=0)
    # Taking mean value of previous losses as loss per epoch
    loss = np.mean(losses)

    return loss, preds, gt


def fit_model(cfg: DictConfig, pretrained_model: nn.Module, train_loader: DataLoader[List[torch.Tensor]],
              dev_loader: DataLoader[List[torch.Tensor]], test_loader: DataLoader[List[torch.Tensor]],
              device: torch.device) -> float:
    """
    Function with main training loop with training, validation and testing of the model.

    Args:
        cfg: configuration file loaded by Hydra
        pretrained_model: model to be trained
        train_loader: DataLoader object with train data
        dev_loader: DataLoader object with dev data
        test_loader: DataLoader object with test data
        device: torch.device (cuda or cpu)

    Returns:
        Test accuracy (required for Optuna hyperparameter automatic tuning)
    """

    # init metrics dict for experiment tracking
    metrics = dict(
        train_loss=[],
        dev_loss=[],
        dev_accuracy=[],
        test_accuracy=None
    )
    best_optim_metric = 0
    tracking_data_logger = MLflowLogger(cfg=cfg)

    # copy pre-trained model to start each run from same starting point
    model = pretrained_model # copy.deepcopy(pretrained_model)

    # init criterion, optimizer and scheduler
    criterion = instantiate(cfg.model.criterion)
    optimizer = instantiate(cfg.optimizer, params=model.parameters(), lr=cfg.params.learning_rate)
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    epoch_iterator = tqdm(range(cfg.params.epochs), desc="Epoch X: train_loss=X, dev_loss=X, dev_acc=X",
                          bar_format="{l_bar}{r_bar}", dynamic_ncols=True, disable=False)

    for epoch in epoch_iterator:
        train_loss = train(cfg=cfg, model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader,
                           device=device)
        dev_loss, dev_preds, dev_gt = test(model=model, criterion=criterion, test_loader=dev_loader,
                                           device=device)
        dev_accuracy = (dev_preds == dev_gt).sum().item() / len(dev_gt)
        # Track metrics
        metrics["train_loss"].append(train_loss)
        metrics["dev_loss"].append(dev_loss)
        metrics["dev_accuracy"].append(dev_accuracy)
        epoch_iterator.set_description(
          f"Epoch {epoch + 1}: train_loss={train_loss:.5f}, dev_loss={dev_loss:.5f}, dev_acc={dev_accuracy:.3f}")

        # save checkpoint if optimizing metric improved
        if dev_accuracy > best_optim_metric:
            state_dict = dict(
                epoch=epoch+1,
                state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                )
            tracking_data_logger.log_model_to_mlflow(state_dict=state_dict, model=model)
            # torch.save(state_dict, cfg.paths.model_checkpoint)
            best_optim_metric = dev_accuracy  # update best_optim_metric

        # adjust learning rate
        scheduler.step()

    # run best model on test set
    loaded_model = tracking_data_logger.load_logged_model()
    loaded_model.to(device=device)
    # checkpoint = torch.load(cfg.paths.model_checkpoint, map_location=device)
    # model.load_state_dict(checkpoint["state_dict"])
    test_loss, test_preds, test_gt = test(model=loaded_model, criterion=criterion, test_loader=test_loader,
                                          device=device)
    test_accuracy = (test_preds == test_gt).sum().item() / len(test_gt)
    metrics["test_accuracy"] = test_accuracy
    metrics["test_loss"] = test_loss

    # log metrics and parameters
    tracking_data_logger.log_to_mlflow(metrics=metrics)

    return test_accuracy
