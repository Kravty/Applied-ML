# external lib imports
from omegaconf import DictConfig
import torch
from torch import nn
import torchvision


def freeze_backbone(model: torch.nn.Module, unfreezed_layer: torch.nn.Module
                    ) -> torch.nn.Module:
    for param in model.parameters():
            param.requires_grad = False
    for param in unfreezed_layer.parameters():
            param.requires_grad = True
    
    return model


def get_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    """
    Function loads pre-trained (on ImageNet dataset) model from torchvision and change last layer to num classes size.

    Args:
        cfg: configuration file loaded by Hydra
        device: device (torch.device("cuda") or "cpu")

    Returns:
        pretrained_model on device with last linear layer output size corresponding to numer of dataset classes
    """

    torch.manual_seed(cfg.params.seed)  # random state for reproducibility
    # change number of output classes
    if cfg.model.model_type.lower() == 'mobilenet_v3_l':
        model = torchvision.models.mobilenet_v3_large(weights="DEFAULT")
        num_in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_in_features, cfg.dataset.num_classes)
        nn.init.xavier_uniform_(model.classifier[-1].weight)  # Xavier weights initialization for the new layer
        model = freeze_backbone(model=model, unfreezed_layer=model.classifier[-1])
    elif cfg.model.model_type.lower() == 'efficientnet_b0':
        model = torchvision.models.efficientnet_b0(weights="DEFAULT")
        num_in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_in_features, cfg.dataset.num_classes)
        nn.init.xavier_uniform_(model.classifier[-1].weight)  # Xavier weights initialization for the new layer
        model = freeze_backbone(model=model, unfreezed_layer=model.classifier[-1])
    elif cfg.model.model_type.lower() == 'shufflenet_v2_x2_0':
        model = torchvision.models.shufflenet_v2_x2_0(weights="DEFAULT")
        last_layer = model.fc
        num_in_features = model.fc.in_features
        model.fc = nn.Linear(num_in_features, cfg.dataset.num_classes)
        nn.init.xavier_uniform_(model.fc.weight)  # Xavier weights initialization for the new layer
        model = freeze_backbone(model=model, unfreezed_layer=model.fc)
    else:
        raise ValueError('Model not available')

    model = model.to(device)

    return model
