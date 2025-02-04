import torch
from torchvision import models

import re


def freeze_model(model, pattern='^.*$'):
    """freeze model parameters

    Args:
        model (torch.nn.Module): model
        re (str): regular expression to match parameter names

    Returns:
        None

    """
    for name, param in model.named_parameters():
        if re.search(pattern, name):
            param.requires_grad = False
        else:
            param.requires_grad = True

    return


def get_convnext(model_name, n_classes=2, new_classifier=False):
    """get ConvNext model with modified classifier

    Args:
        model_name (str): model name
        n_classes (int): number of classes
        new_classifier (bool): whether to use new classifier
    
    Returns:
        model: ConvNext model with modified classifier

    """
    # load model and weights
    if model_name == 'convnext_tiny':
        model = models.convnext_tiny()
        weights = torch.load('pretrained_weights/convnext_tiny.pth', weights_only=False)
    elif model_name == 'convnext_small':
        model = models.convnext_small()
        weights = torch.load('pretrained_weights/convnext_small.pth', weights_only=False)
    elif model_name == 'convnext_large':
        model = models.convnext_large()
        weights = torch.load('pretrained_weights/convnext_large.pth', weights_only=False)
    else:
        raise ValueError('Unknown model: %s' % model_name)

    model.load_state_dict(weights)

    # modify classifier
    if new_classifier:
        in_features = model.classifier[2].in_features  # 1000
        new_classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features, 512, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_classes, bias=True)
        )
        model.classifier.add_module('3', torch.nn.Linear(in_features, n_classes, bias=False))
    else:
        in_features = model.classifier[2].in_features
        model.classifier[2] = torch.nn.Linear(in_features, n_classes, bias=True)

    return model
