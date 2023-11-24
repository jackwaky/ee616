from .cnn import CNN
from .nn import NN
import torchvision.models as models

def get_model(model_name, args):
    conv_channels = -1
    print(f"Model architecture : {model_name}")

    if model_name == "lenet":
        raise DeprecationWarning

    elif model_name == "resnet18":
        return models.resnet18(num_classes=10, zero_init_residual=True)
    elif model_name == "resnet34":
        return models.resnet34(num_classes=10, zero_init_residual=True)
    elif model_name == "resnet50":
        return models.resnet50(num_classes=10, zero_init_residual=True)
    elif model_name == 'cnn':
        return CNN(args)
    elif model_name == 'nn':
        return NN(args)
    else:
        raise NotImplementedError