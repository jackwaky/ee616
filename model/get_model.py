from .cnn import CNN
from .nn import NN
import torchvision.models as models
from .models import ModelFedCon

def get_model(model_name, args):
    conv_channels = -1
    print(f"Model architecture : {model_name}")

    if model_name == "lenet":
        raise DeprecationWarning

    elif model_name == "resnet18":
        model = ModelFedCon(base_model='resnet18', out_dim=256, n_classes=10)
        return model
    elif model_name == 'cnn':
        model = ModelFedCon(base_model='simple-cnn-mnist', out_dim=256, n_classes=10)
        return model

    else:
        raise NotImplementedError