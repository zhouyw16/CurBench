from .lenet import LeNet
from .vgg import VGG16
from .resnet import ResNet18, ResNet50
from .wide_resnet import WideResNet28 
from .vit import RelViT


def get_net(net_name, dataset):
    net_dict = {
        'lenet': LeNet,
        'vgg16': VGG16,
        'resnet18': ResNet18,
        'resnet50': ResNet50,
        'wideresnet28': WideResNet28,
        'vit': RelViT,
    }
    assert net_name in net_dict, \
        'Assert Error: net_name should be in ' + str(list(net_dict.keys()))

    _, _, test_dataset = dataset
    num_labels = test_dataset.num_classes
    return net_dict[net_name](num_labels=num_labels)