MNISTOpt = {
    'input_dim': 784,
    'num_classes': 10,
    'file_path': './data/mnist',
    'batch_size': 64,
    'learning_rate': 0.1,
    'weight_decay': 0.0001,
    'classes': [i for i in range(0, 10)],
}

CIFAR10Opt = {
    'num_classes': 10,
    'file_path': './data/cifar10',
    'batch_size': 64,
    'learning_rate': 0.1,
}