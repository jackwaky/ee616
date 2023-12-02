import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset

CORRUPTIONS = ('spatter', 'dotted_line', 'zigzag', 'scale', 'translate',
              'brightness', 'motion_blur', 'rotate', 'canny_edges', 'shear',
              'stripe', 'identity', 'glass_blur', 'fog', 'impulse_noise',
              'shot_noise') # 16

domain_label_dict = {domain: i+1 for i, domain in enumerate(CORRUPTIONS)}

class MNIST_Dataset(torch.utils.data.Dataset):
    def __init__(self, train=True, domain=None, transform=None, file_path='/home/twinklesu/mnist_c/mnist_c'):
        self.domain = domain
        self.img_shape = 32
        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = file_path
        self.transform = transform

        assert (len(domain) > 0)
        self.sub_path = domain
        self.data_filename = 'train_images.npy'
        self.label_filename = 'train_labels.npy'

        if not train:
            self.data_filename = 'test_images.npy'
            self.label_filename = 'test_labels.npy'

        self.preprocessing()

    def preprocessing(self):
        path = f'{self.file_path}/{self.sub_path}/'

        data = np.load(path + self.data_filename)
        # reshape to add channel dim
        data = np.expand_dims(data, axis=-1)
        data = data.astype(np.float32)/ 255.0
        self.features = data
        self.class_labels = np.load(path + self.label_filename)

        self.domain_labels = np.array([domain_label_dict[self.domain] for _ in range(len(self.features))])

        self.dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.features),
            torch.from_numpy(self.class_labels),
            torch.from_numpy(self.domain_labels)
        )

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domain)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl, dl = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, cl, dl

def MNISTC(**kwargs):
    data_list = []
    class_labels_list = []
    domain_labels_list = []

    for corruption in CORRUPTIONS:
        corrupted_dataset = MNIST_Dataset(domain=corruption, **kwargs)
        data_list.append(corrupted_dataset.features)
        class_labels_list.append(corrupted_dataset.class_labels)
        domain_labels_list.append(corrupted_dataset.domain_labels)

    combined_dataset = torch.utils.data.TensorDataset(data_list, class_labels_list, domain_labels_list)

    return combined_dataset














