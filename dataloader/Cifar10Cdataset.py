import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset


CORRUPTIONS = ("shot_noise", "motion_blur", "snow", "pixelate",
               "gaussian_noise", "defocus_blur", "brightness", "fog",
               "zoom_blur", "frost", "glass_blur", "impulse_noise", "contrast",
               "jpeg_compression", "elastic_transform")

domain_label_dict = {domain: i+1 for i, domain in enumerate(CORRUPTIONS)}

class CIFAR10C_Dataset(torch.utils.data.Dataset):

    def __init__(self, train=True, domain=None, severity=5, transform=None, file_path = '/mnt/sting/chahh/TTA/dataset/CIFAR-10-C'):
        self.domain = domain
        self.img_shape = 32
        self.features = None
        self.class_labels = None
        self.domain_labels = None
        self.file_path = file_path
        self.transform = transform

        assert (len(domain) > 0)
        if domain=='original':
            self.sub_path1 = 'origin'
            self.sub_path2 = ''
            self.data_filename = 'original.npy'
            self.label_filename = 'labels.npy'
        else:
            self.sub_path1 = 'corrupted'
            self.sub_path2 = 'severity-'+str(severity)
            self.data_filename = domain+'.npy'
            self.label_filename = 'labels.npy'
        
            if not train:
                self.data_filename = 'test.npy'
                self.label_filename = 'labels.npy'

        self.preprocessing()

    def preprocessing(self):
        if self.sub_path2 == '':
            path = f'{self.file_path}/{self.sub_path1}/'
        else:
            path = f'{self.file_path}/{self.sub_path1}/{self.sub_path2}/'

        data = np.load(path + self.data_filename)
        # change NHWC to NCHW format
        data = np.transpose(data, (0, 3, 1, 2))
        # make it compatible with our models (normalize)
        data = data.astype(np.float32)/ 255.0
        self.features = data
        self.class_labels = np.load(path + self.label_filename)
        # assume that single domain is passed as List
        if self.domain=='original' or self.domain=='test':
            self.domain_labels = np.array([0 for i in range(len(self.features))])
        else:
            self.domain_labels = np.array([domain_label_dict[self.domain] for i in range(len(self.features))])

        self.dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(self.features),
            torch.from_numpy(self.class_labels),
            torch.from_numpy(self.domain_labels))

    def __len__(self):
        return len(self.dataset)

    def get_num_domains(self):
        return len(self.domains)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        img, cl, dl = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, cl, dl
    
    
def CIFAR10C(**kwargs):
    data_list = []
    class_labels_list = []
    domain_labels_list = []
    
    ori_dataset = CIFAR10C_Dataset(domain='original', **kwargs)
    # To match the # balance between ori - corruption samples
    random_indices = np.random.choice(len(ori_dataset), size=10000, replace=False)    
    data_list.append(ori_dataset.features[random_indices])
    class_labels_list.append(ori_dataset.class_labels[random_indices])
    domain_labels_list.append(ori_dataset.domain_labels[random_indices])

    for corruption in CORRUPTIONS:
        corrupted_dataset = CIFAR10C_Dataset(domain=corruption, **kwargs)
        data_list.append(corrupted_dataset.features)
        class_labels_list.append(corrupted_dataset.class_labels)
        domain_labels_list.append(corrupted_dataset.domain_labels)

    combined_data = torch.cat([torch.from_numpy(data) for data in data_list], dim=0)
    combined_class_labels = torch.cat([torch.from_numpy(class_label) for class_label in class_labels_list], dim=0)
    combined_domain_labels = torch.cat([torch.from_numpy(domain_label) for domain_label in domain_labels_list], dim=0)

    combined_dataset = torch.utils.data.TensorDataset(combined_data, combined_class_labels, combined_domain_labels)

    return combined_dataset