"""dataset.py"""

import os
import random
import numpy as np
import logging

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms


class LabelHandler(object):
    def __init__(self, labels, label_weights, class_values):
        self.labels = labels
        self._label_weights = None
        self._num_classes_torch = None
        self._num_classes_list = None
        self._class_values = None
        if labels is not None:
            self._label_weights = [torch.tensor(w) for w in label_weights]
            self._num_classes_torch = torch.tensor([len(cv) for cv in class_values])
            self._num_classes_list = [len(cv) for cv in class_values]
            self._class_values = class_values

    def get_label_weights(self, i):
        return self._label_weights[i]

    def get_num_classes(self, as_tensor=True):
        if as_tensor:
            return self._num_classes_torch
        else:
            return self._num_classes_list

    def get_class_values(self):
        return self._class_values

    def get_label(self, idx):
        if self.labels is not None:
            return torch.tensor(self.labels[idx], dtype=torch.float)
        return None

    def has_labels(self):
        if self.labels is not None:
            return True
        return False


class CustomImageFolder(ImageFolder):
    def __init__(self, root, transform, labels, label_weights, name, class_values, num_channels):
        super(CustomImageFolder, self).__init__(root, transform)
        self.indices = range(len(self))
        self._num_channels = num_channels
        self._name = name

        self.label_handler = LabelHandler(labels, label_weights, class_values)

    def name(self):
        return self._name

    def label_weights(self, i):
        return self.label_handler.get_label_weights(i)

    def num_classes(self, as_tensor=True):
        return self.label_handler.get_num_classes(as_tensor)

    def class_values(self):
        return self.label_handler.get_class_values()

    def has_labels(self):
        return self.label_handler.has_labels()

    def num_channels(self):
        return self._num_channels

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        path1 = self.imgs[index1][0]
        path2 = self.imgs[index2][0]
        img1 = self.loader(path1)
        img2 = self.loader(path2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.label_handler.has_labels():
            label1 = self.label_handler.get_label(index1)
            label2 = self.label_handler.get_label(index2)
            return img1, img2, label1, label2
        return img1, img2


class CustomNpzDataset(Dataset):
    def __init__(self, data_images, transform, labels, label_weights, name, class_values, num_channels):

        self.data_npz = data_images
        self._name = name
        self._num_channels = num_channels

        self.label_handler = LabelHandler(labels, label_weights, class_values)

        self.transform = transform
        self.indices = range(len(self))

    def name(self):
        return self._name

    def label_weights(self, i):
        return self.label_handler.get_label_weights(i)

    def num_classes(self, as_tensor=True):
        return self.label_handler.get_num_classes(as_tensor)

    def class_values(self):
        return self.label_handler.get_class_values()

    def has_labels(self):
        return self.label_handler.has_labels()

    def num_channels(self):
        return self._num_channels

    def __getitem__(self, index1):
        index2 = random.choice(self.indices)

        img1 = Image.fromarray(self.data_npz[index1] * 255)
        img2 = Image.fromarray(self.data_npz[index2] * 255)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        if self.label_handler.has_labels():
            label1 = self.label_handler.get_label(index1)
            label2 = self.label_handler.get_label(index2)
            return img1, img2, label1, label2
        return img1, img2

    def __len__(self):
        return self.data_npz.shape[0]


def get_dataloader(args):
    name = args.dset_name
    dset_dir = args.dset_dir
    batch_size = args.batch_size
    num_workers = args.num_workers
    image_size = args.image_size
    shuffle = not args.test
    droplast = not args.test
    include_labels = args.include_labels

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), ])
    labels = None
    label_weights = None

    # check if labels are provided as indices or names
    label_idx = None
    label_names = None
    class_values = None
    if include_labels is not None:
        try:
            int(include_labels[0])
            label_idx = [int(s) for s in include_labels]
        except ValueError:
            label_names = include_labels
    logging.info('include_labels: {}'.format(include_labels))

    if name.lower() == 'celeba':
        root = os.path.join(dset_dir, 'CelebA')
        labels_file = os.path.join(root, 'list_attr_celeba.csv')

        # celebA images are properly numbered, so the order should remain intact in loading
        labels = None
        if label_names is not None:
            labels = []
            labels_all = np.genfromtxt(labels_file, delimiter=',', names=True)
            for label_name in label_names:
                labels.append(labels_all[label_name])
            labels = np.array(labels).transpose()
        elif label_idx is not None:
            labels_all = np.genfromtxt(labels_file, delimiter=',', skip_header=True)
            labels = labels_all[:, label_idx]

        if labels is not None:
            # celebA labels are all binary with values -1 and +1
            labels[labels == -1] = 0
            from pathlib import Path
            num_l = labels.shape[0]
            num_i = len(list(Path(root).glob('**/*.jpg')))
            assert num_i == num_l, 'num_images ({}) != num_labels ({})'.format(num_i, num_l)

            # calculate weight adversely proportional to each class's population
            num_labels = labels.shape[1]
            label_weights = []
            for i in range(num_labels):
                ones = labels[:, i].sum()
                prob_one = ones / labels.shape[0]
                label_weights.append([prob_one, 1 - prob_one])
            label_weights = np.array(label_weights)

            # all labels in celebA are binary
            class_values = [[0, 1]] * num_labels

        data_kwargs = {'root': root,
                       'labels': labels,
                       'transform': transform,
                       'label_weights': label_weights,
                       'class_values': class_values,
                       'name': name,
                       'num_channels': 3}
        dset = CustomImageFolder
    elif name.lower() == 'dsprites':
        root = os.path.join(dset_dir, 'dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        npz = np.load(root)

        if label_idx is not None:
            labels = npz['latents_values'][:, label_idx]
            if 1 in label_idx:
                index_shape = label_idx.index(1)
                labels[:, index_shape] -= 1

            # dsprite has uniformly distributed labels
            num_labels = labels.shape[1]
            label_weights = []
            class_values = []
            for i in range(num_labels):
                unique_values, count = np.unique(labels[:, i], axis=0, return_counts=True)
                weight = 1 - count / labels.shape[0]
                if len(weight) == 1:
                    weight = np.array(1)
                else:
                    weight /= sum(weight)
                label_weights.append(np.array(weight))

                # always set label values to integers starting from zero
                unique_values_mock = np.arange(len(unique_values))
                class_values.append(unique_values_mock)
            label_weights = np.array(label_weights)

        data_kwargs = {'data_images': npz['imgs'],
                       'labels': labels,
                       'transform': transform,
                       'label_weights': label_weights,
                       'class_values': class_values,
                       'name': name,
                       'num_channels': 1}
        dset = CustomNpzDataset
    else:
        raise NotImplementedError

    data = dset(**data_kwargs)
    data_loader = DataLoader(data,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers,
                             pin_memory=True,
                             drop_last=droplast)

    logging.info('Number of samples: {}'.format(data.__len__()))

    return data_loader
