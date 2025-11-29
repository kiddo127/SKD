# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

import torch
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from torch.utils.data import Dataset

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import numpy as np
import imageio
from PIL import Image
import scipy
from scipy import io
import matplotlib.pyplot as plt




class CUB():
    def __init__(self, root, is_train=True, data_len=None, transform=None):
        self.root = root
        self.is_train = is_train
        self.transform = transform
        img_txt_file = open(os.path.join(self.root, 'images.txt'))
        label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
        train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
        img_name_list = []
        for line in img_txt_file:
            img_name_list.append(line[:-1].split(' ')[-1])
        label_list = []
        for line in label_txt_file:
            label_list.append(int(line[:-1].split(' ')[-1]) - 1)
        train_test_list = []
        for line in train_val_file:
            train_test_list.append(int(line[:-1].split(' ')[-1]))
        train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
        test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]

        if self.is_train:
            self.train_img = [imageio.imread(os.path.join(self.root, 'images', train_file)) for train_file in
                              train_file_list[:data_len]]
            self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
            self.train_imgname = [x for x in train_file_list[:data_len]]

        if not self.is_train:
            self.test_img = [imageio.imread(os.path.join(self.root, 'images', test_file)) for test_file in
                             test_file_list[:data_len]]
            self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
            self.test_imgname = [x for x in test_file_list[:data_len]]

    def __getitem__(self, index):
        if self.is_train:
            img, target, imgname = self.train_img[index], self.train_label[index], self.train_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)
        else:
            img, target, imgname = self.test_img[index], self.test_label[index], self.test_imgname[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            if self.transform is not None:
                img = self.transform(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)



class CarsDataset(Dataset):
    def __init__(self, mat_anno, data_dir, car_names, cleaned=None, transform=None, is_train=True):
        """
        Args:
            mat_anno (string): Path to the MATLAB annotation file.
            data_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.full_data_set = io.loadmat(mat_anno)
        self.car_annotations = self.full_data_set['annotations']
        self.car_annotations = self.car_annotations[0]

        if cleaned is not None:
            cleaned_annos = []
            print("Cleaning up data set (only take pics with rgb chans)...")
            clean_files = np.loadtxt(cleaned, dtype=str)
            for c in self.car_annotations:
                if c[-1][0] in clean_files:
                    cleaned_annos.append(c)
            self.car_annotations = cleaned_annos

        data_set = []
        for annotation in self.car_annotations:
            # Check if the last element's value is 1 (assuming 1=test, 0=train)
            if is_train and annotation[-1][0][0] == 0:
                data_set.append(annotation)
            elif not is_train and annotation[-1][0][0] == 1:
                data_set.append(annotation)
        self.car_annotations = np.array(data_set)

        self.car_names = scipy.io.loadmat(car_names)['class_names']
        self.car_names = np.array(self.car_names[0])

        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.car_annotations)

    def __getitem__(self, idx):
        # img_name = os.path.join(self.data_dir, self.car_annotations[idx][-1][0])
        img_name = os.path.join(self.data_dir, self.car_annotations[idx][0][0])
        image = Image.open(img_name).convert('RGB')
        car_class = self.car_annotations[idx][-2][0][0]
        car_class = torch.from_numpy(np.array(car_class.astype(np.float32))).long() - 1
        assert car_class < 196
        
        if self.transform:
            image = self.transform(image)

        # return image, car_class, img_name
        return image, car_class

    def map_class(self, id):
        id = np.ravel(id)
        ret = self.car_names[id - 1][0][0]
        return ret

    def show_batch(self, img_batch, class_batch):
        for i in range(img_batch.shape[0]):
            ax = plt.subplot(1, img_batch.shape[0], i + 1)
            title_str = self.map_class(int(class_batch[i]))
            img = np.transpose(img_batch[i, ...], (1, 2, 0))
            ax.imshow(img)
            ax.set_title(title_str.__str__(), {'fontsize': 5})
            plt.tight_layout()



class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)


    if args.data_set == 'FLOWERS102':
        split = 'train' if is_train else 'test'
        dataset = datasets.Flowers102(
            root=args.data_path,
            split=split,
            download=True,
            transform=transform
        )
        nb_classes = 102
    elif args.data_set == 'CUB200':
        dataset = CUB(root=args.data_path, is_train=is_train, transform=transform)
        nb_classes = 200
    elif args.data_set == 'Cars196':
        dataset = CarsDataset(
            mat_anno = os.path.join(args.data_path,'cars_annos.mat'),
            data_dir=args.data_path,
            car_names=os.path.join(args.data_path,'cars_annos.mat'),
            transform=transform,
            is_train=is_train
            )
        nb_classes = 196

    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100



    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000

    elif args.data_set == 'Food101':
        split = 'train' if is_train else 'test'
        dataset = datasets.Food101(root=args.data_path, split=split, transform=transform)
        nb_classes = 101
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
