import os
from PIL import Image

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import fourier_transform
import random

from __main__ import args

#to perform augumentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    'pre-train': transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
        transforms.RandomGrayscale(p=0.2),lambda x: np.asarray(x)
        ]),
    'post-train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

# function is not used
def image_transform(image, augmenters):
    """Transform an image

    Arguments:
        image : input image
        augmenters : data_transforms

    Returns:
        image : transformed image

    """
    # image preprocessing
    image = augmenters(image)
    return image


class BaseDataset(Dataset):
    """Base dataset."""

    def __init__(self, images, labels, domain, pre_transform_train=None,post_transform_train=None, test=None):
        """
        Args:
            images (list of string): Paths of the images.
            labels (list of int): labels of the images.
            domain (int): domain label
            transform (callable, optional): Optional transform to be applied
                on an image.
        """
        self.images = images  #paths
        self.labels = labels
        self.domain = domain
        self.pre_transform_train=pre_transform_train
        self.post_transform_train=post_transform_train
        self.test=test
        #self.post_transform = fourier_transform.get_post_transform()
        self.alpha=1.0

        #self.flat_names = []
        #self.flat_labels = []
        #self.flat_domains = []
        #for i in range(len(images)):
        #   self.flat_names += images[i]
        #   self.flat_labels += labels[i]
        #   self.flat_domains += [i] * len(images[i])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = self.images[idx]
        lbls = self.labels[idx]
        domain = self.domain

        imgs = Image.open(imgs).convert('RGB')

        if self.pre_transform_train==None and self.post_transform_train==None:
            if self.test:
                imgs = self.test(imgs)
            return {'image': imgs, 'label': lbls, 'domain': domain}
        
        
        img_o = self.pre_transform_train(imgs)
        img_s, label_s = self.sample_image()
        img_s2o, img_o2s = fourier_transform.colorful_spectrum_mix(img_o, img_s, alpha=self.alpha)
        img_o, img_s = self.post_transform_train(img_o), self.post_transform_train(img_s)
        
        img_s2o, img_o2s = self.post_transform_train(img_s2o), self.post_transform_train(img_o2s)
        imgs = [img_o, img_s, img_s2o, img_o2s]
        lbls = [lbls, label_s, lbls, label_s]
        
        return {'image': imgs, 'label': lbls, 'domain': domain}

    def sample_image(self):    
        img_idx = random.randint(0, len(self.images)-1)
        imgn_ame_sampled = self.images[img_idx]
        img_sampled = Image.open(imgn_ame_sampled).convert('RGB')
        label_sampled = self.labels[img_idx]
        return self.pre_transform_train(img_sampled), label_sampled
    
def get_image_label(category_list, label_list, domain_path):
    image_list = []
    lbl_list = []
    for category, label in zip(category_list, label_list):
        image_name = sorted(os.listdir(os.path.join(domain_path, category)))
        image_list.append([os.path.join(domain_path, category, e) for e in image_name])
        lbl_list.append([label]*len(image_name))
    image_list = np.array([x for e in image_list for x in e])
    lbl_list = np.array([x for e in lbl_list for x in e])
    return image_list, lbl_list


class PACS(object):
    """PACS train/val data

    7 categories

    images are from: http://www.eecs.qmul.ac.uk/~dl307/project_iccv2017
    These images are in the folder ./input/pacs/PACS/kfold/, e.g., ./input/pacs/PACS/kfold/APR/dog/***.jpg.
    training and validation split txt files are from:
    https://github.com/DeLightCMU/RSC/tree/master/Domain_Generalization/data/correct_txt_lists
    Put these txt files in ./input/pacs/PACS/kfold/
    """

    def __init__(self):
        data_path = os.path.join(args.datadir, 'Data')
        label_path=os.path.join(args.datadir,'Labels')

        #domain_dic = {'art': 0, 'cartoon': 1, 'photo': 2, 'sketch': 3}
        domain_dic={'APR':0,'Photos':1,'Multispectral':2}
        #self.category_list = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']
        self.category_list = args.class_list.split(',')
        label_list = range(len(self.category_list))

        image_art, lbl_art = get_image_label(self.category_list, label_list, os.path.join(data_path, 'APR')) #art=apr
        print(image_art)
        image_cartoon, lbl_cartoon = get_image_label(self.category_list, label_list, os.path.join(data_path, 'Photos')) #cartoon=Photos
        image_photo, lbl_photo = get_image_label(self.category_list, label_list, os.path.join(data_path, 'Multispectral')) #photo=multispectral
        #image_sketch, lbl_sketch = get_image_label(self.category_list, label_list, os.path.join(data_path, 'sketch'))

        dataset_art = BaseDataset(image_art, lbl_art, domain_dic['APR'], None,None,data_transforms['test'])
        dataset_cartoon = BaseDataset(image_cartoon, lbl_cartoon, domain_dic['Photos'],None,None, data_transforms['test'])
        dataset_photo = BaseDataset(image_photo, lbl_photo, domain_dic['Multispectral'],None,None, data_transforms['test'])
        # dataset_sketch = BaseDataset(image_sketch, lbl_sketch, domain_dic['sketch'],None,None, data_transforms['test'])

        # datasets for each domain
        # self.datasets = {'APR': dataset_art, 'Photos': dataset_cartoon, 'Multispectral': dataset_photo, 'sketch': dataset_sketch}
        self.datasets = {'APR': dataset_art, 'Photos': dataset_cartoon, 'Multispectral': dataset_photo}

        # number of categories
        self.num_class = len(self.category_list)
        class_list_folders='+'.join(self.category_list)

        train_images_art = pd.read_csv(os.path.join(label_path, class_list_folders,'APR_train.txt'), header=None, sep=' ')
        image_train_art = data_path + '/' + train_images_art[0].values
        lbl_train_art = train_images_art[1].values - 1
        val_images_art = pd.read_csv(os.path.join(label_path, class_list_folders, 'APR_val.txt'), header=None, sep=' ')
        image_val_art = data_path + '/' + val_images_art[0].values
        lbl_val_art = val_images_art[1].values - 1

        train_images_cartoon = pd.read_csv(os.path.join(label_path, class_list_folders, 'Photos_train.txt'), header=None, sep=' ')
        image_train_cartoon = data_path + '/' + train_images_cartoon[0].values
        lbl_train_cartoon = train_images_cartoon[1].values - 1
        val_images_cartoon = pd.read_csv(os.path.join(label_path, class_list_folders, 'Photos_val.txt'), header=None, sep=' ')
        image_val_cartoon = data_path + '/' + val_images_cartoon[0].values
        lbl_val_cartoon = val_images_cartoon[1].values - 1

        train_images_photo = pd.read_csv(os.path.join(label_path, class_list_folders, 'Multispectral_train.txt'), header=None, sep=' ')
        image_train_photo = data_path + '/' + train_images_photo[0].values
        lbl_train_photo = train_images_photo[1].values - 1
        val_images_photo = pd.read_csv(os.path.join(label_path, class_list_folders, 'Multispectral_val.txt'), header=None, sep=' ')
        image_val_photo = data_path + '/' + val_images_photo[0].values
        lbl_val_photo = val_images_photo[1].values - 1

        # train_images_sketch = pd.read_csv(os.path.join(data_path, 'sketch_train_kfold.txt'), header=None, sep=' ')
        # image_train_sketch = data_path + '/' + train_images_sketch[0].values
        # lbl_train_sketch = train_images_sketch[1].values - 1
        # val_images_sketch = pd.read_csv(os.path.join(data_path, 'sketch_crossval_kfold.txt'), header=None, sep=' ')
        # image_val_sketch = data_path + '/' + val_images_sketch[0].values
        # lbl_val_sketch = val_images_sketch[1].values - 1

        dataset_train_art = BaseDataset(image_train_art, lbl_train_art, domain_dic['APR'], data_transforms['pre-train'],data_transforms['post-train'])
        dataset_train_cartoon = BaseDataset(image_train_cartoon, lbl_train_cartoon, domain_dic['Photos'], data_transforms['pre-train'],data_transforms['post-train'])
        dataset_train_photo = BaseDataset(image_train_photo, lbl_train_photo, domain_dic['Multispectral'], data_transforms['pre-train'],data_transforms['post-train'])
        # dataset_train_sketch = BaseDataset(image_train_sketch, lbl_train_sketch, domain_dic['sketch'], data_transforms['pre-train'],data_transforms['post-train'])

        dataset_val_art = BaseDataset(image_val_art, lbl_val_art, domain_dic['APR'],None,None, data_transforms['test'])
        dataset_val_cartoon = BaseDataset(image_val_cartoon, lbl_val_cartoon, domain_dic['Photos'],None,None, data_transforms['test'])
        dataset_val_photo = BaseDataset(image_val_photo, lbl_val_photo, domain_dic['Multispectral'],None,None, data_transforms['test'])
        # dataset_val_sketch = BaseDataset(image_val_sketch, lbl_val_sketch, domain_dic['sketch'],None,None, data_transforms['test'])

        # datasets for each domain
        # self.datasets_kfold = {'APR': {'train': dataset_train_art, 'val': dataset_val_art}, 'Photos': {'train': dataset_train_cartoon, 'val': dataset_val_cartoon}, 'Multispectral': {'train': dataset_train_photo, 'val': dataset_val_photo}, 'sketch': {'train': dataset_train_sketch, 'val': dataset_val_sketch}}
        self.datasets_kfold = {'APR': {'train': dataset_train_art, 'val': dataset_val_art}, 'Photos': {'train': dataset_train_cartoon, 'val': dataset_val_cartoon}, 'Multispectral': {'train': dataset_train_photo, 'val': dataset_val_photo}}

        # number of data samples
        # self.num_sample = {'APR': {'train': len(image_train_art), 'val': len(image_val_art)}, 'Photos': {'train': len(image_train_cartoon), 'val': len(image_val_cartoon)}, 'Multispectral': {'train': len(image_train_photo), 'val': len(image_val_photo)}, 'sketch': {'train': len(image_train_sketch), 'val': len(image_val_sketch)}}
        self.num_sample = {'APR': {'train': len(image_train_art), 'val': len(image_val_art)}, 'Photos': {'train': len(image_train_cartoon), 'val': len(image_val_cartoon)}, 'Multispectral': {'train': len(image_train_photo), 'val': len(image_val_photo)}}


def get_data(domain_id):
    """Return Domain object based on domain_id

    Arguments:
        domain_id (string): domain name.

    Returns:
        Domain instance or None

    """
    if domain_id == 'pacs':
        return PACS()
    return None
