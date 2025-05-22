import os
import json
import torchvision
import numpy as np
import math
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment
from semilearn.datasets.utils import split_ossl_data, reassign_target

'''
mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]
'''
mean = [0.466, 0.471, 0.380]
std = [0.195, 0.194, 0.192]

def find_classes(directory):
    classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(path, df):
    imgs = []
    targets = []

    for i, j in df.iterrows():
        img_path = os.path.join(path,'ISIC2018_Dataset',df.iloc[i][1],f"{df.iloc[i][0]}.jpg")
        #img = Image.open(img_path)
        #if (img.mode != 'RGB'):
        #    img = img.convert("RGB")
        imgs.append(img_path)
        label = int(df.iloc[i][2])
        targets.append(label)

    return imgs, targets

class isic2018_dataset(Dataset):
    def __init__(self,path, mode='train'):
        self.path = path
        
        self.mode = mode

        classes, class_to_idx = find_classes(os.path.join(self.path, 'ISIC2018_Dataset'))

        if self.mode == 'train':
            self.df = pd.read_csv(os.path.join(path,'ISIC2018_train.csv'))
            #self.df = pd.read_csv('/data1/Medical/ECL/ISIC2018_train_6.csv')
        elif self.mode == 'valid':
            self.df = pd.read_csv(os.path.join(path,'ISIC2018_val.csv'))
        else:
            self.df = pd.read_csv(os.path.join(path,'ISIC2018_test.csv'))
            #self.df = pd.read_csv('/data1/Medical/ECL/ISIC2018_test_6.csv')
        
        imgs, targets = make_dataset(self.path, self.df)
        self.data = imgs
        self.targets = targets

    def __getitem__(self, item):
        img_path = os.path.join(self.path,'ISIC2018_Dataset',self.df.iloc[item]['category'],f"{self.df.iloc[item]['image']}.jpg")
        img = Image.open(img_path)
        if (img.mode != 'RGB'):
            img = img.convert("RGB")

        label = int(self.df.iloc[item]['label'])
        
        label = torch.LongTensor([label])
        
        return img, label
       
    def __len__(self):
        return len(list(self.df['image']))
# def image_to_tensor(image_path):
#     # Define a transformation to convert the image to a tensor
#     transform = transforms.Compose([
#         transforms.Resize((224, 224)),  # Resize the image to 224x224 (you can change the size as needed)
#         transforms.ToTensor(),          # Convert the image to a PyTorch tensor
#    ])

def image_to_scalar_array(image_path):
    # Open the image using PIL (Python Imaging Library)
    image = Image.open(image_path)

    # Convert the image to grayscale
    image_gray = image.convert('L')

    # Convert the grayscale image to a NumPy array
    image_array = np.array(image_gray)

    return image_array


def get_isic2018_openset(args, alg, name, num_labels, num_classes, data_dir='./data', pure_unlabeled=False):
    #name = name.split('_')[0]  # cifar10_openset -> cifar10
    #data_dir = os.path.join(data_dir, name.lower())
    #dset = getattr(torchvision.datasets, name.upper())
    #dset = dset(data_dir, train=True, download=False)

    #data, targets = dset.data, dset.targets

    dset = isic2018_dataset(data_dir, mode='train')
    data, targets = dset.data, dset.targets
    #data_p=data
    dset_tst= isic2018_dataset(data_dir, mode='test')
    data_p, targets_p = dset_tst.data, dset_tst.targets
    

    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to 224x224 (you can change the size as needed)
        transforms.ToTensor(),
        ])
    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std )
    ])

    seen_classes = set(range(0, 5))
    num_all_classes = 7
    #num_classes=7
    data_tens = []
    for i_dir in data_p:
        # filename = os.path.basename(i_dir)
        img = Image.open(i_dir)
        #img = img.convert('L')
        img = np.array(img)

        data_tens.append(img)

    lb_data, lb_targets, ulb_data, ulb_targets = split_ossl_data(args, data, targets, num_labels, num_all_classes,
                                                                 seen_classes, None, True)
    
    save_dir = "saved_images"
    os.makedirs(save_dir, exist_ok=True)

# Iterate over image paths
  
    # print(len(lb_data))
    # for i in len(lb_data):
        

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    if pure_unlabeled:
        seen_indices = np.where(ulb_targets < num_classes)[0]
        ulb_data = ulb_data[seen_indices]
        ulb_targets = ulb_targets[seen_indices]

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_all_classes, transform_weak, True, transform_strong, False)

    #dset = getattr(torchvision.datasets, name.upper())
    #dset = dset(data_dir, train=False, download=False)
    #dset=isic2018_dataset(data_dir,mode='test')
    test_data, test_targets = data_tens, reassign_target(targets_p, num_all_classes, seen_classes)
    seen_indices = np.where(test_targets < num_classes)[0]
    test_array=[]

    for i in seen_indices:
        element=test_data[i]
        test_array.append(element)
    #eval_dset = BasicDataset(alg, test_data[seen_indices], test_targets[seen_indices],
     #                        len(seen_classes), transform_val, False, None, False)

    eval_dset = BasicDataset(alg, test_array, test_targets[seen_indices],
                             len(seen_classes), transform_val, False, None, False)
    test_full_dset = BasicDataset(alg, test_data, test_targets, num_all_classes, transform_val, False, None, False)
    return lb_dset, ulb_dset, eval_dset, test_full_dset
