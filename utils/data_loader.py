# encoding: utf-8
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageCms
import os
from sklearn import preprocessing
import pandas as pd
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import copy

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        check_shape = img[:, :][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img


def load_eyeQ_excel(list_file, n_class=5):
    image_names = []
    labels = []
    df_tmp = pd.read_csv(list_file)
    img_num = len(df_tmp)

    for idx in range(img_num):
        image_names.append(df_tmp['path'][idx])
        label = int(df_tmp['DR_grade'][idx])
        labels.append(label)

    return image_names, labels


class DatasetGenerator(Dataset):
    def __init__(self, list_file, transform1=None, transform2=None, n_class=5, set_name='train', mode = 'normal'):

        image_names, labels = load_eyeQ_excel(list_file, n_class=5)

        self.image_names = image_names
        self.labels = labels
        self.n_class = n_class
        self.transform1 = transform1
        self.transform2 = transform2
        self.set_name = set_name
        self.mode = mode
    #     mode = 'normal' or 'attention'
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (512, 512))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 10), -4, 128)

        image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        image_hsv = cv2.resize(image_hsv, (512, 512))
        image_v = torch.tensor(image_hsv)[:,:,2].reshape(512,512,1).expand(512,512,3)
        image_v = image_v.numpy()
        image_v = Image.fromarray(image_v)
        image = Image.fromarray(image)
        if self.transform1 is not None:
            image = self.transform1(image)
            image_v = self.transform1(image_v)
        #img_hsv = image.convert("HSV")
        #img_lab = ImageCms.applyTransform(image, self.rgb2lab_transform)

        img_rgb = np.asarray(image).astype('float32')
        #img_hsv = np.asarray(img_hsv).astype('float32')
        #img_lab = np.asarray(img_lab).astype('float32')

        if self.transform2 is not None:
            img_rgb = self.transform2(img_rgb)
            #img_hsv = self.transform2(img_hsv)
            #img_lab = self.transform2(img_lab)
            image_v = self.transform2(image_v)

        if self.mode == 'normal':
            if self.set_name == 'train' or self.set_name == 'val':
                label = self.labels[index]
                return torch.FloatTensor(img_rgb), label
            else:
                return torch.FloatTensor(img_rgb)
        elif self.mode == 'attention':
            if self.set_name == 'train' or self.set_name == 'val':
                label = self.labels[index]
                return torch.FloatTensor(img_rgb), torch.FloatTensor(image_v), label
            else:
                return torch.FloatTensor(img_rgb),torch.FloatTensor(image_v)


    def __len__(self):
        return len(self.image_names)



transformList2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

transform_list_val1 = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
    ])


def get_image(image_name):

    image = Image.open(image_name).convert('RGB')
    transformList2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    transform_list_val1 = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
    ])

    image = transform_list_val1(image)


    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(srgb_profile, lab_profile, "RGB", "LAB")
    img_hsv = image.convert("HSV")
    img_lab = ImageCms.applyTransform(image, rgb2lab_transform)

    img_rgb = np.asarray(image).astype('float32')
    img_hsv = np.asarray(img_hsv).astype('float32')
    img_lab = np.asarray(img_lab).astype('float32')
    img_rgb = transformList2(img_rgb).unsqueeze(0)
    img_hsv = transformList2(img_hsv).unsqueeze(0)
    img_lab = transformList2(img_lab).unsqueeze(0)
    return torch.FloatTensor(img_rgb), torch.FloatTensor(img_hsv), torch.FloatTensor(img_lab)




