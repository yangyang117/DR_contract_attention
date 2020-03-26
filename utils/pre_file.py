import glob
import numpy as np
import os
import cv2
import pandas as pd


files = os.listdir('/home/yyang2/data/yyang2/Data/EyeQ-master')
print('Label_EyeQ.csv' in files) #Is the labels csv in the directory?

base_image_dir = '/home/yyang2/data/yyang2/Data/EyeQ-master'
df = pd.read_csv(os.path.join(base_image_dir, 'Label_EyeQ.csv'))
df = df.drop(columns=['Unnamed: 0'])
# 对文件进行重采样
df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe

df['path'] = df['image'].map(lambda x: os.path.join(base_image_dir, 'all', '{}'.format(x)))
df['exists'] = df['path'].map(os.path.exists) #Most of the files do not exist because this is a sample of the original dataset
df = df[df['exists']]
# 显示数据分布
df['DR_grade'].hist()
df.pivot_table(index='DR_grade', aggfunc=len)
df.to_csv('/home/yyang1/data/yyang1/yyang2/Data/EyeQ-master/y1_df_test.csv')

# 对数据进行分类及处理，训练集0.8，测试集和验证集都是0.1
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2)

test_df, val_df = train_test_split(val_df, test_size=0.5)

# 由于样本不均衡，对训练数据进行过采样
def balance_data(class_size,df):
    train_df = df.groupby(['DR_grade']).apply(lambda x: x.sample(class_size, replace = True)).reset_index(drop = True)
    train_df = train_df.sample(frac=1).reset_index(drop=True)
    print('New Data Size:', train_df.shape[0], 'Old Size:', df.shape[0])
    train_df['DR_grade'].hist(figsize = (10, 5))
    return train_df

train_df = balance_data(train_df.pivot_table(index='DR_grade', aggfunc=len).max().max(),df) # I will oversample such that all classes have the same number of images as the maximum
train_df['DR_grade'].hist(figsize=(10, 5))
train_df.pivot_table(index='DR_grade', aggfunc=len)

df.to_csv('/home/yyang2/data/yyang2/Data/EyeQ-master/results/contract/contract_all.csv')
train_df.to_csv('/home/yyang2/data/yyang2/Data/EyeQ-master/results/contract/contract_train_over.csv')
test_df.to_csv('/home/yyang2/data/yyang2/Data/EyeQ-master/results/contract/contract_test.csv')
val_df.to_csv('/home/yyang2/data/yyang2/Data/EyeQ-master/results/contract/contract_val.csv')

'''
# 本部分为从灰色图中提取出眼底图
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
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


def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (1024, 1024))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    return image

path_train = '/home/yyang1/data/yyang1/yyang2/Data/EyeQ-master/train'
path_test = '/home/yyang1/data/yyang1/yyang2/Data/EyeQ-master/test'

path_o = '/home/yyang2/data/yyang2/Data/EyeQ-master/vefig'

def get_image(path,path_o):
    img_list = []
    img_list = glob.glob(os.path.join(path, '*.jpeg'))
    for img in img_list:
        img_2 = load_ben_color(img)
        path_a = img.split('/')[-1]
        path_b = path.split('/')[-1]
        cv2.imwrite(path_o + '/' + path_b + '/' + path_a, img_2)


get_image(path_train, path_o)
get_image(path_test, path_o)
'''



