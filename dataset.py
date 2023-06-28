import torch
from PIL import Image
 
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
 
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import cv2 as cv


def rgb2gray(rgb):
    # print(rgb.shape)
    bn_imgs = rgb[:,:,0]*0.299 + rgb[:,:,1]*0.587 + rgb[:,:,2]*0.114
    # print(bn_imgs.shape)
    bn_imgs = np.reshape(bn_imgs,(rgb.shape[0],rgb.shape[1], 1))
    return bn_imgs

def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs-imgs_mean)/imgs_std
    
    imgs_normalized = ((imgs_normalized - np.min(imgs_normalized)) / (np.max(imgs_normalized)-np.min(imgs_normalized)))*255

    return imgs_normalized

def clahe_equalized(imgs):
    clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8,8))
    imgs_equalized = np.empty(imgs.shape)
    imgs_equalized= clahe.apply(np.array(imgs, dtype = np.uint8))
    
    return imgs_equalized

def adjust_gamma(imgs, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    new_imgs = cv.LUT(np.array(imgs, dtype = np.uint8), table)
    
    return new_imgs


def getimages(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info

def compute_mean_and_std(path):
    # 输入PyTorch的dataset，输出均值和标准差
    mean_r = 0
    mean_g = 0
    mean_b = 0
    print("计算均值>>>")
    # print(dataset)
    imgs_info = getimages(path)
    for path in tqdm(imgs_info,ncols=80):
        img=Image.open(path[0])
        img = np.asarray(img) # change PIL Image to numpy array
        mean_b += np.mean(img[:, :, 0])
        mean_g += np.mean(img[:, :, 1])
        mean_r += np.mean(img[:, :, 2])

    mean_b /= len(imgs_info)
    mean_g /= len(imgs_info)
    mean_r /= len(imgs_info)

    diff_r = 0
    diff_g = 0
    diff_b = 0

    N = 0
    print("计算方差>>>")
    for path in tqdm(imgs_info,ncols=80):
      img=Image.open(path[0])
      img = np.asarray(img)
      diff_b += np.sum(np.power(img[:, :, 0] - mean_b, 2))
      diff_g += np.sum(np.power(img[:, :, 1] - mean_g, 2))
      diff_r += np.sum(np.power(img[:, :, 2] - mean_r, 2))

      N += np.prod(img[:, :, 0].shape)

    std_b = np.sqrt(diff_b / N)
    std_g = np.sqrt(diff_g / N)
    std_r = np.sqrt(diff_r / N)

    mean = (round(mean_b.item() / 255.0, 3), round(mean_g.item() / 255.0, 3), round(mean_r.item() / 255.0,3))
    std = (round(std_b.item() / 255.0, 3), round(std_g.item() / 255.0, 3), round(std_r.item() / 255.0, 3))
    return mean, std

 
class TeethData(Dataset):
    def __init__(self, txt_path, train_flag=True, mean=0., std=0.):
        self.imgs_info = self.get_images(txt_path)
        self.train_flag = train_flag

        transform_BZ= transforms.Normalize(
            mean=mean,
            std=std
        )

        self.train_tf = transforms.Compose([
                transforms.Resize([448, 560]),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # transform_BZ
            ])
        self.val_tf = transforms.Compose([
                transforms.ToTensor(),
            ])
 
    def get_images(self, txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split('\t'), imgs_info))
        return imgs_info
 
    def __getitem__(self, index):
        img_path, label = self.imgs_info[index]
        img = cv.imread(img_path)

        img = rgb2gray(img)
        img = dataset_normalized(img)
        img = clahe_equalized(img)
        img = adjust_gamma(img)
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_GRAY2RGB))
        
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        
        label = int(label)

        return img, label
 
    def __len__(self):
        return len(self.imgs_info)
 

def load_data(args):
    # mean, std = compute_mean_and_std("train.txt")
    mean, std = (0.523, 0.39, 0.353), (0.309, 0.276, 0.258)

    print('mean:', mean)
    print('std:', std)

    train_dataset = TeethData(args.train_path, True, mean=mean, std=std)
    print("训练集数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)
 
    test_dataset = TeethData(args.test_path, False, mean=mean, std=std)
    print("测试集数据个数：", len(test_dataset))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                               shuffle=False)
    
    return train_loader, test_loader


def load_test(args):
    test_dataset = TeethData(args.test_path, False, mean=None, std=None)
    print("测试集数据个数：", len(test_dataset))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=1,
                                               shuffle=False)
    
    return test_loader
