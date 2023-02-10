
"""Pascal VOC Semantic Segmentation Dataset."""
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
#from segbase import SegmentationDataset
from .segbase import SegmentationDataset
import cv2
# import gdal
from osgeo import gdal
import random
import torch.utils.data as data
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', '2000000000')
import torchvision.transforms as transforms




class VOCYJSSegmentation(SegmentationDataset):
    """Pascal VOC Semantic Segmentation Dataset.
    Parameters
    ----------
    root : string
        Path to VOCdevkit folder. Default is './datasets/VOCdevkit'
    split: string
        'train', 'val' or 'test'
    transform : callable, optional
        A function that transforms the image
    Examples
    --------
    >>> from torchvision import transforms
    >>> import torch.utils.data as data
    >>> # Transforms for Normalization
    >>> input_transform = transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    >>> ])
    >>> # Create Dataset
    >>> trainset = VOCSegmentation(split='train', transform=input_transform)
    >>> # Create Training Loader
    >>> train_data = data.DataLoader(
    >>>     trainset, 4, shuffle=True,
    >>>     num_workers=4)
    """
    NUM_CLASS = 13

    def __init__(self, root='../VOC/', split='train', mode=None, transform=None, **kwargs):
        super(VOCYJSSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        _voc_root = root
        txt_path = os.path.join(root, split+'.txt')
        self._mask_dir = os.path.join(_voc_root, 'masks_GE')
        self._image_dir = os.path.join(_voc_root, 'images_SE')
        self._mask_LS_dir = os.path.join(_voc_root, 'masks_LS')
        self._image_LS_dir = os.path.join(_voc_root, "images_LS")
        self.image_list = read_text(txt_path)
        self.transform_SE = transforms.Compose([transforms.ToTensor(), transforms.Normalize([.485, .456, .406,.485, .456,.406 ,.406], [.229, .224, .225,.229, .224, .225,.229]),])
        random.shuffle(self.image_list)

    def __getitem__(self, index):
        #print( "image file path is %s "% self.images[index])

        #读取两种类型的图片
        img_HR =  gdal.Open(os.path.join(self._image_dir,self.image_list[index])).ReadAsArray().transpose(1,2,0).astype(np.float32)
        img_LS =  gdal.Open(os.path.join(self._image_LS_dir, self.image_list[index])).ReadAsArray().transpose(1,2,0).astype(np.float32)
        #img_LS = cv2.resize(img_LS,(672,672),interpolation=cv2.INTER_CUBIC)
        #读取两种类型的标注
        mask_HR =  gdal.Open(os.path.join(self._mask_dir,self.image_list[index])).ReadAsArray()
        mask = gdal.Open(os.path.join(self._mask_LS_dir,self.image_list[index])).ReadAsArray()
        # synchronized transform
        #只包含两种模式： train 和 val
        if self.mode == 'train':
            #img, mask = self._sync_transform_tif(img, mask)
            img_LS, mask, img_HR = self._sync_transform_tif_geofeat(img_LS, mask, img_HR)
        elif self.mode == 'val':
            #img, mask = self._val_sync_transform_tif(img, mask)
            img_LS, mask, img_HR = self._sync_transform_tif_geofeat(img_LS, mask, img_HR)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img_HR = cv2.resize(img_HR, (448,448), interpolation = cv2.INTER_CUBIC)
            img_HR = self.transform_SE(img_HR)
            img_LS = self.transform(img_LS)
            #img_feat = torch.from_numpy(img_feat)
        #多返回了一个img_feat
        return img_LS, mask, img_HR#,transforms.ToTensor()(img_feat), os.path.basename(self.images[index])

    def __len__(self):
        return len(self.image_list)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target>12] = -1
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category names."""
        return ('0', '1', '2', '3', '4', '5','6','7','8','9','10','11','12')


def generator_list_of_imagepath(path):
    image_list = []
    for image in os.listdir(path):
        # print(path)
        # print(image)
        if not image == '.DS_Store' and 'tif' == image.split('.')[-1]:
            image_list.append(image)
    return image_list

def read_text(textfile):
    list = []
    with open(textfile, "r") as lines:
        for line in lines:
            list.append(line.rstrip('\n'))
    return list

def dataset_segmentation(textpath, imagepath, train_percent):
    image_list = generator_list_of_imagepath(imagepath)
    num = len(image_list)
    list = range(num)
    train_num = int(num * train_percent)# training set num
    train_list = random.sample(list, train_num)
    print("train set size", train_num)
    ftrain = open(os.path.join(textpath, 'train.txt'), 'w')
    fval = open(os.path.join(textpath, 'val.txt'), 'w')
    for i in list:
        name = image_list[i] + '\n'
        if i in train_list:
            ftrain.write(name)
        else:
            fval.write(name)
    ftrain.close()
    fval.close()



if __name__ == '__main__':
    #path = r'C:\Users\51440\Desktop\WLKdata\googleEarth\train\images'
    #list=generator_list_of_imagepath(path)
    #print(list)
    #切割数据集

    textpath = r'C:\Users\51440\Desktop\WLKdata\WLKdata_1111\WLKdataset'
    imagepath = r'C:\Users\51440\Desktop\WLKdata\WLKdata_1111\WLKdataset\images_GE'
    train_percent = 0.8
    dataset_segmentation(textpath,imagepath,train_percent)
    #显示各种图片

    #img=r'C:\\Users\\51440\\Desktop\\WLKdata\\WLKdata_1111\\train\\images_GE\\322.tif'
    #img = gdal.Open(img).ReadAsArray().transpose(1,2,0)
    #cv2.imshow('img', img)
    #img = Image.fromarray (img,'RGB')
    #img.show()
    #img2=r'C:\\Users\\51440\\Desktop\\WLKdata\\WLKdata_1111\\train\\images_LS\\322.tif'
    #img2 = gdal.Open(img2).ReadAsArray().transpose(1,2,0).astype(np.uint8)
    #img2 = cv2.resize(img2, (672, 672), interpolation=cv2.INTER_CUBIC)
    #img2 = Image.fromarray (img2,'RGB')
    #img2.show()
    #img3 = r'C:\\Users\\51440\\Desktop\\WLKdata\\WLKdata_1111\\train\\masks_LS\\322.tif'
    #img3 = gdal.Open(img3).ReadAsArray()
    #img3 = Image.fromarray (img3)
    #img3.show()

    #dataset和dataloader的测试

    #测试dataloader能不能用
    '''
    data_dir = r'C:/Users/51440/Desktop/WLKdata/WLKdata_1111/WLKdataset'
    input_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
    dataset_train = VOCYJSSegmentation(data_dir, 'train',mode='train',transform=input_transform, base_size=224, crop_size=224)
    dataset_val = VOCYJSSegmentation(data_dir, 'val', mode='val', transform=input_transform, base_size=224, crop_size=224)
    train_data = data.DataLoader(dataset_train, 4, shuffle=True, num_workers=4)
    test_data = data.DataLoader(dataset_val, 4, shuffle=True, num_workers=4)
    '''

