"""Pascal VOC Semantic Segmentation Dataset."""
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from .segbase import SegmentationDataset
import cv2
# import gdal
from osgeo import gdal
os.environ.setdefault('OPENCV_IO_MAX_IMAGE_PIXELS', '2000000000')
import torchvision.transforms as transforms
class VOCJibutiSegmentation(SegmentationDataset):
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
    #BASE_DIR = 'VOC_YJS'
    BASE_DIR = 'VOC-yingjisha-256'
    NUM_CLASS = 16


    def __init__(self, root='../VOC/', split='train', mode=None, transform=None, **kwargs):
        super(VOCJibutiSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        _voc_root = os.path.join(root, self.BASE_DIR)
        _mask_dir = os.path.join(_voc_root, 'SegmentationClass')
        #现在增加image feature ，将来再增加光谱信息
        #9月1日增加了光谱解译的知识
        _image_NDWI_dir = os.path.join(_voc_root,"NDWIpatch")
        _image_NDSI_dir = os.path.join(_voc_root, "NDSIpatch")
        _image_dir = os.path.join(_voc_root, 'JPEGImages')
        # train/val/test splits are pre-cut
        _splits_dir = os.path.join(_voc_root, 'ImageSets/Segmentation')
        if split == 'train':
            _split_f = os.path.join(_splits_dir, 'train.txt')
        elif split == 'val':
            _split_f = os.path.join(_splits_dir, 'val.txt')
        elif split == 'test':
            _split_f = os.path.join(_splits_dir, 'test.txt')
        else:
            raise RuntimeError('Unknown dataset split.')

        self.images = []
        #image feature list 增加了NDWI和NDSI的两个参数
        self.images_NDWI = []
        self.images_NDSI = []
        self.masks = []
        with open(os.path.join(_split_f), "r") as lines:
            for line in lines:
                _image = os.path.join(_image_dir, line.rstrip('\n') + ".tif")
                assert os.path.isfile(_image)
                self.images.append(_image)
                #iamge feature
                #_image_feat = os.path.join(_image_feat_dir,line.rstrip('\n')+'.tif')
                #self.images_features.append(_image_feat)
                #assert os.path.isfile(_image_feat)
                #image NDSI
                _image_NDSI = os.path.join(_image_NDSI_dir, line.rstrip('\n')+'.tif')
                assert  os.path.isfile(_image_NDSI)
                self.images_NDSI.append(_image_NDSI)
                #image NDWI
                _image_NDWI = os.path.join(_image_NDWI_dir,line.rstrip('\n')+'.tif')
                assert os.path.isfile(_image_NDWI)
                self.images_NDWI.append(_image_NDWI)
                if split != 'test':
                    _mask = os.path.join(_mask_dir, line.rstrip('\n') + ".tif")
                    assert os.path.isfile(_mask)
                    self.masks.append(_mask)

        if split != 'test':
            assert (len(self.images) == len(self.masks))
        print('Found {} images in the folder {}'.format(len(self.images), _voc_root))

    def __getitem__(self, index):
        #print( "image file path is %s "% self.images[index])
        img = Image.open(self.images[index]).convert('RGB')
        #image_feature
        #print("image NDWI file path is %s " % self.images_NDWI[index])
        img_NDWI = gdal.Open(self.images_NDWI[index]).ReadAsArray()
        #print("image NDSI file path is %s " % self.images_NDSI[index])
        img_NDSI = gdal.Open(self.images_NDSI[index]).ReadAsArray().astype(np.float32)
        img_feat = np.dstack((img_NDSI,img_NDWI))#.transpose(2,0,1)
        #plt.figure("jibuti")
        #plt.imshow(img_feat)
        #plt.show()
        if self.mode == 'test':
            img = self._img_transform(img)
            img_feat = self._img_transform(img_feat)
            if self.transform is not None:
                img = self.transform(img)
                #img feature transform
                #img_feat =self.transform(img_feat)
            # 需要修改
            return img,img_feat, os.path.basename(self.images[index])
        mask_gdal = gdal.Open(self.masks[index])
        mask = mask_gdal.GetRasterBand(1).ReadAsArray()
        #mask = Image.open(self.masks[index])
        # synchronized transform
        if self.mode == 'train':
            #img, mask = self._sync_transform_tif(img, mask)
            img, mask, img_feat = self._sync_transform_tif_geofeat(img, mask, img_feat)

        elif self.mode == 'val':
            #img, mask = self._val_sync_transform_tif(img, mask)
            img, mask, img_feat = self._sync_transform_tif_geofeat(img, mask, img_feat)
        else:
            assert self.mode == 'testval'
            #img, mask = self._img_transform(img), self._mask_transform(mask)
            #numpy 化输入
            img, mask, img_feat = self._sync_transform_tif_geofeat(img, mask, img_feat)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
            #img_feat = self.transform(img_feat)
            #img_feat = torch.from_numpy(img_feat)
        #多返回了一个img_feat
        return img, mask, transforms.ToTensor()(img_feat), os.path.basename(self.images[index])

    def __len__(self):
        return len(self.images)

    def _mask_transform(self, mask):
        target = np.array(mask).astype('int32')
        target[target == 255] = -1
        #target[target == 255] = -1
        #target[target == 40] = 1
        #target[target == 80] = 2
        #target[target == 120] = 3
        #target[target == 160] = 4
        target[target == 16] = 0
        return torch.from_numpy(target).long()

    @property
    def classes(self):
        """Category names."""
        return ('0', '1', '2', '3', '4', '5','6','7','8','9','10','11','12','13','14','15')


if __name__ == '__main__':
    dataset = VOCJibutiSegmentation()