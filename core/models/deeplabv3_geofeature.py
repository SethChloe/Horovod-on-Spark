"""Pyramid Scene Parsing Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import fcn_resnet50, deeplabv3_resnet50
from .segbase import SegBaseModel
from .fcn import _FCNHead

__all__ = ['DeepLabV3_Geofeat', 'get_deeplabv3_geofeat', 'get_deeplabv3_geofeat_resnet50_voc']

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        #print(avgout.shape)
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        #print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out



class DeepLabV3_Geofeat(SegBaseModel):
    r"""DeepLabV3_Geofeat

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`nn.BatchNorm`;
        for Synchronized Cross-GPU BachNormalization).
    aux : bool
        Auxiliary loss.

    Reference:
        Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation."
        arXiv preprint arXiv:1706.05587 (2017).
    """

    def __init__(self, nclass, backbone='resnet50', aux=False, pretrained_base=True, **kwargs):
        super(DeepLabV3_Geofeat, self).__init__(nclass, aux, backbone, pretrained_base=pretrained_base, **kwargs)
        self.head = _DeepLabHead(nclass, **kwargs)
        self.pool = nn.MaxPool2d(3)
        self.conv1 = nn.Conv2d(2048, 128, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(26, 13, kernel_size=1, stride=1)
        #self.cbam1 = CBAM(2048)
        #self.cbam2 = CBAM(2080)
        #self.conv2 = nn.Conv2d(2048, 2048, kernel_size=3, stride=2, padding=1)
        if self.aux:
            self.auxlayer = _FCNHead(1024, nclass, **kwargs)

        self.__setattr__('exclusive', ['head', 'auxlayer'] if aux else ['head'])

    def forward(self, x_ls, lbr=None):
        #这里的lbr是多种特征了
        #landsat
        size = x_ls.size()[2:] #（[224,224]）
        _, _, _, x = self.base_forward(x_ls)  #(2048,28,28)
        outputs = []
        #GE
        if lbr is not None:
            #lbr =self.pool(lbr)
            #x_ls = F.interpolate(x_ls, lbr.size()[2:], mode='bilinear', align_corners=True)
            #lbr = torch.cat([lbr, x_ls], dim=1)

            _, _, _, lbr_c4 = self.base_forward(lbr) #(2048,84,84)
            #lbr_c4 = self.cbam1(lbr_c4)
            lbr_c4 = self.conv1(lbr_c4)
             #第一次使用注意力
            #lbr_c4 = self.conv1(lbr_c4) #相当于降采样
            #lbr_c4 = self.conv2(lbr_c4)
            #x = F.interpolate(x, lbr_c4.size()[2:], mode='bilinear', align_corners=True)
            #x = torch.cat([lbr_c4,c4], dim=1) #拼接
            x = c4
            #x = self.cbam2(x) #第二次使用注意力
            #c4 = self.conv2(c4) #降低通道数

        x = self.head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        outputs.append(x)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, size, mode='bilinear', align_corners=True)
            outputs.append(auxout)
        return tuple(outputs)

Out_Channel = 256

class _DeepLabHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DeepLabHead, self).__init__()
        self.aspp = _ASPP(2048, [12, 24, 36], norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        self.block = nn.Sequential(
            nn.Conv2d(Out_Channel, Out_Channel, 3, padding=1, bias=False),
            norm_layer(Out_Channel, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Conv2d(Out_Channel, nclass, 1)
        )

    def forward(self, x):
        x = self.aspp(x)
        return self.block(x)


class _ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rate, norm_layer, norm_kwargs):
        super(_ASPPConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate, dilation=atrous_rate, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class _AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, norm_kwargs, **kwargs):
        super(_AsppPooling, self).__init__()
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

    def forward(self, x):
        size = x.size()[2:]
        pool = self.gap(x)
        out = F.interpolate(pool, size, mode='bilinear', align_corners=True)
        return out


class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, norm_kwargs, **kwargs):
        super(_ASPP, self).__init__()
        out_channels = Out_Channel
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b1 = _ASPPConv(in_channels, out_channels, rate1, norm_layer, norm_kwargs)
        self.b2 = _ASPPConv(in_channels, out_channels, rate2, norm_layer, norm_kwargs)
        self.b3 = _ASPPConv(in_channels, out_channels, rate3, norm_layer, norm_kwargs)
        self.b4 = _AsppPooling(in_channels, out_channels, norm_layer=norm_layer, norm_kwargs=norm_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        feat1 = self.b0(x)
        feat2 = self.b1(x)
        feat3 = self.b2(x)
        feat4 = self.b3(x)
        feat5 = self.b4(x)
        x = torch.cat((feat1, feat2, feat3, feat4, feat5), dim=1)
        x = self.project(x)
        return x

class deeplabv3(nn.Module):
    def __init__(self, n_class=21):
        super(deeplabv3, self).__init__()
        self.n_class = n_class
        self.fcn = deeplabv3_resnet50(pretrained=True, num_classes=self.n_class)

    def forward(self, x):
        return self.fcn(x)['out']

class Deeplab_Torch_Multisource(nn.Module):
    def __init__(self,n_class=13):
        super(Deeplab_Torch_Multisource, self).__init__()
        self.n_class = n_class
        self.img_HR = models.segmentation.deeplabv3_resnet50(pretrained=True,num_classes=self.n_class)
        self.img_LS = models.segmentation.deeplabv3_resnet50(pretrained=True,num_classes=self.n_class)
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=3)
        self.conv_block_fc = nn.Sequential(
            nn.Conv2d(42, self.n_class, kernel_size=(1, 1), stride=(1, 1)),
        )
    def forward(self, x_ls, lbr=None):
        x = self.img_LS(x_ls)
        if lbr is not None:
            lbr = self.conv1(lbr)
            x_GE = self.img_HR(lbr)
            x = torch.cat([x, x_GE],1)
        out = self.conv_block_fc(x)
        return out







def get_deeplabv3_geofeat(dataset='pascal_voc', backbone='resnet50', pretrained=False, root='~/.torch/models',
                  pretrained_base=True, **kwargs):
    acronyms = {
        'pascal_voc': 'pascal_voc',
        'pascal_aug': 'pascal_aug',
        'ade20k': 'ade',
        'coco': 'coco',
        'pascal_voc_jibuti': 'pascal_voc_jibuti',
        'citys': 'citys',
    }
    from ..data.dataloader import datasets
    model = DeepLabV3_Geofeat(datasets[dataset].NUM_CLASS, backbone=backbone, pretrained_base=pretrained_base, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        device = torch.device(kwargs['local_rank'])
        model.load_state_dict(torch.load(get_model_file('deeplabv3_%s_%s' % (backbone, acronyms[dataset]), root=root),
                              map_location=device))
    return model




def get_deeplabv3_geofeat_resnet50_voc(**kwargs):
    return get_deeplabv3_geofeat('pascal_voc', 'resnet50', **kwargs)



if __name__ == '__main__':
    model = get_deeplabv3_geofeat_resnet50_voc()
    img = torch.randn(2, 3, 480, 480)
    output = model(img)
