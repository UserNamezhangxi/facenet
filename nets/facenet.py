import torch.nn as nn
from nets.mobilenet import MobileNetV1
from torch.nn import functional as F


class mobilenet(nn.Module):
    def __init__(self):
        super(mobilenet, self).__init__()
        self.model = MobileNetV1()

        del self.model.avg
        del self.model.fc

    def forward(self, x):
        x = self.model.stage1(x)
        x = self.model.stage2(x)
        x = self.model.stage3(x)
        return x


class Facenet(nn.Module):
    def __init__(self, backbone, dropout_prob=0.5, embedding_size=128, num_classes=None, mode="train"):
        super(Facenet, self).__init__()
        if backbone == 'mobilenet':
            self.backbone = mobilenet()
            flat_shape = 1024
        else:
            raise ValueError('Unsupported backbone - `{}`, Use mobilenet, inception_resnetv1.'.format(backbone))
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.Bottleneck = nn.Linear(flat_shape, embedding_size, bias=False)
        self.last_bn = nn.BatchNorm1d(embedding_size, eps=0.001, momentum=0.1, affine=True)
        if mode == "train":
            self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, mode = "predict"):
        if mode == 'predict':
            x = self.backbone(x)  # (2,3,160,160) ->  (2,1024,5, 5)
            x = self.avg(x)       # (2,1024,5, 5) ->  (2,1024,1, 1)
            x = x.view(x.size(0), -1)  # (2,1024,1, 1) -> (2,1024)
            x = self.dropout(x)        # (2,1024)   -> (2,1024)
            x = self.Bottleneck(x)     # (2,1024) -> (2,128)
            x = self.last_bn(x)        # (2,128) -> (2,128)
            x = F.normalize(x, p=2, dim=1)  # (2,128) ->    (2,128)
            return x

        x = self.backbone(x)
        x = self.avg(x)
        x = x.reshape(x.size(0), -1)
        x = self.dropout(x)
        x = self.Bottleneck(x)
        before_normalize = self.last_bn(x)

        x = F.normalize(before_normalize, p=2, dim=1)
        cls = self.classifier(before_normalize)
        return x, cls
