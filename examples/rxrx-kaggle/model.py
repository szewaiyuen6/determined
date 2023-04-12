import math

import torch
import torchvision
from torch import nn
from torch.nn import functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        kwargs = {}
        backbone = "resnet18"

        first_conv = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        pretrained_backbone = getattr(torchvision.models, backbone)(pretrained=True, **kwargs)
        self.features = nn.Sequential(
            first_conv,
            pretrained_backbone.bn1,
            pretrained_backbone.relu,
            pretrained_backbone.maxpool,
            pretrained_backbone.layer1,
            pretrained_backbone.layer2,
            pretrained_backbone.layer3,
            pretrained_backbone.layer4,
        )
        features_num = pretrained_backbone.fc.in_features

        self.concat_cell_type = True
        self.classes = 1139

        features_num = features_num + (4 if self.concat_cell_type else 0)
        embedding_size = 1024
        self.neck = nn.Sequential(
            nn.BatchNorm1d(features_num),
            nn.Linear(features_num, embedding_size, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embedding_size),
            nn.Linear(embedding_size, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        self.arc_margin_product = ArcMarginProduct(embedding_size, self.classes)

        self.head = nn.Linear(embedding_size, self.classes)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.05

    def embed(self, x, s):
        print("inside embed")
        x = self.features(x)
        print("got features")
        x = F.adaptive_avg_pool2d(x, (1, 1))
        print("got adaptive_avg_pool2d")
        x = x.view(x.size(0), -1)
        print("got view")
        if self.concat_cell_type:
            print("in concat_cell_type")
            x = torch.cat([x, s], dim=1)
            print("got torch.cat([x, s], dim=1)")
        embedding = self.neck(x)
        print("got self.neck")
        return embedding

    def metric_classify(self, embedding):
        return self.arc_margin_product(embedding)

    def classify(self, embedding):
        return self.head(embedding)


class ModelAndLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = Model()
        self.metric_crit = ArcFaceLoss()
        self.crit = DenseCrossEntropy()

    def train_forward(self, x, s, y):
        embedding = self.model.embed(x, s)

        metric_output = self.model.metric_classify(embedding)
        metric_loss = self.metric_crit(metric_output, y)

        output = self.model.classify(embedding)
        loss = self.crit(output, y)

        acc = (output.max(1)[1] == y.max(1)[1]).float().mean().item()

        coeff = 0.2
        return loss * (1 - coeff) + metric_loss * coeff, acc

    def eval_forward(self, x, s):
        print("inside eval_forward")
        embedding = self.model.embed(x, s)
        print("got embedding")
        output = self.model.classify(embedding)
        print("got classify")
        return output

    def embed(self, x, s):
        return self.model.embed(x, s)


class DenseCrossEntropy(nn.Module):
    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.modules.Module):
    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss / 2


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, features):
        cosine = F.linear(F.normalize(features), F.normalize(self.weight))
        return cosine
