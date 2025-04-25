import torch.nn as nn
from models.impl.REDNet_impl import REDNet30
import torch

from utils.tester import Tester
from utils.trainer import Trainer


class QISNet(nn.Module):
    def __init__(
            self,
            T,
            alpha,
    ):
        super(QISNet, self).__init__()
        self.T = T
        self.alpha = alpha
        self.model = REDNet30(num_layers=15, num_features=64)

    def __str__(self):
        return \
                f"T: {self.T}\n" + \
                f"alpha: {self.alpha}\n" + \
                f'model: {self.model.__class__.__name__}\n'

    def forward(self, bitplane_imgs):
        prob_num0 = 1 - torch.sum(bitplane_imgs, dim=1).unsqueeze(1) / self.T
        rets = self.model(prob_num0)

        return rets


class QISNet_Trainer(Trainer):

    def preprocess(self, **kwargs):
        input_ = kwargs['input']
        target = kwargs['target']

        return input_, target

    def body(self, **kwargs):
        input_ = kwargs['input']
        rets = self.model(input_)

        return rets

    def postprocess(self, **kwargs):
        rets = kwargs['rets']

        return rets


class QISNet_Tester(Tester):

    def preprocess(self, **kwargs):
        input_ = kwargs['input']

        target = kwargs['target']
        target = target.squeeze()

        return input_, target

    def body(self, **kwargs):
        input_ = kwargs['input']
        rets = self.model(input_)

        return [rets]

    def postprocess(self, **kwargs):
        rets = kwargs['rets']

        img_size = kwargs['img_size']

        for i in range(len(rets)):
            rets[i] = torch.reshape(rets[i], img_size)

        return rets
