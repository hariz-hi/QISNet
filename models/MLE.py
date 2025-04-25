from utils.tester import Tester
import torch
import torch.nn as nn


class MLE(nn.Module):
    def __init__(self, T, alpha, epsilon=1e-6):
        super(MLE, self).__init__()
        self.T = T
        self.alpha = alpha
        self.epsilon = epsilon

    def __str__(self):
        return \
                f"T: {self.T}\n" + \
                f"alpha: {self.alpha}\n"

    def forward(self, bitplane_imgs):
        num_bit1 = torch.sum(bitplane_imgs, dim=1).unsqueeze(1)
        num_bit0 = self.T - num_bit1

        num_bit0[num_bit0 == 0] = self.epsilon

        x = - 1 / self.alpha * torch.log(num_bit0 / self.T)

        return x


class MLE_Tester(Tester):
    def preprocess(self, **kwargs):
        input_ = kwargs['input']

        target = kwargs['target']
        target = target.squeeze()

        return input_, target

    def body(self, **kwargs):
        input_ = kwargs['input']

        ret_imgs = self.model(input_)

        return [ret_imgs]

    def postprocess(self, **kwargs):
        rets = kwargs['rets']

        img_size = kwargs['img_size']

        for i in range(len(rets)):
            rets[i] = torch.reshape(rets[i], img_size)

        return rets
