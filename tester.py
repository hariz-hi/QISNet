from collections import OrderedDict

import numpy as np
import torch
import cv2
from tqdm import tqdm
import utils.utils as utils
from abc import ABCMeta, abstractmethod

from utils.utils import timer


class Tester(metaclass=ABCMeta):
    def __init__(self, model, test_loader, save_path, device):
        super(Tester, self).__init__()
        self.model = model
        self.test_loader = test_loader
        self.save_path = save_path
        self.device = device

    @abstractmethod
    def preprocess(self, **kwargs):
        pass

    @abstractmethod
    def body(self, **kwargs):
        pass

    @abstractmethod
    def postprocess(self, **kwargs):
        pass

    @timer('test')
    def test(self):
        model = self.model
        device = self.device
        test_loader = self.test_loader

        model = model.to(device)
        model.eval()

        with open(f"{self.save_path}/hyper_parameters.txt", "w", encoding="utf-8") as file:
            file.write(str(model))

        psnr_all_imgs = []
        ssim_all_imgs = []
        with (torch.inference_mode()):
            with tqdm(enumerate(test_loader), total=len(test_loader)) as pbar:
                pbar.set_description(f"[test]")
                for idx, (input, target) in pbar:
                    input = input.to(device)
                    target = target.to(device)
                    height, width = input.shape[2], input.shape[3]

                    input, target = self.preprocess(input=input, target=target)
                    rets = self.body(input=input)
                    ret_imgs = self.postprocess(rets=rets, img_size=(height, width))

                    for i in range(len(ret_imgs)):
                        ret_imgs[i] = torch.clip(ret_imgs[i] * 255, 0, 255).detach().cpu().numpy()
                    target = torch.clip(target * 255, 0, 255).detach().cpu().numpy()

                    psnrs = []
                    ssims = []
                    for i in range(len(ret_imgs)):
                        psnr = cv2.PSNR(ret_imgs[i], target, 255)
                        ssim, _ = cv2.quality.QualitySSIM_compute(ret_imgs[i], target)
                        psnrs.append(psnr)
                        ssims.append(ssim[0])

                    cv2.imwrite(f"{self.save_path}/reconstructed_data{idx + 1}.png", ret_imgs[-1].astype(np.uint8))

                    psnr_all_imgs.append(psnrs)
                    ssim_all_imgs.append(ssims)
                    pbar.set_postfix(OrderedDict(PSNR=psnrs[-1], SSIM=ssims[-1]))

        utils.metrics_csv(psnr_all_imgs, self.save_path, "psnr")
        utils.metrics_csv(ssim_all_imgs, self.save_path, "ssim")
