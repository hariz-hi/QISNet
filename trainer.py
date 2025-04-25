import torch
import os
import gc
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from tqdm import tqdm
from dataloaders.get_dataloaders import get_train_val_dataloaders
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from dataloaders.patch_dataloader_wrapper_ import PatchBitplaneDataloaderWrapper


class Trainer(metaclass=ABCMeta):
    def __init__(
            self,
            model,
            num_epochs,
            lr,
            criterion,
            T,
            alpha,
            batch_size,
            train_rate,
            shuffle,
            drop_last,
            save_path,
            device,
    ):
        super(Trainer, self).__init__()
        self.model = model
        self.optimizer = None
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.T = T
        self.alpha = alpha
        self.lr = lr
        self.criterion = criterion
        self.train_loader = None
        self.val_loader = None
        self.train_rate = train_rate
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.save_path = save_path
        self.device = device

        self.is_debug = False

        self.preparation()

        self.calcPSNR = PeakSignalNoiseRatio(data_range=1.0).to(self.device)
        self.calcSSIM = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def __str__(self):
        return \
                f'model: {self.model.__class__.__name__}\n' + \
                f"num_epochs: {self.num_epochs}\n" + \
                f"batch_size: {self.batch_size}\n" + \
                f"T: {self.T}\n" + \
                f"lr: {self.lr}\n" + \
                f'criterion: {self.criterion.__class__.__name__}\n' + \
                f"train_rate: {self.train_rate}\n" + \
                f"shuffle: {self.shuffle}\n" + \
                f"drop_last: {self.drop_last}\n"

    def preparation(self):
        is_patch = False
        if self.model.__class__.__name__ == 'DU_ISTA':
            # is_patch = True
            is_patch = False
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.train_loader, self.val_loader = get_train_val_dataloaders(
            self.batch_size, self.T, self.train_rate, os.cpu_count() // 2, self.shuffle, self.drop_last,
            is_patch=is_patch
        )

    @abstractmethod
    def preprocess(self, **kwargs):
        pass

    @abstractmethod
    def body(self, **kwargs):
        pass

    @abstractmethod
    def postprocess(self, **kwargs):
        pass

    def train(self):
        model = self.model
        optimizer = self.optimizer
        train_loader = self.train_loader
        if self.model.__class__.__name__ == 'DU_ISTA':
            train_loader = PatchBitplaneDataloaderWrapper(
                train_loader, self.batch_size * self.batch_size * 4,
                self.model.patch_size, self.model.stride, self.shuffle, self.drop_last
            )

        device = self.device

        model = model.to(device)
        model.train()

        with open(f"{self.save_path}/training_settings.txt", "w", encoding="utf-8") as file:
            file.write(str(self))

        with open(f"{self.save_path}/hyper_parameters.txt", "w", encoding="utf-8") as file:
            file.write(str(model))

        for epoch in range(self.num_epochs):
            print(f'epoch: {epoch + 1}')
            gc.collect()
            torch.cuda.empty_cache()

            total_loss = 0
            with tqdm(enumerate(train_loader), total=len(train_loader)) as pbar:
                pbar.set_description(f"[train] Epoch {epoch + 1}")
                for bidx, (input_, target_) in pbar:
                    input_ = input_.to(device)
                    target_ = target_.to(device)

                    optimizer.zero_grad()

                    input_, target_ = self.preprocess(input=input_, target=target_)
                    rets = self.body(input=input_)
                    rets = self.postprocess(rets=rets)

                    loss = self.criterion(rets, target_)

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                    pbar.set_postfix(OrderedDict(loss=loss.item()))
            print(
                f"[train]epoch: {epoch + 1}/{self.num_epochs}, "
                f"train loss: {total_loss / len(train_loader)}"
            )

            self.validation(epoch)
            torch.save(self.model.state_dict(), f"{self.save_path}/model.pt")

    def validation(self, epoch):
        model = self.model
        val_loader = self.val_loader
        device = self.device

        model = model.to(device)
        model.eval()

        gc.collect()
        torch.cuda.empty_cache()

        with (torch.inference_mode()):
            total_loss = 0
            total_psnr = 0
            total_ssim = 0
            with tqdm(enumerate(val_loader), total=len(val_loader)) as pbar:
                pbar.set_description(f"[val] Epoch {epoch + 1}")
                for bidx, (input_, target_) in pbar:
                    input_ = input_.to(device)
                    target_ = target_.to(device)

                    input_, target_ = self.preprocess(input=input_, target=target_)
                    rets = self.body(input=input_)
                    rets = self.postprocess(rets=rets)

                    loss = self.criterion(rets, target_)
                    psnr = self.calcPSNR(rets, target_)
                    ssim = self.calcSSIM(rets, target_)

                    total_loss += loss.item()
                    total_psnr += psnr.item()
                    total_ssim += ssim.item()
                    pbar.set_postfix(OrderedDict(loss=loss.item(), PSNR=psnr.item(), SSIM=ssim.item()))

            print(
                f"[val]epoch: {epoch + 1}/{self.num_epochs}, "
                f"loss: {total_loss / len(val_loader)}, "
                f"psnr: {total_psnr / len(val_loader)}, "
                f"ssim: {total_ssim / len(val_loader)}, "
            )
