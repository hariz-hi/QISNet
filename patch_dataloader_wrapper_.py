from math import floor, ceil
import torch

from utils.funcs import extract_patches_from_image


class PatchBitplaneDataloaderWrapper:
    def __init__(self, base_dataloader, batch_size, patch_size=8, stride=4, shuffle=True, drop_last=True):
        self.base_dataloader = base_dataloader
        self.patch_size = patch_size
        self.stride = stride
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.batch_size = batch_size

        # assuming that the same size for all images
        H, W = self.get_image_size()
        num_patches_h = (H - patch_size) // stride + 1
        num_patches_w = (W - patch_size) // stride + 1

        self.num_patches_per_img = num_patches_h * num_patches_w

    def get_image_size(self):
        input_image, _ = next(iter(self.base_dataloader))
        _, _, H, W = input_image.shape

        return H, W

    def __len__(self):
        temp = (self.num_patches_per_img * self.base_dataloader.batch_size) / self.batch_size
        temp = floor(temp) if self.drop_last else ceil(temp)
        return len(self.base_dataloader) * temp

    def __iter__(self):
        for input_, target_ in self.base_dataloader:
            input_patches, _, _ = extract_patches_from_image(
                input_, input_.shape[1], self.patch_size, self.stride
            )
            target_patches, _, _ = extract_patches_from_image(
                target_, target_.shape[1], self.patch_size, self.stride
            )
            if self.shuffle:
                shuffle_indices = torch.randperm(input_patches.shape[0])
                input_patches = input_patches[shuffle_indices]
                target_patches = target_patches[shuffle_indices]

            input_list = list(torch.split(input_patches, self.batch_size))
            target_list = list(torch.split(target_patches, self.batch_size))

            if self.drop_last:
                if input_list[0].shape != input_list[-1].shape:
                    del input_list[-1]
                if target_list[0].shape != target_list[-1].shape:
                    del target_list[-1]

            for input__, target__ in zip(input_list, target_list):
                yield input__, target__
