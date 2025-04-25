from torch.utils.data import Dataset


class PatchBitplaneDatasetWrapper(Dataset):
    def __init__(self, base_dataset, patch_size=8, stride=4):
        self.base_dataset = base_dataset
        self.patch_size = patch_size
        self.stride = stride
        self.image_patches = []

        # assuming that the same size for all images
        input_image, target_image = self.get_image_size(0)
        _, H, W = input_image.shape
        num_patches_h = (H - patch_size) // stride + 1
        num_patches_w = (W - patch_size) // stride + 1

        for i in range(len(base_dataset)):
            self.image_patches.append(num_patches_h * num_patches_w)

    def get_image_size(self, index):
        input_image, target_image = self.base_dataset[index]
        return input_image, target_image

    def __len__(self):
        return sum(self.image_patches)

    def __getitem__(self, idx):
        patch_count = 0
        for img_idx, num_patches in enumerate(self.image_patches):
            if idx < patch_count + num_patches:
                input_image, target_image = self.base_dataset[img_idx]

                local_idx = idx - patch_count
                _, H, W = input_image.shape
                num_patches_w = (W - self.patch_size) // self.stride + 1
                row = local_idx // num_patches_w
                col = local_idx % num_patches_w

                x_start, y_start = col * self.stride, row * self.stride
                input_patch = input_image[:, y_start:y_start + self.patch_size, x_start:x_start + self.patch_size]
                target_patch = target_image[:, y_start:y_start + self.patch_size, x_start:x_start + self.patch_size]

                return input_patch, target_patch

            patch_count += num_patches
