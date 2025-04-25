import torch
import os
from PIL import Image
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset


class BitplaneImageDataset(Dataset):
    def __init__(self, root_dir, num_bitplane):
        self.root_dir = root_dir
        self.num_bitplane = num_bitplane
        self.folder_paths = [
            os.path.join(root_dir, folder)
            for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder))
        ]

    def __len__(self):
        return len(self.folder_paths)

    def __getitem__(self, idx):
        folder_path = self.folder_paths[idx]

        image_filenames = sorted([
            f for f in os.listdir(folder_path)  #
            if f.endswith('.png') and 'target' not in f
        ])

        if len(image_filenames) == 0:
            raise ValueError(f"No input images found in the folder: {folder_path}")

        if len(image_filenames) < self.num_bitplane:
            raise ValueError(f"bitplane_num: {len(image_filenames)}, T: {self.num_bitplane}")

        selected_images = []
        for i in range(self.num_bitplane):
            img_path = os.path.join(folder_path, image_filenames[i])
            image = Image.open(img_path).convert("L")

            input_image = ToTensor()(image)
            selected_images.append(input_image)

        input_image = torch.stack(selected_images).squeeze()

        target_image_path = os.path.join(folder_path, 'target.png')

        if not os.path.exists(target_image_path):
            raise ValueError(f"No target image found in the folder: {folder_path}")

        target_image = Image.open(target_image_path).convert("L")

        target_image = ToTensor()(target_image)

        return input_image, target_image



