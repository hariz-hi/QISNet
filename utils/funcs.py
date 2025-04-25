import torch
import torch.nn.functional as F


def extract_patches_from_image(images, C, patch_size, stride):
    # (B, C, H, W) → (B, C, patch_H, num_patch_W, patch_size * patch_size)
    patches = images.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
    num_patch_H, num_patch_W = patches.shape[2], patches.shape[3]

    # (B, C, patch_H, num_patch_W, patch_size * patch_size) → (B * patch_H * num_patch_W, C, patch_size * patch_size)
    patches = patches.contiguous().view(-1, C, patch_size, patch_size)

    return patches, num_patch_H, num_patch_W


def reconstruct_image_from_patches(patches, B, C, patch_size, stride, num_patch_H, num_patch_W, img_H, img_W):
    patches = patches.view(B, C, num_patch_H, num_patch_W, patch_size, patch_size)

    reconstructed = F.fold(
        patches.permute(0, 1, 4, 5, 2, 3).contiguous().view(
            B, C * patch_size * patch_size, num_patch_H * num_patch_W
        ),
        output_size=(img_H, img_W),
        kernel_size=patch_size,
        stride=stride
    )

    ones_ = torch.ones((B, C, img_H, img_W)).to(patches.device)
    divisor = F.fold(
        ones_.unfold(2, patch_size, stride).unfold(3, patch_size, stride)
        .contiguous().view(B, C, num_patch_H * num_patch_W, patch_size, patch_size)
        .permute(0, 1, 3, 4, 2).contiguous().view(B, C * patch_size * patch_size, num_patch_H * num_patch_W),
        output_size=(img_H, img_W), kernel_size=patch_size, stride=stride
    )
    reconstructed /= divisor

    return reconstructed
