import numpy as np
from skimage.util import view_as_windows
from skimage.io import imread
from sklearn.decomposition import MiniBatchDictionaryLearning
import torch
import os

#path
image_folder = "dataset/train/"


def extract_patches_from_image(image_path, patch_size=8, stride=4):
    """
    1枚の画像を小さなパッチに分割
    - patch_size: 1つのパッチのサイズ
    - stride: パッチをスライドさせる間隔
    """
    image = imread(image_path, as_gray=True)  # グレースケール画像として読み込む
    image = image / 255.0  # 正規化（0〜1）

    # view_as_windows を用いて画像をパッチに分割
    patches = view_as_windows(image, (patch_size, patch_size), step=stride)
    
    # 2D 配列（ベクトル化されたパッチ）に変換
    patches = patches.reshape(-1, patch_size * patch_size)

    return patches  # (N, patch_size*patch_size) の形



# `target.png` だけを抽出し、パッチを生成
def extract_patches_from_dataset(image_root, patch_size=8, stride=4, max_patches=100000):
    """
    `image_root` 内のすべてのサブフォルダから `target.png` を探し、パッチを抽出
    """
    all_patches = []
    target_files = []

    # すべての `target.png` を取得
    for root, _, files in os.walk(image_root):
        if "target.png" in files:
            target_path = os.path.join(root, "target.png")
            target_files.append(target_path)

    if not target_files:
        raise ValueError("No 'target.png' files found. Check the dataset structure.")

    # `target.png` のみ処理
    for idx, target_path in enumerate(target_files):
        print(f"[{idx+1}/{len(target_files)}] Processing: {target_path}")
        patches = extract_patches_from_image(target_path, patch_size, stride)
        if patches.shape[0] > 0:
            all_patches.append(patches)

    if not all_patches:
        raise ValueError("No valid patches extracted. Check the images.")

    # すべてのパッチを1つの行列に結合
    X = np.vstack(all_patches)

    # 必要ならランダムに max_patches のみ選択
    if X.shape[0] > max_patches:
        indices = np.random.choice(X.shape[0], max_patches, replace=False)
        X = X[indices]

    return X  # (N, patch_size*patch_size) の形

# XをもとにK-SVDによる辞書Dを学習
def learn_dictionary(X, n_components=128, alpha=1.0, max_iter=100):
    """
    K-SVD を用いた辞書学習
    - n_components: 学習する基底の数（辞書の列数）
    - alpha: L1正則化の強さ
    - n_iter: 反復回数
    """
    k_svd = MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha, max_iter=max_iter)
    D = k_svd.fit(X).components_
    
    return D  # 学習された辞書 (n_components, patch_size * patch_size)

# 辞書DをPyTorchで扱える形に変換
def convert_dictionary_to_torch(D):
    """
    学習した辞書 D を PyTorch テンソルに変換
    """
    return torch.tensor(D.T, dtype=torch.float32)  # (patch_size*patch_size, n_components) に転置

### main ###
# 画像パッチを抽出
X = extract_patches_from_dataset(image_folder)

# K-SVDによる辞書学習
D = learn_dictionary(X)

# PyTorch用に変換
D_torch = convert_dictionary_to_torch(D)

# PyTorchの辞書を保存
torch.save(D_torch, "dictionary.pt")

# debug
# print(D)
# print(f"Min of D_torch: {D_torch.min().item()}")
# print(f"Max of D_torch: {D_torch.max().item()}")

print("Compleated.")