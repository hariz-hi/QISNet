import numpy as np
import cv2
import os
from PIL import Image

# 使用データ番号の指定
num_data = 5

# ファイルパス
path_imagein = f'image/Original/data{num_data}.png'
path_imageout_folder = f'dataset/val/bitplane_data{num_data}/'

# 共通パラメータ設定
num_bitplanes = 1  # 使用bit-plane画像枚数
q = 1  # 入射光子数>=qで,bit-plane画像の画素値1
alpha = 1  # 輝度調整用パラメータ(alpha=1でグレースケール画像の255の画素で入射光子数=1 photon/bit-plane/pix)


# 画像のサイズを取得
def get_image_size(path):
    with Image.open(path) as img:
        width, height = img.size

    return width, height


# 画像の縦横
width, height = get_image_size(path_imagein)

# 入力画像読み込み
imagein = np.zeros((height, width, num_bitplanes), dtype=np.uint8)

tmp = cv2.imread(path_imagein, cv2.IMREAD_GRAYSCALE)
tmp_resized = cv2.resize(tmp, (width, height))  # サイズ変更
for t_tmp in range(num_bitplanes):
    imagein[:, :, t_tmp] = tmp_resized


# ビットプレーン画像生成関数
def Function_BitplaneGen(img_input, output_subframe_number, q, alpha, DC_rate):
    # 入力画像の正規化
    img_DR = img_input.astype(np.float64) / 255.0
    img_normalized = img_DR * output_subframe_number

    # ビットプレーンと入射光子数の初期化
    bitplane = np.zeros((img_input.shape[0], img_input.shape[1], output_subframe_number))
    incident_photons = np.zeros((img_input.shape[0], img_input.shape[1], output_subframe_number))

    # ポアソン分布を使ったフォトンカウントシミュレーションによるビットプレーンの生成
    for t in range(output_subframe_number):
        # 入射光子の平均値
        incident_photon_average = alpha * (img_normalized / output_subframe_number)

        # ポアソン乱数で入射光子数を生成
        incident_photon = np.random.poisson(incident_photon_average[:, :, t])
        incident_photons[:, :, t] = incident_photon

        # ダークカウントノイズの生成
        DC = (np.random.rand(*bitplane[:, :, t].shape) < DC_rate).astype(np.float64)

        # ビットプレーンの生成
        bitplane[:, :, t] = ((incident_photon + DC) >= q).astype(np.float64)

    return bitplane, incident_photons


# ビットプレーン画像生成
DC_rate = 0  # ノイズパラメータ
bitplanes, Incident_photons = Function_BitplaneGen(imagein, num_bitplanes, q, alpha, DC_rate)

# ビットプレーン画像の保存
bitplanes = bitplanes * 255  # 0または1を0または255に変換
os.makedirs(path_imageout_folder, exist_ok=True)
for i in range(num_bitplanes):
    output_path = os.path.join(path_imageout_folder, f'test{i + 1}.png')
    cv2.imwrite(output_path, bitplanes[:, :, i].astype(np.uint8))

print('Completed.')
print('Completed.')
