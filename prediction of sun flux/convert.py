import astropy.io.fits as fits
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('/work/ymatumot/solar_image/learning_data')
import os
import cv2

count = 1

# 解像度の選択
shape = 384
SHAPE  = str(shape)

# 画像、真値の配列データを格納する配列の作成
x_data = np.empty((0, 1, shape, shape), dtype=np.float32)
t_data = np.empty(0, dtype=np.float32)

#時間の測定
import time
start = time.time()

# csv ファイルを開く
with open('ar3.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for key in reader:

        # key より csv ファイルを読み込む
        year = str(key[0])
        month = str(key[1])
        day = str(key[2])
        hour = str(key[3])
        h = "/work/ysawamura/solar_image/learning_data/hmi.M_45s."+year+month+day+"_"+hour+"0045_TAI.2.magnetogram"+".fits"
        files = os.listdir('/work/ysawamura/solar_image/learning_data')
        if os.path.exists(h):

            # fits データを開く
            hublist = fits.open(h)
            hublist.verify('fix')
            s_img = hublist[1].data

            # 画像の解像度の変換
            s_img = cv2.resize(s_img, (shape,shape), cv2.INTER_AREA)
            s_img[np.isnan(s_img)] = 0

            # CNN に合わせてリシェイプ
            s_img = s_img.reshape(1, 1, shape, shape)
            x_data = np.append(x_data, s_img, axis=0)
            print(x_data.shape)

            # 真値データの作成
            flux = float(key[6]) + abs(float(key[8]))
            t_data = np.append(t_data, flux)

            # 計算できているか確認
            print(count)
            count += 1
        else:
            next(reader)
# 計算時間の表示
elapsed_time = time.time() - start
print(f"time:{elapsed_time}")

# 訓練データを配列として保存
np.save('/work/ysawamura/solar_image/x_data_r'+ SHAPE +'.npy', x_data)
np.save('/work/ysawamura/solar_image/t_data_r.npy', t_data)
