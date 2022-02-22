import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import xarray as xr
import rioxarray as rxr
import earthpy as et
import earthpy.plot as ep

from tqdm.notebook import tqdm, trange
import time

import skimage
from skimage.feature import greycomatrix, greycoprops

from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import copy

import torch

def SRDM(X, Ngs = 8, Q = 0, isBorder = True):
# Функция для расчета тензора, в координате [i][j][k] которого лежит координата матрицы SRDM к которой прибавляется единица
# Входные параметры:
#    X - исходное изображение (one channel)
#    Ngs = число градаций серого (по умолчанию 8)
#    Q - пороговое значение (по умолчанию 0)
#    isBorder - флаг включения граничных пикселей в расчет
#       true - изображение зеркально расширяется на 3 пикселя по каждой границе
#       false - используются только внутренние пиксели
#       по умолчанию - true
# Выходные параметры:
#    тензор размера ... в котором лежат координаты
    X = ImageToNgs(X, Ngs)
    m, n = X.shape
    # расстояние от края окружения до центрального пикселя (border-center) 
    Dbc = 3
    # Размеры матрицы SRDM
    M = 16 + 1
    N = 24 + 1
    if isBorder:
        upper_part = X[0:3, :]
        lower_part = X[m - 3:m, :]
        X = np.vstack((np.flipud(upper_part), X, np.flipud(lower_part)))
        left_part = X[:, 0:3]
        right_part = X[:, n-3:n]
        X = np.hstack((np.fliplr(left_part), X, np.fliplr(right_part)))
        m, n = X.shape
        Istart = Dbc
        Iend = m - Dbc
        Jstart = Dbc
        Jend = n - Dbc
        SRDM_tensor = np.zeros((m - 2 * Dbc, n - 2 * Dbc, 1, 2))
        for i in tqdm(range(Istart, Iend)):
            for j in range(Jstart, Jend):
                Xtemp = X[i - Dbc : i + Dbc + 1, j - Dbc:j + Dbc + 1]
                alist=[Xtemp[0,:-1], Xtemp[:-1,-1], Xtemp[-1,::-1], Xtemp[-2:0:-1,0]]
                R2 = np.concatenate(alist)
                Alpga_j = np.sum((X[i, j] - R2) > Q) 
                X_temp_1 = Xtemp[1:6, 1:6]
                R1 = [X_temp_1[0,:-1], X_temp_1[:-1,-1], X_temp_1[-1,::-1], X_temp_1[-2:0:-1,0]]
                R1 = np.concatenate(R1)
                Alpga_i = np.sum((X[i, j] - R1) > Q) 
                SRDM_tensor[i - Dbc][j - Dbc][0] = (Alpga_i, Alpga_j)
    else:
        m, n = X.shape
        Istart = Dbc
        Iend = m - Dbc
        Jstart = Dbc
        Jend = n - Dbc
        SRDM = np.zeros((M, N))
        SRDM_tensor = np.zeros((m - 2 * Dbc, n - 2 * Dbc, 1, 2))
        for i in tqdm(range(Istart, Iend)):
            for j in range(Jstart, Jend):
                Xtemp = X[i - Dbc : i + Dbc + 1, j - Dbc:j + Dbc + 1]
                alist=[Xtemp[0,:-1], Xtemp[:-1,-1], Xtemp[-1,::-1], Xtemp[-2:0:-1,0]]
                R2 = np.concatenate(alist)
                Alpga_j = np.sum((X[i, j] - R2) > Q) 
                X_temp_1 = Xtemp[1:6, 1:6]
                R1 = [X_temp_1[0,:-1], X_temp_1[:-1,-1], X_temp_1[-1,::-1], X_temp_1[-2:0:-1,0]]
                R1 = np.concatenate(R1)
                Alpga_i = np.sum((X[i, j] - R1) > Q) 
                SRDM_tensor[i - Dbc][j - Dbc][0] = (Alpga_i, Alpga_j)
    return SRDM_tensor 
  
  def SRDM_props(SRDM):
    # SurroundingRegionDependencyMatrix_props - extract features from Surrounding Region Dependency Matrix.  
    # Input:
    # SRDM - Surrounding Region Dependency Matrix
    # Properties: 
    #        Horizontal Weighted Sum (HWS)
    # Vertical Weighted Sum (VWS)
    # Diagonal Weighted Sum (DWS)
    # Grid Weighted Sum (GWS)
    # Output:
    # dict of features
    N_SRDM = np.sum(SRDM)
    if N_SRDM == 0:
        N_SRDM = 1
    M, N = SRDM.shape
    I = np.ones((M, N))
    SRDM = np.where(SRDM <= 0, -1., SRDM)
    R = I.astype(float) / SRDM
    R = np.where(R < 0, 0, R)
    m = M - 1
    n = N - 1
    I = np.arange(0, M).reshape((1, M))
    J = np.arange(0, N).reshape((1, N)).T
    HWS = np.sum(R @ (J * J)) / N_SRDM
    VWS = np.sum((I * I) @ R) / N_SRDM
    diags_sum = np.array([np.sum(np.diag(np.fliplr(R), i)) for i in range(-M + 1, N)])
    diags_sum = np.fliplr(diags_sum.reshape((1, len(diags_sum))))
    K = np.arange(0, diags_sum.shape[1]).reshape((1, diags_sum.shape[1]))
    DWS = np.sum(K * K @ diags_sum.T.astype(float)) / N_SRDM
    GWS = np.sum(I.T * J.T * R) / N_SRDM
    return np.array([HWS, VWS, DWS, GWS]) 
  
  def build_SRDM(small_tensor):
    M = 16 + 1
    N = 24 + 1
    m, n = small_tensor.shape[0], small_tensor.shape[1]
    SRDM = np.zeros((M, N))
    for i in range(m):
        for j in range(n):
            x = small_tensor[i][j][0][0]
            y = small_tensor[i][j][0][1]
            SRDM[int(x)][int(y)] += 1
    return SRDM
  
  
  def image_to_SRDM_props_tensor(T, X_pan_filename, X_mult_filename, window_size,  Ngs = 8, isSymmetric = True, isBorder = True):
    #на вход подается тензор координат в матрице CDTM
    half_size = int(window_size / 2) # вообще предполагаю, что window_size кратно 2
    
    X_pan_gdal = gdal.Open(X_pan_filename)
    X_mult_gdal = gdal.Open(X_mult_filename)
    X_pan_rasterio = rxr.open_rasterio(X_pan_filename).squeeze()
    X_mult_rasterio = rxr.open_rasterio(X_mult_filename)
    m = X_pan_rasterio.values.shape[0]
    n = X_pan_rasterio.values.shape[1]
    m_m = X_mult_rasterio.values.shape[1]
    m_n = X_mult_rasterio.values.shape[2]
    
    geo_trans_pan = X_pan_gdal.GetGeoTransform()
    geo_trans_mult = X_mult_gdal.GetGeoTransform()
    proj_pan = X_pan_gdal.GetProjection()
    proj_mult = X_mult_gdal.GetProjection()
    
    #тут нужно сделать перевод пикселей мультиканалного изображения в пиксели одноканального изображения
    pixels_pan_list = []
    for i in range(1, m_m + 1):
        a = []
        for j in range(1, m_n + 1):
            a.append((i, j))
        a_coords = pix2coord(a, geo_trans_mult, proj_mult) #были пиксели в мультиканальном, стали координаты
        a_pixels_in_pan = coord2pix(a_coords, geo_trans_pan, proj_pan)
        pixels_pan_list.append(a_pixels_in_pan)
    #теперь в pixels_list лежат листы с пикселями каждой строки
    
    props_tensor = np.zeros((m_m, m_n, 4))
    cnt_i = 0
    cnt_j = 0
    
     #тут нужно отсимметрить все стороны тензора, чтобы посчитать srdm для всех пикселей исходного изображения
    T_ = torch.tensor(copy.deepcopy(T))
    upper_part = torch.flip(T_[0:half_size, :, :, :], [0])
    lower_part = torch.flip(T_[m - half_size:m, :, :, :], [0])
    T_ = torch.cat((upper_part, T_, lower_part), dim = 0)
    left_part = torch.flip(T_[:, 0:half_size, :, :], [1])
    right_part = torch.flip(T_[:, n-half_size:n, :, :], [1])
    T_ = torch.cat((left_part, T_, right_part), dim = 1)
    T_ = np.array(T_)
    
    for line in tqdm(pixels_pan_list):
        cnt_j = 0
        for elem in line:
            i, j = elem
            i = int(i + half_size - 1)
            j = int(j + half_size - 1)
            temp = T_[i - half_size : i + half_size + 1, j - half_size:j + half_size + 1, :, :]
            SRDM_temp = build_SRDM(temp)
            props = SRDM_props(SRDM_temp)
            props_tensor[cnt_i, cnt_j, :] = props
            cnt_j += 1
        cnt_i += 1
    return props_tensor
