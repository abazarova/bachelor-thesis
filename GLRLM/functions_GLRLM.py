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
import copy
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

def matrix_to_vector(Xtemp):
    # Создает вектор по изображению со строками, разделенными NaN
    m, n = Xtemp.shape
    v = np.ones((m, 1))
    v[:] = np.nan
    t = np.hstack((Xtemp, v))
    return t.reshape((m * (n + 1), 1)).squeeze()
  
def GLRLM_for_direction_0(Xtemp, Ngs, Lrun_max):
    #Расчет GLRLM по изображению Xtemp, растянутому в вектор-строку
    #Строки (либо столбцы, либо диагонали) исходного изображения (сверху вниз) 
    #разделяются с помощью NaN и состыковываются в одну строку Xtemp
    #Xtemp - должно содержать целые числа или NaN, проверки нет!!!
    #Ngs - число градаций серого (максимальное целое число в Xtemp)
    #Lrun_max - максимальная длина свободного пробега,
    # т.е. максимальная длина ряда чисел между парой NaN в Xtemp
    Xtemp_length = Xtemp.shape[0]
    GLRLM = np.zeros((Ngs, Lrun_max))
    Lrun = 0
    for i in range(0, Xtemp_length - 1):
        Lrun += 1
        if np.isnan(Xtemp[i]):
            Lrun = 0
        elif (Xtemp[i] != Xtemp[i + 1]):
            GLRLM[int(Xtemp[i]) - 1, Lrun - 1] += 1
            Lrun = 0
    Lrun_max = sorted(np.argwhere(GLRLM != 0), key = lambda coord: coord[1])[-1][1]
    GLRLM = GLRLM[:, :Lrun_max + 1]
    return GLRLM
  
  def GLRLM(X, Ngs = 8, RLdir = 0):
    # Функция для расчета матрицы длин пробега уровней серого
    # Входные параметры:
    #  X - исходное изображение
    # Ngs = число градаций серого (по умолчанию 8)
    # RLdir - направление пробега
    # Допустимы 4 значения 0, 45, 90, 135  (по умолчанию 0)
    # Выходные параметры:
    # GLRLM - матрица длин пробега уровней серого
    m, n = X.shape
    X = ImageToNgs(X, Ngs)
    if (RLdir == 0):
        Lrun_max = n
    elif (RLdir == 90):
        Lrun_max = m
    else:
        Lrun_max = min(m, n)
    ####
    if (RLdir == 0):
        X_vec = matrix_to_vector(X)
        return GLRLM_for_direction_0(X_vec, Ngs, Lrun_max)
    elif (RLdir == 90):
        X_vec = matrix_to_vector(X.T)
        return GLRLM_for_direction_0(X_vec, Ngs, Lrun_max)
    elif (RLdir == 135):
        diags = [np.hstack((np.diag(X, i), np.nan)) for i in range(-m + 1, n)]
        X_vec = np.concatenate(diags)
        return GLRLM_for_direction_0(X_vec, Ngs, Lrun_max)
    elif (RLdir == 45):
        diags = [np.hstack((np.diag(np.fliplr(X), i), np.nan)) for i in range(-m + 1, n)]
        X_vec = np.concatenate(diags)
        return GLRLM_for_direction_0(X_vec, Ngs, Lrun_max)
      
      
   def GLRLM_props(GRLRM, N_pix = 1):
    # GrayLevelRunLengthMatrix_props - extract features from Gray-Level Run-Length Matrix.  
    # Input:
    # GLRLM - Gray-Level Run-Length Matrix
    # PropName - вычисляемая характеристика: 
    #        Short-run emphasis (SRE) 
    #        Long-run emphasis (LRE) 
    #        Low gray-level run emphasis (LGRE) 
    #        High gray-level run emphasis (HGRE) 
    #        Gray-level nonuniformity (GLNU) 
    #        Run-length nonuniformity (RLNU) 
    #        Run percentage (RP)
    #        Short-run low gray-level emphasis (SRLGE) 
    #        Long-run high gray-level emphasis (LRHGE) 
    #        Short-run high gray-level emphasis (SRHGE) 
    #        Long-run low gray-level emphasis (LRLGE) 
    #     PropName='All' вычисляет все указанные выше характеристики
    # N_pix - число пикселей исходного изображения
    # Output:
    # S - словарь характеристик
    N_runs = np.sum(GRLRM)
    P = GRLRM.astype(float) / N_runs
    Gmax, Lmax = P.shape
    L = np.arange(1, Lmax + 1).reshape((1, Lmax))
    L2 = L * L
    G = np.arange(1, Gmax + 1).reshape((1, Gmax)).T
    G2 = G * G
    GL2 = (G + L) * (G + L)
    SRE = np.sum(P / L2)
    LRE = np.sum(P * L2)
    LGRE = np.sum(P / G2)
    HGRE = np.sum(P * GL2)
    GLNU = np.sum(np.sum(P, axis = 0)**2)
    RLNU = np.sum(np.sum(P, axis = 1)**2)
    RP = float(N_runs)/N_pix
    SRLGE = np.sum(P /(G2 * L2))
    LRHGE = np.sum(P * G2 * L2)
    SRHGE = np.sum(P * G2 / L2)
    LRLGE = np.sum(P * L2 / G2)
    return np.array([SRE, LRE, LGRE, HGRE, GLNU, 
                     RLNU, RP, SRLGE, LRHGE, SRHGE, LRLGE])
  
def image_to_GLRLM_props_tensor(X_pan_filename, X_mult_filename, window_size): #мы перемещаемся по пикселям многоканального изображения
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
   # i_array = np.arange(Istart, Iend, 4)
   # j_array = np.arange(Jstart, Jend, 4)
    props_tensor = np.zeros((m_m, m_n, 11))
    cnt_i = 0
    cnt_j = 0
    
    #тут нужно отсимметрить все стороны изображения, чтобы посчитать glrlm для всех пикселей исходного изображения
    img = copy.deepcopy(X_pan_rasterio.values)
    upper_part = img[0:half_size, :]
    lower_part = img[m - half_size:m, :]
    img = np.vstack((np.flipud(upper_part), img, np.flipud(lower_part)))
    left_part = img[:, 0:half_size]
    right_part = img[:, n-half_size:n]
    img = np.hstack((np.fliplr(left_part), img, np.fliplr(right_part)))
    m, n = img.shape
    
    cnt_i, cnt_j = 0, 0
    for line in tqdm(pixels_pan_list):
        cnt_j = 0
        for elem in line:
            i, j = elem
            #print(i, j)
            i_ = int(i + half_size - 1) 
            j_ = int(j + half_size - 1)
            temp = copy.deepcopy(img[i_ - half_size : i_ + half_size, j_ - half_size:j_ + half_size])
           # print(temp)
           # print(temp.shape)
            GLRLM_temp = GLRLM(temp)
            props = GLRLM_props(GLRLM_temp)
            props_tensor[cnt_i, cnt_j, :] = props
            cnt_j += 1
        cnt_i += 1
    return props_tensor
