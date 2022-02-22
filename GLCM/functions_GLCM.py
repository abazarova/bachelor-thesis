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

import copy

def to_calc_textures(window, directions, texture_list, dist=[1]):
    glcm = greycomatrix(np.int32(window), dist, directions, int(np.max(window)) + 1, symmetric=False, normed=False) 
    texture_dict = greycoprops_dir(window, glcm, texture_list)
    return texture_dict


def greycoprops_dir(window, glcm, texture_list):
    # матрица glcm по направлениям сопряженности
    textures = []
    for i in range(glcm.shape[2]):
        textures_directions = []
        for j in range(glcm.shape[3]):
            glcm_direction = glcm[:, :, i, j]
            textures_directions.append(greycoprops_mat(window, glcm_direction, texture_list))
        textures.append(textures_directions)
    texture_dict = {}
    for texture_name in texture_list:
        texture_dict.update({texture_name: np.zeros(np.array(textures).shape)})
    for i, textures_direction_dict in enumerate(textures):
        for j, texture in enumerate(textures_direction_dict):
            for texture_name in texture_list:
                texture_dict[texture_name][i][j] = texture[texture_name]
    return texture_dict


def greycoprops_mat(window, glcm, texture_list):
    glcm_width = len(glcm)
    I = np.repeat(np.swapaxes(np.array([np.arange(1, glcm_width + 1)]), 0, 1), glcm_width, axis=1) #
    J = np.repeat(np.array([np.arange(1, glcm_width + 1)]), glcm_width, axis=0) #
    mui = 0.0
    sigi = 0.0
    muj = 0.0
    sigj = 0.0
    hxy = 0.0
    normed_glcm_x = 0
    normed_glcm_y = 0
    texture_dict = {}
    glcm_sum = np.sum(glcm)
    if glcm_sum != 0:
        normed_glcm = (glcm / glcm_sum)
    else:
        normed_glcm = glcm
    is_sum_average = False
    is_hxy = False
    is_mui = False
    if any(['ClusterProminence' in texture_list, 'Correlation' in texture_list, 'ClusterShade' in texture_list]):
        mui = mean_index(I, normed_glcm)
        is_mui = True
        sigi = std_index(I, normed_glcm, mui)
        muj = mean_index(J, normed_glcm)
        sigj = std_index(J, normed_glcm, muj)
    if any(['InfMeasureCorr1' in texture_list, 'InfMeasureCorr2' in texture_list]):
        normed_glcm_x = np.sum(normed_glcm, axis=1)
        normed_glcm_y = np.sum(normed_glcm, axis=0)

        x_temp = np.log(normed_glcm)
        x_temp[np.isinf(x_temp)] = 0
        hxy = -np.sum(normed_glcm * x_temp)
        is_hxy = True
    # Вычисление Autocorrelation
    if 'Autocorrelation' in texture_list:
        texture_dict.update({'Autocorrelation': np.sum(I * J * normed_glcm)}) #
    # Вычисление ClusterProminence
    if 'ClusterProminence' in texture_list:
        texture_dict.update({'ClusterProminence': np.sum((I + J - mui - muj) ** 4 * normed_glcm)})
    # Вычисление ClusterShade
    if 'ClusterShade' in texture_list:
        texture_dict.update({'ClusterShade': np.sum((I + J - mui - muj) ** 3 * normed_glcm)})
    # Вычисление Contrast
    if 'Contrast' in texture_list:
        texture_dict.update({'Contrast': np.sum((I - J) ** 2 * normed_glcm)})
    # Вычисление Correlation
    if 'Correlation' in texture_list:
        texture_dict.update({'Correlation': np.sum(normed_glcm * (I - mui) * (J - muj) / (sigi* sigj))})
    # Вычисление DiffEntropy
    if 'DiffEntropy' in texture_list:
        x_temp1 = p_X_minus_Y(normed_glcm)
        x_temp2 = np.log(x_temp1)
        x_temp2[np.isinf(x_temp2)] = 0
        texture_dict.update({'DiffEntropy': -np.sum(x_temp1 * x_temp2)})
    # Вычисление DiffVariance
    if 'DiffVariance' in texture_list:
        udiff = np.sum(np.arange(0, glcm_width) * p_X_minus_Y(normed_glcm))
        texture_dict.update(
            {'DiffVariance': np.sum(((np.arange(0, glcm_width) - udiff) ** 2) * p_X_minus_Y(normed_glcm))})
    # Вычисление Dissimilarity
    if 'Dissimilarity' in texture_list:
        texture_dict.update({'Dissimilarity': np.sum(np.abs(I - J) * normed_glcm)})
    # Вычисление Energy
    if 'Energy' in texture_list:
        texture_dict.update({'Energy': np.sum(normed_glcm ** 2)})
    # Вычисление Entropy
    if 'Entropy' in texture_list:
        if is_hxy:
            texture_dict.update({'Entropy': hxy})
        else:
            x_temp = np.log(normed_glcm)
            x_temp[np.isinf(x_temp)] = 0
            texture_dict.update({'Entropy': -np.sum(normed_glcm * x_temp)})
    # Вычисление Homogeneity
    if 'Homogeneity' in texture_list:
        texture_dict.update({'Homogeneity': np.sum(normed_glcm / (abs(I - J) + 1))})
    # Вычисление Homogeneity2
    if 'Homogeneity2' in texture_list:
        texture_dict.update({'Homogeneity2': np.sum(normed_glcm / ((I - J) ** 2 + 1))})
    # Вычисление InfMeasureCorr1
    if 'InfMeasureCorr1' in texture_list:
        x_temp = np.log(np.dot(np.array([normed_glcm_x]).T, np.array([normed_glcm_y])))
        x_temp[np.isinf(x_temp)] = 0
        hxy1 = -np.sum(normed_glcm * x_temp)
        x_temp = np.log(normed_glcm_x)
        x_temp[np.isinf(x_temp)] = 0
        hx = -np.sum(normed_glcm_x * x_temp)
        x_temp = np.log(normed_glcm_y)
        x_temp[np.isinf(x_temp)] = 0
        hy = -sum(normed_glcm_y * x_temp)
        texture_dict.update({'InfMeasureCorr1': (hxy - hxy1) / max(hx, hy)})
    # Вычисление InfMeasureCorr2
    if 'InfMeasureCorr2' in texture_list:
        x_temp1 = np.dot(np.array([normed_glcm_x]).T, np.array([normed_glcm_y]))
        x_temp2 = np.log(x_temp1)
        x_temp2[np.isinf(x_temp2)] = 0
        hxy2 = -np.sum(x_temp1 * x_temp2)
        texture_dict.update({'InfMeasureCorr2': np.sqrt(1 - np.exp(-2. * (hxy2 - hxy)))})
    # Вычисление MaxProb
    if 'MaxProb' in texture_list:
        texture_dict.update({'MaxProb': np.max(normed_glcm)})
    # Вычисление SumAverage
    if 'SumAverage' in texture_list:
        texture_dict.update(
            {'SumAverage': np.sum(np.dot(np.arange(2, 2 * glcm_width + 1), p_X_plus_Y(normed_glcm)))})
        is_sum_average = True
    # Вычисление SumEntropy
    if 'SumEntropy' in texture_list:
        x_temp1 = p_X_plus_Y(normed_glcm)
        x_temp2 = np.log(x_temp1)
        x_temp2[np.isinf(x_temp2)] = 0
        texture_dict.update({'SumEntropy': -np.sum(x_temp1 * x_temp2)})
    # Вычисление SumSquares
    if 'SumSquares' in texture_list:
        if ~is_mui:
            mui = mean_index(I, normed_glcm)
        texture_dict.update({'SumSquares': np.sum(((I - mui) ** 2 * normed_glcm))})
    # Вычисление SumVariance
    if 'SumVariance' in texture_list:
        if is_sum_average:
            f_sum_average = texture_dict['SumAverage']
        else:
            f_sum_average = np.sum(np.dot(np.arange(2, 2 * glcm_width + 1), p_X_plus_Y(normed_glcm)))
        texture_dict.update({'SumVariance': np.sum(
            ((np.array([np.arange(2, 2 * glcm_width + 1)]).T - f_sum_average) ** 2) * p_X_plus_Y(normed_glcm))})
    if 'SDGL' in texture_list:
        texture_dict.update({'SDGL': window.std()})
    return texture_dict

def std_index(IDX, GLCMnorm, IDXmean):
    return np.sqrt(np.sum((IDX - IDXmean) ** 2 * GLCMnorm))


def mean_index(IDX, GLCMnorm):
    return np.sum(IDX * GLCMnorm)


def p_X_minus_Y(P):
    pxmy = np.zeros((1, len(P)))[0]
    for i in range(0, len(P)):
        for j in range(0, len(P)):
            k = abs(i - j)
            pxmy[k] += P[i, j]
    return pxmy


def p_X_plus_Y(P):
    pxpy = np.zeros((2 * len(P) - 1, 1))
    for i in range(0, len(P)):
        for j in range(0, len(P)):
            k = i + j
            pxpy[k] += P[i, j]
    return pxpy
  
  def image_to_greycomatrix_props_tensor(image, X_pan_filename, X_mult_filename, window_size, directions = [0],
                                       texture_list = ['Autocorrelation', 'ClusterProminence', 'ClusterShade', 'Contrast',
                                                       'Correlation', 'DiffEntropy', 'DiffVariance', 'Dissimilarity', 'Energy',
                                                       'Entropy', 'Homogeneity', 'Homogeneity2', 'InfMeasureCorr1',
                                                      'InfMeasureCorr2', 'MaxProb', 'SumAverage', 'SumEntropy', 
                                                      'SumSquares', 'SumVariance'],  distances = [1]):
    #на вход подается изображение
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
    
    props_tensor = np.zeros((m_m, m_n, 19))
    cnt_i = 0
    cnt_j = 0
    
     #тут нужно отсимметрить все стороны изображения, чтобы посчитать srdm для всех пикселей исходного изображения
    img = copy.deepcopy(X_pan_rasterio.values)
    upper_part = img[0:half_size, :]
    lower_part = img[m - half_size:m, :]
    img = np.vstack((np.flipud(upper_part), img, np.flipud(lower_part)))
    left_part = img[:, 0:half_size]
    right_part = img[:, n-half_size:n]
    img = np.hstack((np.fliplr(left_part), img, np.fliplr(right_part)))
    
    img = ImageToNgs(img, 64) - 1.

    for line in tqdm(pixels_pan_list):
        cnt_j = 0
        for elem in line:
            i, j = elem
            i = int(i + half_size - 1)
            j = int(j + half_size - 1)
            window = copy.deepcopy(img[i - half_size : i + half_size + 1, j - half_size:j + half_size + 1]).astype(int)
            d = to_calc_textures(window, directions, texture_list, distances)
            props = np.array([d[elem] for elem in texture_list]).squeeze()
            props_tensor[cnt_i, cnt_j, :] = props
            cnt_j += 1
        cnt_i += 1
    return props_tensor
