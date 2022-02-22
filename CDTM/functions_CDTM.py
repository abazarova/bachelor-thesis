from tqdm.notebook import tqdm, trange
import time
import os

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import xarray as xr
import rioxarray as rxr
import earthpy as et
import earthpy.plot as ep
import rasterio
from rasterio.plot import show

import copy

import torch

def to_decimal(vec): # vec - массив из 4х векторов длины 4 (с элементами 0, 1 или 2); функция возвращает число в десятичном формате
    dcm = np.zeros(4)
    for i in range(4):
        dcm[i] = vec[i][0] + 3. * vec[i][1] + 9. * vec[i][2] + 27. * vec[i][3]
    return dcm
  
def to_decimal_roll(vec):
  """
  На вход: троичный вектор длины 4
  Возвращает: вектор длины 4 из чисел-циклических сдвигов входного вектора
  """
  result = [0] * 4
  for roll in range(4):
      for rank in range(4):
          result[roll] += digit * (3 ** ((rank + roll) % 4))
  return result

def getTextureUnits(E): #на входе подается матрица 3х3
    #Расчет текстурных блоков
    #ED  диагональный, EC - крестовой
    #на вход подается матрица I размера 3х3
    #функция расчета элемента текстурного блока
    V0 = E[1, 1]
    E = np.sign(E - V0) + 1
    E[1, 1] = V0
    ED = np.array([E[0, 0], E[0, 2], E[2, 2], E[2, 0]])
    EC = np.array([E[0, 1], E[1, 2], E[2, 1], E[1, 0]])
    return ED, EC
  
  def CDTM(X, Ngs = 8, isSymmetric = True, isBorder = True): #
    # Функция для расчета кресто-диагональной текстурной матрицы
# Входные параметры:
#    X - исходное изображение
#    Ngs = число градаций серого (по умолчанию 8)
#    isSymmetric - флаг симметричности матрицы
#       true - применяется симметризация
#       false - симметризация не применяется
#       по умолчанию - true
#    isBorder - флаг включения граничных пикселей в расчет
#       true - изображение зеркально расширяется на 1 пиксель по каждой границе
#       false - используются только внутренние пиксели
#       по умолчанию - true
# Выходные параметры:
#    тензор размера m x n x 2 х 4: в каждом пикселе лежат два вектора NDTU и NCTU
    X  = ImageToNgs(X, Ngs)
    m, n = X.shape
    Dbc = 1
   # N_CDTM = 81
    if isBorder:
        upper_part = X[0:1, :]
        lower_part = X[m - 1:m, :]
        X = np.vstack((np.flipud(upper_part), X, np.flipud(lower_part)))
        left_part = X[:, 0:1]
        right_part = X[:, n-1:n]
        X = np.hstack((np.fliplr(left_part), X, np.fliplr(right_part)))
        m, n = X.shape
        Istart = Dbc
        Iend = m - Dbc
        Jstart = Dbc
        Jend = n - Dbc
        pow_3_roll_matrix = np.array([[1, 27, 9, 3], [3, 1, 27, 9], [9, 3, 1, 27], [27, 9, 3, 1]])
        NDTU_NCTU_tensor = np.zeros((m - 2 * Dbc, n - 2 * Dbc, 16, 2))
        for i in tqdm(range(Istart, Iend)):
            for j in range(Jstart, Jend):
                Xtemp = copy.deepcopy(X[i - Dbc : i + Dbc + 1, j - Dbc:j + Dbc + 1])
                ED_0, EC_0 = getTextureUnits(Xtemp)
                NDTU = (ED_0 @ pow_3_roll_matrix).astype(int)
                NCTU = (EC_0 @ pow_3_roll_matrix).astype(int)
                mesh = np.array(np.meshgrid(NDTU, NCTU))
                combinations = mesh.T.reshape(-1, 2)
                NDTU_NCTU_tensor[i - Dbc, j - Dbc, :, :] = combinations
    else:
        m, n = X.shape
        Istart = Dbc
        Iend = m - Dbc
        Jstart = Dbc
        Jend = n - Dbc
        pow_3_roll_matrix = np.array([[1, 27, 9, 3], [3, 1, 27, 9], [9, 3, 1, 27], [27, 9, 3, 1]])
        NDTU_NCTU_tensor = np.zeros((m - 2 * Dbc, n - 2 * Dbc, 16, 2))
        for i in tqdm(range(Istart, Iend)):
            for j in range(Jstart, Jend):
                Xtemp = copy.deepcopy(X[i - Dbc : i + Dbc + 1, j - Dbc:j + Dbc + 1])
                ED_0, EC_0 = getTextureUnits(Xtemp)
                NDTU = (ED_0 @ pow_3_roll_matrix).astype(int)
                NCTU = (EC_0 @ pow_3_roll_matrix).astype(int)
                mesh = np.array(np.meshgrid(NDTU, NCTU))
                combinations = mesh.T.reshape(-1, 2)
                NDTU_NCTU_tensor[i - Dbc, j - Dbc, :, :] = combinations  
    return NDTU_NCTU_tensor
  
  
  def pXminusY(P, N):
    vec = np.zeros((1, N))
    for i in range(N):
        for j in range(N):
            k = int(np.abs(i - j))
            vec[0, k] += P[i, j]
    return vec
  
  
  def pXplusY(P, N):
    vec = np.zeros((1, 2 * N + 1))
    for i in range(N):
        for j in range(N):
            k = int(np.abs(i + j))
            vec[0, k] += P[i, j]
    return vec
  
  
  def CDTM_props(CDTM):
    # CrossDiagonalTextureMatrix_props - extract features from Cross Diagonal Texture Matrix.  
    # Input:
    # CDTM - Cross Diagonal Texture Matrix
    # Output:
    # dict of extracted features
    if np.sum(CDTM) == 0:
        P = CDTM.astype(float)
    else:
        P = CDTM.astype(float) / np.sum(CDTM)
    N_CDTM = 81
    I = np.arange(1, N_CDTM + 1).reshape((1, N_CDTM))
    J = I.T
    mu_i = np.sum(I.T * P)
    mu_j = np.sum(I * P)
    sigma_i = np.sqrt(np.sum((I - mu_i)**2 * P))
    sigma_j = np.sqrt(np.sum((J - mu_j)**2 * P))
    #autocor = np.sum(I * J * P)
    #cluster_prominence = np.sum((I + J - mu_i - mu_j)**4  * P)
    cluster_shade = np.sum((I + J - mu_i - mu_j)**3  * P)
   # contrast = np.sum((I - J)**2 * P)
    if (sigma_i * sigma_j != 0):
        correlation = np.sum((I - mu_i) * (J - mu_j) * P)/(sigma_i * sigma_j)
    else: #it happens when CDTM is a 0 matrix
        correlation = 0
    pxmy = pXminusY(P, N_CDTM)
   # pxmy_new = np.where(pxmy == 0, 1, pxmy)
   # diff_entropy = -np.sum(pxmy_new @ np.log(pxmy_new).T)
    mu_i_minus_j = np.sum(np.arange(0, N_CDTM).reshape((1, N_CDTM)) @ pxmy.T)
    K = np.arange(0, N_CDTM).reshape((1, N_CDTM))
    diff_variance = np.sum((K - mu_i_minus_j)**2 * pxmy)
  #  dissimilarity = np.sum(np.abs(I - J) * P)
   # energy = np.sum(P * P)
    P_without_zeros = np.where(P == 0, 1, P)
    entropy = - np.sum(P * np.log(P_without_zeros))
    homogeneity = np.sum(P / (1 + np.abs(I - J)))
   # homogeneity2 = np.sum(P / (1 + (I - J)**2))
   # py = np.sum(P, axis = 0).reshape((1, N_CDTM))
   # px = np.sum(P, axis = 1).reshape((1, N_CDTM))
   # px_without_zeros = np.where(px == 0, 1, px)
   # py_without_zeros = np.where(py == 0, 1, py)
   # HX = -np.sum(px @ np.log(px_without_zeros).T)
   # HY = -np.sum(py @ np.log(py_without_zeros).T)
   # HXY = entropy
   # px_dot_py = px.T @  py
   # px_dot_py_no_zeros = np.where(px_dot_py == 0, 1, px_dot_py)
   # HXY_1 = -np.sum(P * np.log(px_dot_py_no_zeros))
   # if max(HX, HY) != 0:
   #     inf_measure_corr1 = (HXY - HXY_1)/max(HX, HY)
   # else:
   #     inf_measure_corr1 = 0
   # HXY_2 = -np.sum(px_dot_py * np.log(px_dot_py_no_zeros))
  #  inf_measure_corr2 = np.sqrt(1 - np.exp(-2 * (HXY_2 - HXY)))
   # maxprob = np.max(P)
   # pxpy = pXplusY(P, N_CDTM)
   # k = np.arange(2, 2 * N_CDTM + 3).reshape(1, 2 * N_CDTM + 1)
   # sum_average = np.sum(k @ pxpy.T)
   # pxpy_no_zeros = np.where(pxpy == 0, 1, pxpy)
   # sum_entropy = -np.sum(pxpy @ np.log(pxpy_no_zeros).T)
   # sum_squares = np.sum((I.T - mu_i)**2 * P)
   # mu_i_p_j = sum_average
   # sum_variance = np.sum((k - mu_i_p_j) ** 2 * pxpy)
   # return np.array([autocor, cluster_prominence, cluster_shade, contrast, correlation, diff_entropy, #correlation and homogeneity2 have problems 
    #      diff_variance, dissimilarity, energy, entropy, homogeneity, homogeneity2, 
     #     inf_measure_corr1, inf_measure_corr2, maxprob, sum_average, sum_entropy, sum_squares, sum_variance])
    return np.array([correlation, diff_variance, entropy, homogeneity, cluster_shade])
  
  
  def build_CDTM(small_tensor, isSymmetric = True):    
    N_CDTM = 81
    m, n = small_tensor.shape[0], small_tensor.shape[1]
    CDTM = np.zeros((N_CDTM, N_CDTM))
    for i in range(m):
        for j in range(n):
            for k in range(16): 
                x = small_tensor[i][j][k][0]
                y = small_tensor[i][j][k][1]
                CDTM[int(y)][int(x)] += 1
    if isSymmetric:
        CDTM += CDTM.T
    return CDTM
  
  
  def image_to_CDTM_props_tensor(T, X_pan_filename, X_mult_filename, window_size,  Ngs = 8, isSymmetric = True, isBorder = True):
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
    
    props_tensor = np.zeros((m_m, m_n, 5)) #19
    cnt_i = 0
    cnt_j = 0
    
     #тут нужно отсимметрить все стороны изображения, чтобы посчитать srdm для всех пикселей исходного изображения
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
            temp = T_[i - half_size : i + half_size + 1, j - half_size:j + half_size + 1][:][:]
            CDTM_temp = build_CDTM(temp, isSymmetric)
            props = CDTM_props(CDTM_temp)
            props_tensor[cnt_i, cnt_j, :] = props
            cnt_j += 1
        cnt_i += 1
    return props_tensor
