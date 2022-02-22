import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import xarray as xr
import rioxarray as rxr
import earthpy as et
import earthpy.plot as ep
from shapely import geometry
from PIL import Image, ImageDraw
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import copy
import shapefile

def coord2pix_new(ds, coordinates_list): 
 #coordinates_list is a list with coordinates in a world system (for example? EPSG(4326)) 
    geo_trans = ds.GetGeoTransform()
    x = np.array([coord[0] for coord in coordinates_list])
    y = np.array([coord[1] for coord in coordinates_list])
    ul_x = geo_trans[0]
    ul_y = geo_trans[3]
    x_dist = geo_trans[1]
    y_dist = geo_trans[5]
    if len(x) > 0: 
        pixel = ((-x * geo_trans[5] + y * geo_trans[2] + geo_trans[0] * geo_trans[5] - geo_trans[2] * geo_trans[3])/(geo_trans[2] * geo_trans[4] - geo_trans[1] * geo_trans[5])).astype(int) 
        line = ((x * geo_trans[4] - y * geo_trans[1] + geo_trans[1] * geo_trans[3] - geo_trans[0] * geo_trans[4])/(geo_trans[2] * geo_trans[4] - geo_trans[1] * geo_trans[5])).astype(int)
        pixel[pixel < 0] = 0
        line[line < 0] = 0
        pixel[pixel > 1793] = 1793
        line[line > 1694] = 1694
        output = list(zip(pixel, line)) 
    else: 
        output = [] 
    return output 

def coord2pix(coordinates_list, geo_trans, projection_ref, to_round_result=True): 
   #input has a latitude-longitude format
    spatial_reference = osr.SpatialReference() 
    spatial_reference.ImportFromWkt(projection_ref) 
    crsGeo = osr.SpatialReference() 
    crsGeo.ImportFromEPSG(4326) 
    transformer_in_shot_world_coordinates = osr.CoordinateTransformation(crsGeo, spatial_reference) 
    coords = transformer_in_shot_world_coordinates.TransformPoints(coordinates_list) 
    if len(coords) > 0: 
        x_coords = np.array(coords)[:, 0] 
        y_coords = np.array(coords)[:, 1] 
 
        x_pix = (-x_coords * geo_trans[5] + y_coords * geo_trans[2] + geo_trans[0] * geo_trans[5] - geo_trans[2] * geo_trans[3]) / (geo_trans[2] * geo_trans[4] - geo_trans[1] * geo_trans[5]) 
        y_pix = (x_coords * geo_trans[4] - y_coords * geo_trans[1] + geo_trans[1] * geo_trans[3] - geo_trans[0] * geo_trans[4]) /  (geo_trans[2] * geo_trans[4] - geo_trans[1] * geo_trans[5]) 
        if to_round_result: 
            x_pix = np.round(x_pix) 
            y_pix = np.round(y_pix) 
        output = list(zip(x_pix, y_pix)) 
    else: 
        output = [] 
    return output 
  
  def pix2coord(pix_coordinates_list, geo_trans, projection_ref): 
    srs = osr.SpatialReference() 
    srs.ImportFromWkt(projection_ref) 
    ct = osr.CoordinateTransformation(srs, srs.CloneGeogCS()) 
    if len(pix_coordinates_list) > 0: 
        x = np.array(pix_coordinates_list)[:, 0] 
        y = np.array(pix_coordinates_list)[:, 1] 
        lon_list = x * geo_trans[1] + geo_trans[0] 
        lat_list = y * geo_trans[5] + geo_trans[3] 
        lon_final_list = [] 
        lat_final_list = [] 
        for i in range(0, len(lon_list)): 
            (lon, lat, holder) = ct.TransformPoint(lon_list[i], lat_list[i]) 
            lon_final_list.append(lon) 
            lat_final_list.append(lat) 
        output = list(zip(lon_final_list, lat_final_list)) 
    else: 
        output = [] 
    return output
  
  def to_clip_shot(image, polygon, geo_trans, projection_ref):
    # новая геопривязка для вырезанного снимка
    new_geo_trans = list(copy.deepcopy(geo_trans))
    new_image = copy.deepcopy(image)#[min_y_pix: max_y_pix, min_x_pix: max_x_pix]) this line was edited by me
    new_geo_trans = tuple(new_geo_trans)
    new_border_points = coord2pix(polygon.points, new_geo_trans, projection_ref)
    new_x_shot_size = len(new_image[0, 0])
    new_y_shot_size = len(new_image[0])
    mask_im = Image.new('L', (new_x_shot_size, new_y_shot_size), 0)
    ImageDraw.Draw(mask_im).polygon(new_border_points, outline=1, fill=1)
    mask = np.array(mask_im)
    for i in range(len(new_image)):
        new_image[i] = new_image[i] * mask
    return new_image, new_geo_trans
