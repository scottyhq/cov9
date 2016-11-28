"""
General utility functions to accompany vmodels (mogi, yang, okada...)
Created on Tue Jun 21 14:59:15 2016

@author: scott
"""
import numpy as np
import rasterio


def world2rc(x,y,affine, inverse=False):
    '''
    World coordinates (lon,lat) to image (row,col) center pixel coordinates
    '''
    #T0 = src.meta['affine']
    T0 = affine
    T1 = T0 * rasterio.Affine.translation(0.5, 0.5)
    rc2xy = lambda r, c: (c, r) * T1
    # can probable simpligy,,, also int() acts like floor()
    xy2rc = lambda x, y: [int(i) for i in [x, y] * ~T1][::-1]

    if inverse:
        return rc2xy(y,x)
    else:
        return xy2rc(x,y)


def save_rasterio(path, data, profile):
    '''
    save single band raster file
    intended to use with load_rasterio() to open georeferenced data manipulate
    with numpy and then resave modified data
    '''
    with rasterio.drivers():
        with rasterio.open(path, 'w', **profile) as dst:
            dst.write(data, 1) #single band


def load_rasterio(path):
    '''
    load single bad georeference data as 'f4', convert NoDATA to np.nan
    not sure this works with 'scale' in vrt
    '''
    with rasterio.drivers():
        with rasterio.open(path, 'r') as src:
            data = src.read()
            meta = src.profile
            extent = src.bounds[::2] + src.bounds[1::2]

    return data, extent, meta


def load_cor_mask(path='phsig.cor.8alks_8rlks.geo.vrt', corthresh=0.1):
    '''
    load geocoded correlation file to use as mask
    '''
    cordata, extent, meta = load_rasterio(path)
    cor = cordata[0]
    # Geocoding seems to create outliers beyond realistic range (maybe interpolation)
    ind_outliers = (np.abs(cor) > 1)
    cor[ind_outliers] = 0.0
    # Can go further and remove pixels with low coherence (or just set to 0.0)
    mask = (cor < corthresh)
    #data[mask] = np.nan

    return mask


def get_cart2los(incidence,heading):
    '''
    coefficients for projecting cartesian displacements into LOS vector
    '''
    incidence = np.deg2rad(incidence)
    heading = np.deg2rad(heading)

    EW2los = np.sin(heading) * np.sin(incidence)
    NS2los = np.cos(heading) * np.sin(incidence)
    Z2los = -np.cos(incidence)

    cart2los = np.dstack([EW2los, NS2los, Z2los])

    return cart2los


def cart2pol(x1,x2):
    #theta = np.arctan(x2/x1)
    theta = np.arctan2(x2,x1) #sign matters -SH
    r = np.hypot(x2,x1)
    return theta, r


def pol2cart(theta,r):
    x1 = r * np.cos(theta)
    x2 = r * np.sin(theta)
    return x1,x2
