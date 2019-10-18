"""
a tool which extracts NRT Soil Moisture Maps from the mhm model as GIF image,
reprojects them, georeference them, assigns soil moisture values to color code
and saves result as geopandas dataframe/shapefile
Can be used as template for a general framework of georefencing raster data
from images with Python
"""
__author__ = "Erik Nixdorf"
__propertyof__ = "Helmholtz-Zentrum fuer Umweltforschung GmbH - UFZ. "
__email__ = "erik.nixdorf@ufz.de"
__version__ = "0.1"

#%% We start by retrieving the gif from ftp
import urllib.request
import numpy as np
#image things
from PIL import Image
import os
#some functions from radohydro
from radohydro import buffered_raster_clipping, create_footprint_cells
from scipy import ndimage
import datetime
import time
from itertools import product
from osgeo import gdal, osr


#%% predefined functions
def warp_with_gcps(input_path,
                   output_path,
                   gcps,
                   gcp_epsg=3301,
                   output_epsg=3301):
    """
    from https://bit.ly/2OYoTNz
    """
    # Open the source dataset and add GCPs to it
    src_ds = gdal.OpenShared(str(input_path), gdal.GA_ReadOnly)
    gcp_srs = osr.SpatialReference()
    gcp_srs.ImportFromEPSG(gcp_epsg)
    gcp_crs_wkt = gcp_srs.ExportToWkt()
    src_ds.SetGCPs(gcps, gcp_crs_wkt)

    # Define target SRS
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(output_epsg)
    dst_wkt = dst_srs.ExportToWkt()

    error_threshold = 0.125  # error threshold --> use same value as in gdalwarp
    resampling = gdal.GRA_Bilinear

    # Call AutoCreateWarpedVRT() to fetch default values for target raster dimensions and geotransform
    tmp_ds = gdal.AutoCreateWarpedVRT(
        src_ds,
        None,  # src_wkt : left to default value --> will use the one from source
        dst_wkt,
        resampling,
        error_threshold)
    dst_xsize = tmp_ds.RasterXSize
    dst_ysize = tmp_ds.RasterYSize
    dst_gt = tmp_ds.GetGeoTransform()
    tmp_ds = None

    # Now create the true target dataset
    dst_path = output_path
    dst_ds = gdal.GetDriverByName('GTiff').Create(
        dst_path, dst_xsize, dst_ysize, src_ds.RasterCount)
    dst_ds.SetProjection(dst_wkt)
    dst_ds.SetGeoTransform(dst_gt)
    dst_ds.GetRasterBand(1).SetNoDataValue(0)

    # And run the reprojection
    gdal.ReprojectImage(
        src_ds,
        dst_ds,
        None,  # src_wkt : left to default value --> will use the one from source
        None,  # dst_wkt : left to default value --> will use the one from destination
        resampling,
        0,  # WarpMemoryLimit : left to default value
        error_threshold,
        None,  # Progress callback : could be left to None or unspecified for silent progress
        None)  # Progress callback user data
    dst_ds = None


def ffill(arr, mask, mode='nd_max', footprint_size=3):
    """
    different functions to fill holes in 2D array, position of holes is identified by ~mask values
    """
    if mode == 'acc':  # row or column based max value is used for filling holes
        idx = np.where(~mask, np.arange(mask.shape[1]), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        out = arr[np.arange(idx.shape[0])[:, None], idx]
    if mode == 'nn':  #interpolate 1D by next neighbor by https://bit.ly/2QGvbQo
        arr[mask] = np.interp(
            np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
        out = arr
    if mode == 'nd_max':  # inspired by @ https://bit.ly/31q9Miz
        arr_mask = arr.copy()
        arr_mask = arr_mask * mask
        arr_mask = arr_mask.astype(float)
        arr_mask[arr_mask == 0] = np.nan
        mask_int = np.ones((footprint_size, footprint_size))
        mask_int[1, 1] = 0
        arr_int = ndimage.generic_filter(
            arr_mask,
            np.nanmax,
            footprint=mask_int,
            mode='constant',
            cval=np.nan)
        #replace only the indices where arr is nan by the interpolated ones
        arr[~mask] = arr_int[~mask]
        out = arr.astype(int)
        out[out < 0] = 0
    return out


def daterange(start_date, end_date):    
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def main(fname='nFK_0_25_daily_n14.gif',
         georeferencing=False,
         crs='epsg:4326',
         boundary_shp='.\geo_bounds\\DEU_adm0.shp',
         Output=True):
    """
    The main function which downloads the gif and processes
    """

    #%% First some standard defintions of the SM Files
    # Define the colorcode_sm
    color_code_RGB = np.array(
        ([230, 0, 0], [255, 170, 0], [252, 211, 127], [242, 242, 242],
         [230, 230, 230], [217, 217, 217], [189, 235, 191], [90, 204, 95],
         [8, 168, 30], [5, 101, 120], [0, 0, 255]))
    color_code_grey = np.array((106, 0, 157, 253, 248, 240, 210, 126, 118, 75,
                                165))
    #define corresponding bandmean of soil moisture
    sm_code = np.array((5, 15, 25, 35, 45, 55, 65, 75, 85, 95, 97.5))
    #define the image_transform. If dont know it, do georeferencing first
    # In case of north up images, the GT(2) and GT(4) coefficients are zero,
    # and the GT(1) is pixel width, and GT(5) is pixel height. The (GT(0),GT(3))
    #  position is the top left corner of the top left pixel of raster.
    #for WGS84
    image_transform = ((4.756996182800601, 0.016483987942747476, 0.0,
                        55.10523733895226, 0.0, -0.010363875929350508))
    #the Resolution of the mhm modell
    mhm_res = (175, 225)
    #%% Download the file from `url` and save it locally under `file_name`:
    #connect to server
    con = urllib.request.urlopen('https://files.ufz.de/~drought/' + fname)
    #get last modification date and add 6 hours to be sure to be on same date
    time_struct = datetime.datetime.strptime(
    con.headers['last-modified'],
    '%a, %d %b %Y %H:%M:%S %Z') + datetime.timedelta(hours=6)
                                             
    obs_enddate = datetime.date(time_struct.year, time_struct.month,
                                time_struct.day-1)
    obs_startdate = datetime.date(time_struct.year, time_struct.month,
                                  time_struct.day - 14)

    # open the image directly from URL
    im = Image.open(
        urllib.request.urlopen('https://files.ufz.de/~drought/' + fname))

    #Next we do georeferencing if necessary
    if georeferencing == True:
        #save the first frame of the image
        im.save('for_georef.png', format='png')
        #the gcp points
        groundpnts_x = np.array((6, 10, 14))
        groundpnts_y = np.array((54, 52, 50, 48))
        imageind_x = np.array((75, 317, 560))
        imageind_y = np.array((115, 316, 508, 693))
        # GCP input
        xyz = np.zeros((12, 3))
        xyz[:, :-1] = np.array(list(product(groundpnts_x, groundpnts_y)))
        row_col = np.array(list(product(imageind_x, imageind_y))).astype(float)
        gcp_array = np.hstack((xyz, row_col))
        #make a list of the gcps
        gcps = [
            gdal.GCP(gcp_array[i, 0], gcp_array[i, 1], gcp_array[i, 2],
                     gcp_array[i, 3], gcp_array[i, 4])
            for i in range(0, gcp_array.shape[0])
        ]
        #the actual warping
        warp_with_gcps(
            'for_georef.png',
            'test_ref.tif',
            gcps,
            gcp_epsg=4326,
            output_epsg=4326)
        # Write out final one
        time.sleep(2)
        gdal.Translate(
            'test_ref_resize.tif',
            'test_ref.tif',
            format='GTiff',
            width=Image.open('test.png').size[0],
            height=Image.open('test.png').size[1])
        #we open the rasterfile in order to get the geotransform
        # open dataset
        ds = gdal.Open('test_ref_resize.tif')
        image_transform = ds.GetGeoTransform()
    #small correction of image transform from visual inspection
    image_transform = np.array(image_transform)
    image_transform[3] = image_transform[3] + 0.1
    image_transform = tuple(image_transform)
    #%% Create our future geodataframerearrange the transformation cellwidth

    pix_clip_data, pix_clip_transform, cols, rows = buffered_raster_clipping(
        np.array(im),
        boundary_shp,
        raster_transfrm=image_transform,
        raster_proj=crs,
        buffrcllsz=0)
    cellwidth = pix_clip_transform[1] * pix_clip_data.shape[1] / mhm_res[0]
    cellheight = pix_clip_transform[5] * pix_clip_data.shape[0] / mhm_res[1]
    pix_resized_transform = (pix_clip_transform[0], cellwidth,
                             pix_clip_transform[2], pix_clip_transform[3],
                             pix_clip_transform[4], cellheight)
    #we create polygoncells for each pixel
    pixcells = create_footprint_cells(
        transform=pix_resized_transform,
        data_size=(mhm_res[1], mhm_res[0]),
        proj_crs=crs)

    # start our main loop trough the gif file
    nframes = 0
    for single_date in daterange(obs_startdate, obs_enddate):
        print('Extract mhm nrt soil moisture at date', single_date.strftime("%Y-%m-%d"))
        im_rgb = im.convert('RGB')
        pix_rgb = np.array(im_rgb)
        pix_rgb_clip = pix_rgb[rows[1]:rows[0], cols[0]:cols[1], :]
        im_clip = Image.fromarray(pix_rgb_clip)
        #%% do the initial processing
        # reduce size of image to mhm size
        im_resized = im_clip.resize(mhm_res)
        # find relevant colors in RGB mode
        im_rgb = im_resized.convert('RGB')
        pix_rgb = np.array(im_rgb)
        #we create the mask by combine logical ands
        mask = np.full(pix_rgb.shape[:2], False)
        for rgb_color in color_code_RGB:
            color_mask = np.logical_and(
                np.logical_and(pix_rgb[:, :, 0] == rgb_color[0],
                               pix_rgb[:, :, 1] == rgb_color[1]),
                pix_rgb[:, :, 2] == rgb_color[2])
            mask = np.logical_or(mask, color_mask)
        # Now we do the part which needs to be done for all files
        # get pixels from grey_scale picture and fill nans
        pix_filled = pix_rgb.copy()
        #for i in range(0,3):
        #    pix_filled[:,:,i]=ffill(pix_filled[:,:,i],mask,mode='nd_max',footprint_size=3)
        # an array which is used to calculated the distance to each color from the map
        pix_dist = np.ones((pix_filled.shape[0], pix_filled.shape[1],
                            len(color_code_RGB))) * 256
        for i in range(0, len(color_code_RGB)):
            pix_dist[:, :, i] = np.abs(
                pix_filled[:, :, 0] - color_code_RGB[i, 0]) + np.abs(
                    pix_filled[:, :, 1] - color_code_RGB[i, 1]) + np.abs(
                        pix_filled[:, :, 2] - color_code_RGB[i, 2])
        #get the color which is closest by
        nearest_colors = np.argmin(pix_dist, axis=2, out=None)
        #assign the sm values
        pix_sm = nearest_colors.copy()
        for i in range(0, len(color_code_grey)):
            pix_sm[pix_sm == i] = sm_code[i]
        #for i in range(0,3):
        pix_sm = ffill(pix_sm, mask, mode='nd_max', footprint_size=3)
        #add as column to cell_array
        pixcells[single_date.strftime("%Y-%m-%d")] = pix_sm.flatten(order='F')
        # go further in the gif file

        nframes += 1
        try:
            im.seek(nframes)
        except EOFError:
            break

    #Intersect with Germany Polygon
    #germany_gdf=gpd.GeoDataFrame.from_file(boundary_shp)
    #mhm_grid=gpd.overlay(pixcells, germany_gdf, how='intersection')

    if Output:
        print('write output as shapefile')
        try:
            os.mkdir('output')
        except OSError:
            pass

        pixcells.to_file('.\output\\'+fname[:-4])

if __name__ == "__main__":
    main()
