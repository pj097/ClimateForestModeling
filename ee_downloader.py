import ee
import geemap
from tqdm.notebook import tqdm
import numpy as np
from time import sleep
from io import BytesIO
import geopandas as gpd
import pandas as pd

ee.Authenticate()
ee.Initialize(project='sentinel-treeclassification')

class EEDownloader:
    def mask_s2_clouds(self, image):
      # Quality assessment with resolution in meters
      qa = image.select('QA60')
      # Bits 10 and 11 are clouds and cirrus, respectively.
      cloud_bit_mask = 1 << 10
      cirrus_bit_mask = 1 << 11
      # Both flags should be set to zero, indicating clear conditions.
      mask = (
          qa.bitwiseAnd(cloud_bit_mask)
          .eq(0)
          .And(qa.bitwiseAnd(cirrus_bit_mask).eq(0))
      )
      return image.updateMask(mask)

    def get_sentinel_image(self, start_date, end_date, selected_bands):
        image = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_date, end_date)
            # Pre-filter to get less cloudy granules.
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .map(self.mask_s2_clouds)
            .select(selected_bands)
            .median()
        )
        return image

    def download_soil(self, ith_chunk, chunk, chunk_size, soil_dir):
        # Check if it has already been downloaded
        if soil_dir.joinpath(f'elevation_{ith_chunk*chunk_size + chunk_size - 1}.npy').is_file():
            return

        sleep(ith_chunk%10)
            
        soilgrids_band = ['bdod', 'cec', 'cfvo', 'clay', 'nitrogen', 'ocd', 'ocs', 'phh2o', 'sand', 'silt', 'soc']
        soilgrids_band = [f'{a}_mean' for a in soilgrids_band]
        params = {'fileFormat': 'NPY'}

        for i, geometry in enumerate(tqdm(chunk, leave=False)):
            band_data = []
            for band in soilgrids_band:
                soilgrids = ee.Image(f'projects/soilgrids-isric/{band}').reduce(ee.Reducer.median())
                bbox = ee.Geometry.BBox(*geometry.bounds)
                params['expression'] = soilgrids.clipToBoundsAndScale(
                    bbox, width=4, height=4)
                pixels = ee.data.computePixels(params)
                data = np.load(BytesIO(pixels)).astype(float)
                band_data.append(data)

            shard_id = ith_chunk*chunk_size + i
            shard_path = soil_dir.joinpath(f'elevation_{shard_id}.npy')

            np.save(shard_path, np.stack(band_data, axis=-1))
                

    def download_elevations(self, ith_chunk, chunk, chunk_size, elevations_dir):
        # Check if it has already been downloaded
        if features_dir.joinpath(f'elevation_{ith_chunk*chunk_size + chunk_size - 1}.npy').is_file():
            return
            
        params = {'fileFormat': 'NPY'}
        image = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').max()
        sleep(ith_chunk%10)
        
        for i, geometry in enumerate(pbar := tqdm(chunk, leave=False)):
            pbar.set_description(f'Downloading {ith_chunk}...')
            
            shard_id = ith_chunk*chunk_size + i
            
            shard_path = elevations_dir.joinpath(f'elevation_{shard_id}.npy')
    
            bbox = ee.Geometry.BBox(*geometry.bounds)
            params['expression'] = image.clipToBoundsAndScale(bbox, width=100, height=100)
            pixels = ee.data.computePixels(params)
            data = np.load(BytesIO(pixels)).astype(float)
            
            np.save(shard_path, data)

    def download_sentinel_shards(
        self, ith_chunk, chunk, chunk_size, 
        features_dir, start_date, end_date, pixel_size, selected_bands):

        sleep_time = ith_chunk%10
        sleep(sleep_time)
    
        sentinel_image = self.get_sentinel_image(start_date, end_date, selected_bands)
            
        # For further options, see
        # https://developers.google.com/earth-engine/apidocs/ee-data-computepixels
        params = {'fileFormat': 'NPY'}
        
        for ii, geometry in enumerate((pbar := tqdm(chunk, leave=False))):
            pbar.set_description(f'Chunk {ith_chunk}')
            
            shard_id = ith_chunk*chunk_size + ii
            shard_path = features_dir.joinpath(f'feature_{shard_id}.npy')
            if shard_path.is_file():
                continue

            this_bbox = ee.Geometry.BBox(*geometry.bounds)
            params['expression'] = sentinel_image.clipToBoundsAndScale(
                this_bbox, width=pixel_size, height=pixel_size)
    
            pixels = ee.data.computePixels(params)
            data = np.load(BytesIO(pixels))
            
            np.save(
                features_dir.joinpath(f'feature_{shard_id}.npy'), 
                data.view((float, len(data.dtype.names)))/10000
            )
            
    def download_era5(self, gdf, start_date, end_date, weather_dir):
        # Only interested in one point per sample (ERA5 is 10km resolution)
        gdf_points = gpd.GeoDataFrame(geometry=gdf.geometry.centroid)
        # Break up the dataframe, otherwise the request is too big
        chunk_size = 10000
        chunks = [gdf_points[i: i + chunk_size] for i in range(0, gdf_points.shape[0], chunk_size)]
        chunks = [geemap.geopandas_to_ee(c) for c in chunks]
        
        start = start_date
        for year in tqdm(range(start_date.year, end_date.year)):
            save_file = weather_dir.joinpath(f'era5_{year}.csv')
            # Check if it has already been downloaded
            if save_file.is_file():
                continue
                
            dfs = []
            for point_collection in tqdm(chunks, leave=False):
                era5_bands = ['temperature_2m', 'total_precipitation_sum']
                era5_image = (
                    ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
                    .select(era5_bands)
                    .filterBounds(point_collection)
                    .filterDate(start.replace(year=year), start.replace(year=year+1))
                    .mean()
                )
                # Convert the image to a feature collection
                means_collection = era5_image.reduceRegions(
                    point_collection, ee.Reducer.mean().forEachBand(era5_image)
                )
                # Create the download and load the URL as a dataframe
                params = {
                    'table': means_collection, 'format': 'CSV',
                    'filename': f'era5_{year}.csv',
                    'selectors': era5_bands
                }
                tableid = ee.data.getTableDownloadId(params)
                url = ee.data.makeTableDownloadUrl(tableid)
                dfs.append(pd.read_csv(url))
            # Finally, concatenate the chunks back into the original shape
            pd.concat(dfs, ignore_index=True).to_csv(save_file, index=False)



