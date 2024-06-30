import ee
from tqdm.notebook import tqdm
import numpy as np
from time import sleep
from io import BytesIO

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

    def get_sentinel_image(self, start_date, end_date):
        selected_bands = [f'B{x}' for x in range(2, 9)] + ['B8A', 'B11', 'B12']
        image = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_date, end_date)
            .filter(ee.Filter.calendarRange(start_date.month, end_date.month, 'month'))
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

    def download_sentinel_shards(self, ith_chunk, chunk, chunk_size, features_dir, start_date, end_date):
        # Check if it has already been downloaded
        if features_dir.joinpath(f'feature_{ith_chunk*chunk_size + chunk_size - 1}.npy').is_file():
            return
            
        # Sleep time helps with parallel processing,
        # if you're brave enough to try it
        sleep_time = ith_chunk%10
        sleep(sleep_time)
    
        # Cloud masked, band selected, mean image of the bbox area. 
        sentinel_image = self.get_sentinel_image(start_date, end_date)
            
        # For further options, see
        # https://developers.google.com/earth-engine/apidocs/ee-data-computepixels
        params = {'fileFormat': 'NPY'}
        
        all_data = []
        for geometry in (pbar := tqdm(chunk, leave=False)):
            pbar.set_description(f'Chunk {ith_chunk}')

            this_bbox = ee.Geometry.BBox(*geometry.bounds)
            params['expression'] = sentinel_image.clipToBoundsAndScale(
                this_bbox, width=100, height=100)

            pixels = ee.data.computePixels(params)
            data = np.load(BytesIO(pixels))

            # Numpy ndarray being appended to a list of ndarrays.
            # Ensure all_data uses python's list instead of ndarray.tolist().
            all_data.append(data)
                    
        raw_features = np.array(all_data)
        features = raw_features.view((float, len(raw_features.dtype.names)))
        features = features/10000
        
        for ii in range(chunk.shape[0]):
            shard_id = ith_chunk*chunk_size + ii
            np.save(
                features_dir.joinpath(f'feature_{shard_id}.npy'), 
                features[ii, ...]
            )