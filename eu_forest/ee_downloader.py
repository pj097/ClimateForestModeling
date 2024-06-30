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

    def get_sentinel_image(self, bbox, start_date, end_date):
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
            .clip(bbox)
        )
        return image

    def create_elevation_inputs(self, shards_dir, geometries, bbox, start_date, end_date):
        chunks = self.create_chunks(geometries, chunk_size=100)
    
        elevations_dir = shards_dir.joinpath('elevations')
        elevations_dir.mkdir(exist_ok=True)

        # Check if this chunk's download has been completed
        exists = self.check_existing(chunk.shape[0], ith_chunk, chunk_size, elevations_dir)
        if exists:
            return

        elevation_inputs = []
        for ith_chunk, chunk in enumerate(chunks):
            elevation_inputs.append((ith_chunk, chunk, chunk_size, bbox, elevations_dir))
                
        return elevation_inputs

    def download_elevations(self, input_tuple):
        ith_chunk, chunk, chunk_size, bbox, elevations_dir = input_tuple
        params = {'fileFormat': 'NPY'}
        image = ee.ImageCollection('COPERNICUS/DEM/GLO30').select('DEM').max().clip(bbox)
        sleep(ith_chunk%10)
        
        for i, geometry in enumerate(pbar := tqdm(chunk, leave=False)):
            pbar.set_description(f'Downloading {ith_chunk}...')
            
            shard_id = ith_chunk*chunk_size + i
            
            shard_path = elevations_dir.joinpath(f'elevation_{shard_id}.npy')
    
            if shard_path.is_file():
                continue
    
            bbox = ee.Geometry.BBox(*geometry.bounds)
            params['expression'] = image.clipToBoundsAndScale(bbox, width=100, height=100)
            pixels = ee.data.computePixels(params)
            data = np.load(io.BytesIO(pixels)).astype(float)
            
            np.save(shard_path, data)

    def create_chunks(self, geometries, chunk_size=100):
        chunk_size = 100
        chunks = [geometries[i: i + chunk_size] for i in range(0, geometries.shape[0], chunk_size)]
        return chunks

    def create_sentinel_inputs(self, shards_dir, geometries, bbox, start_date, end_date):
        chunks = self.create_chunks(geometries, chunk_size=100)
    
        date_tag = f'{start_date.year}{str(start_date.month).zfill(2)}'
        features_dir = shards_dir.joinpath(f'features_{date_tag}')
        features_dir.mkdir(exist_ok=True)

        sentinel_inputs = []
        for ith_chunk, chunk in enumerate(chunks):
            sentinel_inputs.append((ith_chunk, chunk, chunk_size, bbox, start_date, end_date, features_dir))
                
        return sentinel_inputs

    def check_existing(self, chunk_fragments, ith_chunk, chunk_size, shards_dir):
        # Check if this chunk's download has been completed
        total_shards = 0
        for n in range(chunk_fragments):
            shard_id = ith_chunk*chunk_size + n
            s = shards_dir.joinpath(f'*_{shard_id}.npy')
            if s.is_file():
                total_shards += 1
        return total_shards == chunk_fragments:

    def download_sentinel_shards(self, input_tuple):
        ith_chunk, chunk, chunk_size, bbox, start_date, end_date, features_dir = input_tuple
        # Sleep time helps with parallel processing,
        # if you're brave enough to try it
        sleep_time = ith_chunk%10
        sleep(sleep_time)
    
        # Check if this chunk's download has been completed
        exists = self.check_existing(chunk.shape[0], ith_chunk, chunk_size, features_dir)
        if exists:
            return
    
        # Cloud masked, band selected, mean image of the bbox area. 
        sentinel_image = SentinelGetter().get_image(bbox, start_date, end_date)
            
        # For further options, see
        # https://developers.google.com/earth-engine/apidocs/ee-data-computepixels
        params = {'fileFormat': 'NPY'}
        
        all_data = []
        for geometry in (pbar := tqdm(chunk, leave=False)):
            pbar.set_description(f'Chunk {ith_chunk}')
            # Not ideal but a lot of connection errors can occur here.
            # They are (so far) not program ending, simply retry.
            retry = True
            while retry:
                try:
                    this_bbox = ee.Geometry.BBox(*geometry.bounds)
                    params['expression'] = sentinel_image.clipToBoundsAndScale(
                        this_bbox, width=100, height=100)
    
                    # There can be a delay before the URL becomes available,
                    # in which case the loop simply retries (seems rare so far).
                    pixels = ee.data.computePixels(params)
                    data = np.load(BytesIO(pixels))
    
                    # Numpy ndarray being appended to a list of ndarrays.
                    # Ensure all_data uses python's list instead of ndarray.tolist().
                    all_data.append(data)
                    retry = False
                    
                except Exception as e:
                    print(e)
                    # Sleep for 1 second if error, Google seems
                    # fine with 100/s requests.
                    sleep(sleep_time)
                    retry = True
                    
        raw_features = np.array(all_data)
        features = raw_features.view((float, len(raw_features.dtype.names)))
        features = features/10000
        
        for ii in range(chunk.shape[0]):
            shard_id = ith_chunk*chunk_size + ii
            np.save(
                features_dir.joinpath(f'feature_{shard_id}.npy'), 
                features[ii, ...]
            )

    

    