#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
import numpy as np

import geemap
import geemap.colormaps as cm
import ee

import datetime
from pathlib import Path
from time import sleep
import io
from tqdm.notebook import tqdm

from sklearn.utils import shuffle

from multiprocessing import Pool


# In[2]:


data_path = Path.home().joinpath('data')


# In[3]:


ee.Authenticate()
ee.Initialize(project='sentinel-treeclassification')


# Data source:
# 
# https://figshare.com/collections/A_high-resolution_pan-European_tree_occurrence_dataset/3288407

# In[4]:


df = pd.read_csv('EUForestspecies.csv')


# In[5]:


df.shape


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df[['X', 'Y']].value_counts()


# In[9]:


df[['X', 'Y']].value_counts().hist(bins=50)


# In[10]:


df[['X', 'Y']].value_counts().shape


# In[11]:


df['SPECIES NAME'].unique().shape


# In[12]:


df['SPECIES NAME'].value_counts()


# In[13]:


df['SPECIES NAME'].value_counts()[df['SPECIES NAME'].value_counts()>=100].hist(bins=20)


# In[14]:


df['SPECIES NAME'].value_counts()[df['SPECIES NAME'].value_counts()<100].hist(bins=20)


# In[15]:


grouped = df[['X', 'Y', 'SPECIES NAME']].groupby(['X', 'Y'], as_index=False).agg({'SPECIES NAME': ', '.join})


# In[16]:


grouped.to_csv('plots.csv' ,index=False)


# In[17]:


grouped.shape


# In[18]:


grouped.head()


# In[19]:


gdf = gpd.GeoDataFrame(
    grouped.drop(labels=['X', 'Y'], axis=1), 
    geometry=gpd.points_from_xy(x=grouped.X, y=grouped.Y, crs='EPSG:3035')
)
gdf.geometry = gdf.buffer(500, cap_style=3).to_crs(epsg=4326)


# In[20]:


gdf.head()


# In[21]:


# Add some padding to avoid border polygons being cut off.
bbox = ee.Geometry.BBox(*(gdf.geometry.total_bounds + 0.01))


# In[22]:


class SentinelGetter:
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

    def get_image(self, bbox, start_date, end_date):
        selected_bands = [f'B{x}' for x in range(2, 9)] + ['B8A', 'B11', 'B12', 'TCI_R', 'TCI_G', 'TCI_B']
        image = (
            ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterDate(start_date, end_date)
            .filter(ee.Filter.calendarRange(6, 9, 'month'))
            # Pre-filter to get less cloudy granules.
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
            .map(self.mask_s2_clouds)
            .select(selected_bands)
            .median()
            .clip(bbox)
        )
        return image


# In[23]:


def visualise(gdf, bbox):
    rgb_bands = ['B4', 'B3', 'B2']
    visualization = {
        'min': 1,
        'max': 3000,
        'bands': rgb_bands
    }
    
    start_date = datetime.datetime(2017, 3, 1)
    end_date = datetime.datetime(2020, 3, 1)
    
    image = SentinelGetter().get_image(bbox, start_date, end_date)
    
    m = geemap.geemap.Map()
    
    center = (np.array(bbox.getInfo()['coordinates'][0][2]) + np.array(bbox.getInfo()['coordinates'][0][0]))/2
    
    m.set_center(*center, 7)
    
    style = {"stroke": True, "color": "green",
             "weight": 2, "opacity": 1, "fillOpacity": 0.1
    }
    
    m.add_gdf(gdf, layer_name='euforest', style=style)
    
    m.addLayer(image, visualization, 'RGB')
    
    earth_url = 'https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}'
    m.add_tile_layer(earth_url, name='Google Earth', attribution='Google')
    
    m.addLayerControl(position='topright')
    return m
# visualise(gdf.sample(5000), bbox)


# Although it wouldn't matter for training purposes, the centering of each cell matters for comparisons with, e.g TreeSatAI. I expected all EU Forest data to be centered at midway points of 1 km cells, so all coordinates would end with 500 instead of 000. As shown below, that's not the case entirely.

# In[27]:


uncentered_mask = (grouped['X'] % 1000 != 500) | (grouped['Y'] % 1000 != 500)
uncentered_mask.sum()


# In[28]:


# visualise(gdf[uncentered_mask], bbox)


# In[ ]:


def download_npy(bbox, start_date, end_date, gdf, i, save_path):
    # Sleep time helps with parallel processing,
    # if you're brave enough to try it
    sleep_time = i*2
    sleep(sleep_time)

    # Cloud masked, band selected, mean image of the bbox area. 
    sentinel_image = SentinelGetter().get_image(bbox, start_date, end_date)
        
    # For further options, see
    # https://developers.google.com/earth-engine/apidocs/ee-data-computepixels
    params = {'fileFormat': 'NPY'}

    print(f'Downloading part {i}', flush=True)
    all_data = []
    
    # Progress bar, tracks continuations
    for i, row in tqdm(gdf.iterrows(), total=gdf.shape[0]):
        # Not ideal but a lot of connection errors can occur here.
        # They are (so far) not program ending, simply retry.
        retry = True
        while retry:
            try:
                this_bbox = ee.Geometry.BBox(*row.geometry.bounds)
                params['expression'] = sentinel_image.clipToBoundsAndScale(
                    this_bbox, width=100, height=100)

                # There can be a delay before the URL becomes available,
                # in which case the loop simply retries (seems rare so far).
                pixels = ee.data.computePixels(params)
                data = np.load(io.BytesIO(pixels))

                # Numpy ndarray being appended to a list of ndarrays.
                # Ensure all_data uses python's list instead of ndarray.tolist().
                all_data.append(data)
                retry = False

            except Exception as e:
                # Sleep for 1 second if error, Google seems
                # fine with 100/s requests.
                sleep(sleep_time)
                retry = True
                
    save_data = np.array(all_data)
    with open(save_path, 'wb') as f:
        np.save(f, save_data)


# In[25]:


dummy_labels = gdf['SPECIES NAME'].str.get_dummies(sep=', ').astype(float)
gdf = dummy_labels.join(gdf['geometry'])


# In[ ]:


raw_data_path = Path('/sentinel_data').joinpath('raw_data')
start_date = datetime.datetime(2017, 1, 1)
end_date = datetime.datetime(2020, 1, 1)

def run_jobs(this_data_path, this_gdf, this_bbox, start_date, end_date):
    chunk_size = 1000
    chunks = [this_gdf[i: i + chunk_size] for i in range(0, this_gdf.shape[0], chunk_size)]
    map_inputs = []

    for i, chunk in enumerate(chunks):
        save_path = this_data_path.joinpath(f"features_{i}.npy")
        chunk.drop('geometry', axis=1).to_csv(
            this_data_path.joinpath(f'labels_{i}.csv'), index=False)
        if save_path.is_file():
            continue
        map_inputs.append((this_bbox, start_date, end_date, chunk, i, save_path))
        
    pool = Pool(processes=8)
    pool.starmap(download_npy, map_inputs, chunksize=1)
    pool.close()
    pool.join()
    
# run_jobs(raw_data_path, gdf, bbox, start_date, end_date)


# In[ ]:





# In[ ]:





# In[ ]:


# import subprocess
# subprocess.run(['sudo', 'shutdown', 'now'])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




