import json
from collections import OrderedDict
from pathlib import Path
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

class SentinelUtils:
    def __init__(
        self, tmp=Path('tmp'), min_occurrences=0,
        all_labels_path=Path('labels', 'full_dummies.csv')):
        self.tmp = tmp
        self.tmp.mkdir(exist_ok=True)
        self.min_occurrences = min_occurrences
        self.all_labels_path = all_labels_path
        sentinel_bands = [f'B{x}' for x in range(2, 9)] + ['B8A', 'B11', 'B12']
        soilgrids_band = ['bdod', 'cec', 'cfvo', 'clay', 'nitrogen', 'ocd',
                          'ocs', 'phh2o', 'sand', 'silt', 'soc']
        self.all_bands = sentinel_bands + ['Elevation'] + soilgrids_band

    def get_processed_labels(self, overwrite_existing=False):
        selected_classes_path = self.tmp.joinpath(f'selected_classes_{self.min_occurrences}.csv')
        
        if selected_classes_path.is_file() and not overwrite_existing:
            return pd.read_csv(selected_classes_path, index_col='selected_index')
        else:
            return self.process_labels(selected_classes_path)

    def get_data_summary(self, shards_dir, seasons, selected_classes,
                         sample_shards=40000, overwrite_existing=False):
        
        summary_path = self.tmp.joinpath(f'data_summary_{"_".join(seasons)}_{self.min_occurrences}.json')
        
        if summary_path.is_file() and not overwrite_existing:
            return json.loads(summary_path.read_text())
            
        sentinel_shards = []
        for s in seasons:
            path_list = list(shards_dir.joinpath(f'features_2017{s}').glob('feature_*.npy'))
            sentinel_shards.extend(path_list)
            
        sentinel_shards = shuffle(sentinel_shards, random_state=42)[:sample_shards]
        data_summary = self.process_features(self.all_bands, sentinel_shards)

        data_summary['initial_bias'] = self.calculate_initial_bias(selected_classes)
        data_summary['class_weights'] = self.calculate_class_weights(selected_classes)
        
        summary_path.write_text(json.dumps(data_summary))

        return data_summary
    
    def process_features(self, sentinel_shards):
        data_summary = {}
        for stat in ['std', 'mean']:
            data_summary[stat] = OrderedDict()

        shards_dir = sentinel_shards[0].parent.parent
        elevation_shards = list(shards_dir.joinpath('elevations').glob('elevation_*.npy'))
        elevation_shards = shuffle(elevation_shards, random_state=42)[:len(sentinel_shards)]
        soil_shards = list(shards_dir.joinpath('soil').glob('soil_*.npy'))
        soil_shards = shuffle(soil_shards, random_state=42)[:len(sentinel_shards)]

        for i, band in enumerate(tqdm(self.all_bands, leave=False)):
            band_data = []
            if i < 10:
                for shard_path in (pbar := tqdm(sentinel_shards, leave=False)):
                    pbar.set_description(band)
                    with open(shard_path, 'rb') as f:
                        data = np.load(f)
                        band_data.append(np.copy(data[..., i]))
            elif i == 10:
                for shard_path in (pbar := tqdm(elevation_shards, leave=False)):
                    pbar.set_description(f'Soil {band}')
                    with open(shard_path, 'rb') as f:
                        data = np.load(f)
                        band_data.append(np.copy(data))
            else:
                for shard_path in (pbar := tqdm(soil_shards, leave=False)):
                    pbar.set_description(band)
                    with open(shard_path, 'rb') as f:
                        data = np.load(f)
                        band_data.append(np.copy(data[..., i - 11]))

            band_data = np.vstack(band_data)
            data_summary['mean'][band] = band_data.mean()
            data_summary['std'][band] = band_data.std()
        
        return data_summary

    def calculate_initial_bias(self, selected_labels):
        neg, pos = np.bincount(selected_labels.to_numpy().astype(int).flatten())
        initial_bias = np.log([pos/neg])
        return initial_bias[0]

    def calculate_class_weights(self, selected_labels):
        class_weights = selected_labels.shape[0]/(selected_labels.shape[1]*selected_labels.sum(axis=0))
        class_indices = list(range(selected_labels.shape[1]))
        class_weights = dict(zip(class_indices, class_weights.tolist()))
        return class_weights

    def process_labels(self, selected_classes_path):
        all_labels = pd.read_csv(self.all_labels_path)
        
        grouped_labels = all_labels.T.groupby(
            all_labels.columns.str.split().str[0],
        ).max().T

        keep_classes = np.where(grouped_labels.sum(axis=0) >= self.min_occurrences)[0]
        selected_labels = grouped_labels.iloc[:, keep_classes]

        keep_shards = np.where(selected_labels.sum(axis=1) > 0)[0]
        
        selected_labels = selected_labels.iloc[keep_shards, :]

        selected_labels.to_csv(selected_classes_path, index_label='selected_index')

        return selected_labels

    

    