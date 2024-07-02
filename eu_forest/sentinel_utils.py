import json
from collections import OrderedDict
from pathlib import Path
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from concurrent.futures import ThreadPoolExecutor

class SentinelUtils:
    def __init__(self, all_labels, all_bands, sentinel_shards,
                 min_occurrences=0, overwrite_existing=False):
        tmp = Path('tmp')
        tmp.mkdir(exist_ok=True)
        
        summary_path = tmp.joinpath('data_summary.json')
        selected_classes_path = tmp.joinpath('selected_classes.csv')
        
        if summary_path.is_file() and not overwrite_existing:
            self.data_summary = json.loads(summary_path.read_text())
            self.selected_classes = pd.read_csv(selected_classes_path, index_col='selected_index')
        else:
            data_summary = self.process_features(all_bands, sentinel_shards)
            
            self.selected_classes, self.data_summary = self.process_labels(
                all_labels, min_occurrences, data_summary, summary_path, selected_classes_path
            )

    def process_features(self, all_bands, sentinel_shards):
        print('Calculating feature statistics...')
        data_summary = {}
        for stat in ['std', 'mean']:
            data_summary[stat] = OrderedDict()

        shards_dir = sentinel_shards[0].parent.parent
        elevation_shards = list(shards_dir.joinpath('elevations').glob('elevation_*.npy'))
        elevation_shards = shuffle(elevation_shards, random_state=42)[:len(sentinel_shards)]
        soil_shards = list(shards_dir.joinpath('soil').glob('soil_*.npy'))

        for i, band in enumerate(tqdm(all_bands)):
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

    def process_labels(self, all_labels, min_occurrences, data_summary, 
                       summary_path, selected_classes_path):
        keep_classes = np.where(all_labels.sum(axis=0) >= min_occurrences)[0]
        selected_labels = all_labels.iloc[:, keep_classes]
        keep_shards = np.where(selected_labels.sum(axis=1) > 0)[0]
        
        selected_labels = selected_labels.iloc[keep_shards, :]

        neg, pos = np.bincount(selected_labels.to_numpy().astype(int).flatten())
        initial_bias = np.log([pos/neg])
        data_summary['initial_bias'] = initial_bias[0]
        

        class_weights = selected_labels.shape[0]/(selected_labels.shape[1]*selected_labels.sum(axis=0))
        class_indices = list(range(selected_labels.shape[1]))
        class_weights = dict(zip(class_indices, class_weights.tolist()))
        data_summary['class_weights'] = class_weights

        selected_labels.to_csv(selected_classes_path, index_label='selected_index')
        summary_path.write_text(json.dumps(data_summary))
        
        print(f'Dropped {all_labels.shape[1] - selected_labels.shape[1]} columns, '
              f'{all_labels.shape[0] - selected_labels.shape[0]} rows')
        
        return selected_labels, data_summary


    def normalise(self, X, bands):
        stats = {}
        for stat in ['mean', 'std']:
            stats[stat] = np.array(list(self.data_summary[stat].values()))
        return (X - stats['mean'][bands])/stats['std'][bands]
    

    