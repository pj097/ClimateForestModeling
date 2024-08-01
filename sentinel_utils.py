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
        self.sentinel_band_groups = [['B3', 'B8'], ['B6', 'B11']]

    def get_processed_labels(self, overwrite_existing=False):
        selected_classes_path = self.tmp.joinpath(f'selected_classes_{self.min_occurrences}.csv')
        
        if selected_classes_path.is_file() and not overwrite_existing:
            return pd.read_csv(selected_classes_path, index_col='selected_index')
        else:
            return self.process_labels(selected_classes_path)

    def get_data_summary(self, shards_dir, selected_classes,
                         n_samples=40000, overwrite_existing=False):
        
        summary_path = self.tmp.joinpath(f'data_summary_{self.min_occurrences}.json')
        
        if summary_path.is_file() and not overwrite_existing:
            return json.loads(summary_path.read_text())

        data_summary = self.process_features(shards_dir, n_samples)

        data_summary['initial_bias'] = self.calculate_initial_bias(selected_classes)
        data_summary['class_weights'] = self.calculate_class_weights(selected_classes)
        
        summary_path.write_text(json.dumps(data_summary))

        return data_summary
    
    def process_features(self, shards_dir, n_samples):
        data_summary = {}
        for stat in ['std', 'mean']:
            data_summary[stat] = OrderedDict()

        # for band_group in tqdm(self.sentinel_band_groups):
            # dirname = shards_dir.joinpath(
            #     f'features_{"_".join(band_group)}_2017'
            # )
        dirname = shards_dir.joinpath('features_2017')
        shard_list = list(dirname.glob('feature_*.npy'))
        shard_list = shuffle(shard_list, random_state=42)[:n_samples]

        for i, band in enumerate(tqdm(['B3', 'B8', 'B6', 'B11'], leave=False)):
            band_data = []
            for shard in (pbar := tqdm(shard_list, leave=False)):
                pbar.set_description(band)
                data = np.load(shard)
                band_data.append(np.copy(data[..., i]))

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

    

    