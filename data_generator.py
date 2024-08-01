from tensorflow.keras.utils import PyDataset
import numpy as np

class DataGenerator(PyDataset):
    'Generates data for Keras'
    def __init__(self, all_IDs, shuffle, year=2017, **kwargs):
        super().__init__()
        vars(self).update(kwargs)
        self.all_IDs = all_IDs
        self.shuffle = shuffle
        self.year = year
        
        self.use_multiprocessing = True
        self.workers = 4
        self.max_queue_size = 10
        self.on_epoch_end()

    def __len__(self):
        'Number of batches per epoch.'
        return int(np.floor(len(self.all_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Get one batch of data.'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_IDs = [self.all_IDs[k] for k in indexes]
        
        return self.data_generation(batch_IDs)

    def on_epoch_end(self):
        'Update and shuffle indexes after each epoch.'
        self.indexes = np.arange(len(self.all_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
                
    def data_generation(self, batch_IDs):
        'Generate batch.'
        
        sentinel_10m = np.empty((self.batch_size, 100, 100, 2))
        sentinel_20m = np.empty((self.batch_size, 50, 50, 2))

        bands_groups = [
            [['B3', 'B8'], sentinel_10m],
            [['B6', 'B11'], sentinel_20m]
        ]
        for bands, X_sentinel in band_groups:
            dirname = self.shards_dir.joinpath(
                f'features_{"_".join(bands)}_{start_date.year}'
            )
            for i, ID in enumerate(batch_IDs):
                data = np.load(dirname.joinpath(f'feature_{ID}.npy'))
                X_sentinel[i, ...] = self.normalise(data, sentinel_bands, self.data_summary)
        
        y = self.selected_classes.loc[batch_IDs].to_numpy()
        return tuple(sentinel_10m, sentinel_20m), y

    def normalise(self, X, bands, data_summary):
        stats = {}
        for stat in ['mean', 'std']:
            stats[stat] = np.array(list(data_summary[stat].values()))
            
        normalised_X = (X - stats['mean'][bands])/stats['std'][bands]
        return normalised_X





