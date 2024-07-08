from tensorflow.keras.utils import PyDataset
import numpy as np

class DataGenerator(PyDataset):
    'Generates data for Keras'
    def __init__(self, all_IDs, shuffle, **kwargs):
        super().__init__()
        vars(self).update(kwargs)
        self.all_IDs = all_IDs
        self.shuffle = shuffle
        
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
        sentinel_bands = self.bands[:10]
        
        soil_bands = self.bands[11:]
        
        X_sentinel = np.empty((self.batch_size, len(self.seasons), 100, 100, len(sentinel_bands)))

        X_elevation = np.empty((self.batch_size, 100, 100))

        X_soil = np.empty((self.batch_size, 4, 4, len(soil_bands)))

        y = self.utils.selected_classes.loc[batch_IDs].to_numpy()

        for i, ID in enumerate(batch_IDs):
            for t, s in enumerate(self.seasons):
                sentinel_data = np.load(self.shards_dir.joinpath(
                    f'features_{self.year}{s}').joinpath(f'feature_{ID}.npy'))
                
                X_sentinel[i, t, ...] = sentinel_data[..., sentinel_bands]

            X_elevation[i,...] = np.load(self.shards_dir.joinpath(
                'elevations').joinpath(f'elevation_{ID}.npy'))

            X_soil[i,...] = np.load(self.shards_dir.joinpath(
                'soil').joinpath(f'soil_{ID}.npy'))

        
        X_sentinel = self.utils.normalise(X_sentinel, sentinel_bands)
        X_elevation = self.utils.normalise(X_elevation, 10)
        X_soil = self.utils.normalise(X_soil, soil_bands)
        
        return (X_sentinel, X_elevation, X_soil), y





