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
        X = np.empty((self.batch_size, 100, 100, 4))
        # sentinel_10m = np.zeros((self.batch_size, 100, 100, 2))
        # sentinel_20m = np.zeros((self.batch_size, 50, 50, 2))

        # band_groups = [
        #     [['B3', 'B8'], sentinel_10m],
        #     [['B6', 'B11'], sentinel_20m]
        # ]
        
        # for i, (bands, X_sentinel) in enumerate(band_groups):
        #     for year in self.years:
        #         dirname = self.shards_dir.joinpath(
        #             f'features_{"_".join(bands)}_{year}'
        #         )
        #         for ii, ID in enumerate(batch_IDs):
        #             data = np.load(dirname.joinpath(f'feature_{ID}.npy'))
        #             X_sentinel[ii, ...] += data
                    
        #     X_sentinel = self.normalise(
        #         X_sentinel/len(self.years), [i*len(bands), i*len(bands)+1], self.data_summary
        #     )

        dirname = self.shards_dir.joinpath(
            f'features_{self.year}'
        )
        for ii, ID in enumerate(batch_IDs):
            data = np.load(dirname.joinpath(f'feature_{ID}.npy'))
            X[ii, ...] = data
            
        bands = ['B3', 'B8', 'B6', 'B11']
        X = self.normalise(
            X, range(len(bands)), self.data_summary
        )
        
        y = self.selected_classes.loc[batch_IDs].to_numpy()
        return X, y

    def normalise(self, X, bands, data_summary):
        stats = {}
        for stat in ['mean', 'std']:
            stats[stat] = np.array(list(data_summary[stat].values()))
            
        normalised_X = (X - stats['mean'][bands])/stats['std'][bands]
        return normalised_X





