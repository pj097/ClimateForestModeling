import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.PyDataset):
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
        X, y = self.data_generation(batch_IDs)
        return X, y

    def on_epoch_end(self):
        'Update and shuffle indexes after each epoch.'
        self.indexes = np.arange(len(self.all_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_IDs):
        'Generate batch.'
        X = np.empty((self.batch_size, *self.dim, len(self.bands)))

        y = self.utils.selected_classes.loc[batch_IDs].to_numpy()

        for i, ID in enumerate(batch_IDs):
            sentinel_data = np.load(self.shards_dir.joinpath(
                f'features_{self.data_tag}').joinpath(f'feature_{ID}.npy'))
            sentinel_data = self.utils.normalise_sentinel(sentinel_data, self.normal_type)

            if 10 in self.bands:
                elevation_data = np.load(self.shards_dir.joinpath(
                    'elevations').joinpath(f'elevation_{ID}.npy'))
                elevation_data = np.expand_dims(elevation_data/5000, axis=-1)
                X[i,...] = np.concatenate([sentinel_data, elevation_data], axis=-1)[..., self.bands]
            else:
                X[i,...] = sentinel_data[..., self.bands]
 
        return X, y





