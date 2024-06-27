import tensorflow as tf
import numpy as np

class DataGenerator(tf.keras.utils.PyDataset):
    'Generates data for Keras'
    def __init__(self, list_IDs, shuffle, **kwargs):
        super().__init__()
        vars(self).update(kwargs)
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        
        self.use_multiprocessing = True
        self.workers = 4
        self.max_queue_size = 10
        self.on_epoch_end()

    def __len__(self):
        'Number of batches per epoch.'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Get one batch of data.'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Update and shuffle indexes after each epoch.'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_temp):
        'Generate batch.'
        X = np.empty((self.batch_size, *self.dim, len(self.bands)))
        y = np.empty((self.batch_size, len(self.keep_classes)))

        for i, ID in enumerate(list_IDs_temp):
            X[i,...] = np.load(self.shards_dir.joinpath(
                f'features_{self.data_tag}').joinpath(f'feature_{ID}.npy'))[..., self.bands]
            y[i] = np.load(
                self.shards_dir.joinpath(f'labels_{self.data_tag}').joinpath(f'label_{ID}.npy')
            )[self.keep_classes]
            
        return X, y