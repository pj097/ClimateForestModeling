from IPython.display import display, HTML

import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow.keras.layers import *

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import data_generator
from importlib import reload
reload(data_generator)
from data_generator import DataGenerator

def brier_score(y_true, y_pred):
    brier_score = tf.reduce_mean((y_pred - y_true) ** 2, axis=1)
    return brier_score

class KerasModelCreator:
    def __init__(self, **kwargs):
        vars(self).update(kwargs)
        self.kwargs = kwargs
            
    def display_logger(self, log_file, metrics):
        metric_names = [m if isinstance(m, str) else m.name for m in metrics]
        
        if not self.print_log:
            return
            
        if log_file.is_file() and log_file.stat().st_size > 0:
            sort_key = lambda x: x.split('_')[-1]
            val_metrics = ['val_loss'] + ['val_' + x for x in metric_names]
            all_metrics = ['loss'] + metric_names + val_metrics
            df = pd.read_csv(log_file)[['epoch'] + sorted(all_metrics, key=sort_key)]
            df['epoch'] += 1
            df = df.astype(str)
            df.loc[df.shape[0]] = df.columns
            print('Previous training:')
            display(HTML(df.to_html(index=False)))
    
    def define_callbacks_and_logger(self, log_file, metrics):
        metric_names = [m if isinstance(m, str) else m.name for m in metrics]

        callbacks = [
            tf.keras.callbacks.BackupAndRestore(
                self.model_dir, save_freq='epoch', delete_checkpoint=False
            ),
            tf.keras.callbacks.CSVLogger(log_file, append=True),
            tf.keras.callbacks.ModelCheckpoint(
                self.model_dir.joinpath('model.keras'), mode='max',
                monitor='val_recall', save_best_only=True,
                save_freq='epoch', initial_value_threshold=0.5,
                verbose=self.verbose,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_recall', factor=0.5, patience=4, min_lr=1e-7,
                verbose=self.verbose,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_recall', 
                verbose=self.verbose,
                patience=20,
                mode='max',
                restore_best_weights=True
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=self.model_dir.joinpath('tensorboard_logs'),
                histogram_freq=0,
                write_graph=True,
                update_freq='epoch',
                embeddings_freq=0,
            )
        ]
        return callbacks

    def get_metrics(self):
        prc = tf.keras.metrics.AUC(name='prc', curve='PR')

        f1_scores = []
        for average in ['micro', 'macro', 'weighted']:
            f1_scores.append(
                tf.keras.metrics.F1Score(
                    average=average, threshold=0.5, name=f'{average}f1score')
            )
        metrics = [
            'accuracy', 'recall', 'precision', 'auc', prc
        ] + f1_scores

        return metrics
        
    def run(self):
        ''' 
        If not overwrite and there's an existing model, the model will 
        continue training if the given epoch is bigger than the previous,
        else just evaluate.
        Ensure train splits are the same across continuations / evaluations
        by not modifying the random_state in split_and_normalise.
        '''
        log_file = self.model_dir.joinpath('model.log')
        
        metrics = self.get_metrics()

        if self.overwrite:
            for f in self.model_dir.rglob('*'):
                if f.is_file(): f.unlink()
                
        self.display_logger(log_file, metrics)
        
        callbacks = self.define_callbacks_and_logger(log_file, metrics)

        shard_ids = self.selected_classes.index
        training_ids, test_ids = train_test_split(shard_ids, test_size=10000, random_state=42)

        training_generator = DataGenerator(training_ids, shuffle=True, **self.kwargs)
        testing_generator = DataGenerator(test_ids, shuffle=False, **self.kwargs)
        
        model = self.build_model(
            self.selected_classes.shape[1], metrics, self.loss,
            output_bias=self.data_summary['initial_bias'],
        )
        model.fit(
            x=training_generator,
            validation_data=testing_generator,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=self.data_summary['class_weights'],
            verbose=self.verbose
        )
        return model, testing_generator

    def soil_layers(self, x):
        for filters_scale in [2, 4, 8, 16]:
            x = Conv2D(
                filters=self.base_filters*filters_scale, 
                kernel_size=3, padding='same', activation='relu',
            )(x)
            x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout)(x)
        
        x = Flatten()(x)
        
        x = Dense(self.base_filters*8, activation='relu', name='soil')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout*2)(x)
        return x

    def elevation_layers(self, x):
        for filters_scale in [2, 4, 8, 16]:
            x = Conv2D(
                filters=self.base_filters*filters_scale,
                kernel_size=3, padding='same', activation='relu',
            )(x)
            x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout)(x)
        
        x = Flatten()(x)
        
        x = Dense(self.base_filters*4, activation='relu', name='elevation')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout*2)(x)
        return x

    def sentinel_layers(self, x):
        for filters_scale in [2, 4, 8, 16]:
            x = Conv3D(
                filters=self.base_filters*filters_scale, 
                kernel_size=3, padding='same',
                activation='relu',
            )(x)
            x = MaxPooling3D(pool_size=2, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = Dropout(self.dropout)(x)

        x = Conv3D(
            filters=self.base_filters*32, 
            kernel_size=3, padding='same',
            activation='relu',
        )(x)
            
        x = Flatten()(x)

        x = Dense(self.base_filters*8, activation='relu', name='sentinel')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)
        return x

    def build_model(self, output_shape, metrics, loss, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        sentinel_input = Input((
            len(self.seasons), 100, 100, 
            len([b for b in self.band_indices if b < 10])
        ))
       
        x = self.sentinel_layers(sentinel_input)
        
        inputs = [sentinel_input]
        
        if 10 in self.band_indices:
            elevation_input = Input((100, 100, 1))
            inputs += [elevation_input]
            x = concatenate([x, self.elevation_layers(elevation_input)])

        n_soil_bands = len([b for b in self.band_indices if b > 10])
        if n_soil_bands:
            soil_input = Input((4, 4, n_soil_bands))
            inputs += [soil_input]
            x = concatenate([x, self.soil_layers(soil_input)])

        for units_scale in [32, 16, 8, 4]:
            x = Dense(self.base_filters*units_scale, activation='relu')(x)
            x = Dropout(self.dropout)(x)
        
        outputs = Dense(output_shape, activation='sigmoid', bias_initializer=output_bias)(x)

        m = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        opt = tf.keras.optimizers.Adam(
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.999,
        )

        m.compile(optimizer=opt, loss=loss, metrics=metrics)
        
        return m


