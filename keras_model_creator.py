from IPython.display import display, HTML

import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *

import keras_tuner as kt

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from pathlib import Path

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
            # tf.keras.callbacks.BackupAndRestore(
            #     self.model_dir, save_freq='epoch', delete_checkpoint=False
            # ),
            # tf.keras.callbacks.CSVLogger(log_file, append=True),
            # tf.keras.callbacks.ModelCheckpoint(
            #     self.model_dir.joinpath('model.keras'), mode='max',
            #     monitor='val_recall', save_best_only=True,
            #     save_freq='epoch', initial_value_threshold=0.5,
            #     verbose=self.verbose,
            # ),
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

        hypermodel = self.BuildHyperModel(**dict(
            base_filters=self.base_filters,
            output_shape=self.selected_classes.shape[1], 
            metrics=metrics, 
            loss=self.loss,
            output_bias=self.data_summary['initial_bias']
        ))

        tuner = kt.Hyperband(
            hypermodel,
            objective=kt.Objective('val_recall', direction='max'),
            directory=Path('trials'),
            project_name='hyperband',
            overwrite=self.overwrite
        )
             
        tuner.search(
            x=training_generator,
            validation_data=testing_generator,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=self.data_summary['class_weights'],
            verbose=self.verbose
        )
        return tuner, testing_generator

    class BuildHyperModel(kt.HyperModel):
        def __init__(self, **kwargs):
            super().__init__()
            vars(self).update(kwargs)
            
        def build(self, hp):
            sentinel_10m_input = Input((100, 100, 2))
            sentinel_20m_input = Input((50, 50, 2))
            
            x = concatenate([sentinel_10m_input, UpSampling2D(2)(sentinel_20m_input)])
            filter_power = hp.Int(
                'filters_scale', min_value=0, max_value=7, step=1
            )
            for filters_scale in [2**x for x in range(filter_power+1)]:
                x = Conv2D(
                    filters=filters_scale,
                    kernel_size=hp.Int('kernel_size', min_value=3, max_value=7, step=2),
                    padding='same',
                    activation='relu',
                )(x)
                x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)    
                x = BatchNormalization()(x)
            
            x = Flatten()(x)

            units_power = hp.Int(
                'units_power', min_value=0, max_value=8, step=1
            )
            for units_scale in reversed([2**x for x in range(filter_power+1)]):
                x = Dense(units_scale, activation='relu')(x)
                x = BatchNormalization()(x)
    
            x = Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.8, step=0.1))(x)

            if self.output_bias:
                output_bias = tf.keras.initializers.Constant(self.output_bias)
            else:
                output_bias = None

            outputs = Dense(
                self.output_shape,
                bias_initializer=output_bias,
                activation='sigmoid',   
            )(x)
    
            m = tf.keras.models.Model(
                inputs=[sentinel_10m_input, sentinel_20m_input], 
                outputs=outputs
            )
    
            m.compile(
                optimizer=hp.Choice('optimizer', ['adam', 'sgd']), 
                loss=self.loss, metrics=self.metrics)
            
            return m



