from IPython.display import display, HTML

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam, SGD, AdamW
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.initializers import glorot_uniform

from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

import data_generator
from importlib import reload
reload(data_generator)
from data_generator import DataGenerator

class KerasModelCreator:
    def __init__(self, **kwargs):
        vars(self).update(kwargs)
        self.kwargs = kwargs
            
    def display_logger(self, log_file, metrics):
        metric_names = [m if isinstance(m, str) else m.name for m in metrics]
        if log_file.is_file() and log_file.stat().st_size > 0:
            val_metrics = ['val_loss'] + ['val_' + x for x in metric_names]
            df = pd.read_csv(log_file)[['epoch', 'loss'] + metric_names + val_metrics]
            df['epoch'] += 1
            df = df.astype(str)
            df.iloc[df.shape[0] - 1 ] = df.columns
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
                self.model_dir.joinpath('model.keras'), 
                monitor='val_recall', save_best_only=True, 
                save_freq='epoch', initial_value_threshold=0.1,
                verbose=0,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_recall', factor=0.9, patience=2, min_lr=3e-6,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_recall', 
                verbose=1,
                patience=10,
                mode='max',
                restore_best_weights=True
            ),
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
            for f in self.model_dir.glob('*'):
                f.unlink(missing_ok=True)
                
        self.display_logger(log_file, metrics)
        
        callbacks = self.define_callbacks_and_logger(log_file, metrics)
        
        training_ids, test_ids = train_test_split(self.shard_ids, test_size=10000, random_state=42)
        validation_ids, test_ids = train_test_split(test_ids, test_size=5000, random_state=42)

        training_generator = DataGenerator(training_ids, shuffle=True, **self.kwargs)
        testing_generator = DataGenerator(test_ids, shuffle=False, **self.kwargs)
        validation_generator = DataGenerator(validation_ids, shuffle=False, **self.kwargs)
    
        if self.model_dir.joinpath('model.keras').is_file():
            print('Loading model...')
            model = load_model(self.model_dir.joinpath('model.keras'))
        else:
            print('Building model...')
            model = self.build_model(
                self.utils.selected_classes.shape[1], (*self.dim, len(self.bands)), metrics,
                self.architecture, self.loss,
                output_bias=self.utils.data_summary['initial_bias'],
            )
        print('Fitting...')
        model.fit(
            x=training_generator,
            validation_data=validation_generator,
            epochs=self.epochs,
            callbacks=callbacks,
            class_weight=self.utils.data_summary['class_weights']
        )
        return model, testing_generator

    def build_vgg(self, input_layer):
        x = input_layer
        for filters in [self.base_filters, self.base_filters*2]:
            for _ in range(2):
                x = Conv2D(
                    filters=filters, kernel_size=3, padding='same', activation='relu',
                )(x)
            x = MaxPooling2D(pool_size=2,strides=2)(x)
            x = BatchNormalization()(x)

        for filters in [self.base_filters*4, self.base_filters*8]:
            for _ in range(3):
                x = Conv2D(
                    filters=filters, kernel_size=3, padding='same', activation='relu',
                )(x)
            x = MaxPooling2D(pool_size=2,strides=2)(x)
            x = BatchNormalization()(x)

        x = Flatten()(x)

        for n_layers in range(2):
            x = Dense(self.base_filters*64, activation='relu')(x)
            x = Dropout(0.5)(x)
        
        return x
    
    def res_block(self, x, filters):
        r = BatchNormalization()(x)
        r = Activation('relu')(r)
        r = Conv2D(
            filters=filters, kernel_size=3, strides=2, padding='same', 
            kernel_initializer=glorot_uniform(seed=42)
        )(r)
        
        r = BatchNormalization()(r)
        r = Activation('relu')(r)
        
        r = Conv2D(
            filters=filters, kernel_size=3, strides=1, padding='same', 
            kernel_initializer=glorot_uniform(seed=42)
        )(r)
        
        r = Conv2D(
            filters=1, kernel_size=1, strides=1, padding='valid'
        )(r)
        
        x = Conv2D(
            filters=filters, kernel_size=3, strides=2, padding='same', 
            kernel_initializer=glorot_uniform(seed=42)
        )(x)
        
        return Add()([x, r])

    def build_simple(self, input_layer):
        x = Conv2D(
            filters=self.base_filters, kernel_size=3, padding='same', activation='relu',
        )(input_layer)
        x = MaxPooling2D(pool_size=2, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout)(x)

        
        x = Flatten()(x)
        
        x = Dense(self.base_filters*4, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout*2)(x)

        x = Dense(self.base_filters*2, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout*2)(x)
        return x

    def build_resnet(self, input_layer):
        x = self.res_block(input_layer, self.base_filters)
        x = self.res_block(x, self.base_filters*2)
        x = self.res_block(x, self.base_filters*4)
        x = self.res_block(x, self.base_filters*8)
        
        x = Activation('relu')(x)
        x = Flatten()(x)

        x = Dense(self.base_filters*8, activation='relu')(x)

        return x

    def topless_resnet50(self, input_shape, input_layer):
        resnet = tf.keras.applications.ResNet50(
            include_top=False,
            weights=None,
            input_shape=input_shape,
            pooling=None
        )
        x = resnet(input_layer)
        x = Flatten()(x)
        x = Dense(self.base_filters*8, activation='relu')(x)
        return x

    def build_model(self, output_shape, input_shape, metrics, architecture, loss, output_bias=None):
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
         
        input_layer = Input(input_shape)

        match architecture.lower():
            case 'simple':
                x = self.build_simple(input_layer)
            case 'resnet50':
                x = self.topless_resnet50(input_shape, input_layer)
            case 'resnet':
                x = self.build_resnet(input_layer)
            case 'vgg':
                x = self.build_vgg(input_layer)
        
        outputs = Dense(output_shape, activation='sigmoid', bias_initializer=output_bias)(x)
        
        m = Model(inputs=input_layer, outputs=outputs)

        adam = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
        )

        m.compile(optimizer=adam, loss=loss, metrics=metrics)
        
        return m