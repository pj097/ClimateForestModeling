import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

import keras_tuner as kt

from importlib import reload
import sentinel_utils
import data_generator

reload(sentinel_utils)
reload(data_generator)

class BuildHyperModel(kt.HyperModel):
    def __init__(self, trials_dir, trial_metric):
        self.trials_dir = trials_dir
        self.trial_metric = trial_metric
        self.training_years = '2017_2018_2019'

        utils = sentinel_utils.SentinelUtils(min_occurrences=20000)
        self.selected_classes = utils.get_processed_labels()
        self.data_summary = utils.get_data_summary(
            self.selected_classes, training_years=self.training_years
        )

    def get_callbacks(self):
        callbacks = [
            tf.keras.callbacks.TensorBoard(self.trials_dir.joinpath('board')),
            tf.keras.callbacks.EarlyStopping(
                monitor=self.trial_metric, patience=10, mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=self.trial_metric, factor=0.5, patience=8, min_lr=1e-6,
            ),
        ]
        return callbacks

    def get_metrics(self):
        prc = tf.keras.metrics.AUC(name='prc', curve='PR')
        f1_score = tf.keras.metrics.FBetaScore(
            average='weighted', beta=1.0, threshold=0.5, name=f'weightedf1score'
        )
        f2_score = tf.keras.metrics.FBetaScore(
            average='weighted', beta=2.0, threshold=0.5, name=f'weightedf2score'
        )
        metrics = [
            'accuracy', 'recall', 'precision', 'auc', prc, f1_score, f2_score
        ] 
        return metrics
        
    def residual_block(self, x, filters):
        r = BatchNormalization()(x)
        r = Conv2D(
            filters=filters, kernel_size=3, strides=2, padding='same',
            activation='relu', 
            kernel_regularizer='l1l2'
        )(r)
        r = BatchNormalization()(r)
        r = Conv2D(
            filters=filters, kernel_size=3, padding='same',
            activation='relu', 
            kernel_regularizer='l1l2'
        )(r)
        r = Conv2D(
            filters=1, kernel_size=1,
            activation='relu', 
            kernel_regularizer='l1l2'
        )(r)
        x = Conv2D(
            filters=filters, kernel_size=3, strides=2, padding='same',
            activation='relu', 
            kernel_regularizer='l1l2'
        )(x)
        
        return Add()([x, r])
        
    def build(self, hp):
        sentinel_10m_input = Input((100, 100, 2))
        sentinel_20m_input = Input((50, 50, 2))
        
        x0 = concatenate([sentinel_10m_input, UpSampling2D(2)(sentinel_20m_input)])
        x = BatchNormalization()(x0)

        filter_min = hp.Choice('filter_size_min', [128, 64, 32])
        filter_max = hp.Choice('filter_size_max', [1024, 512, 256])

        n_conv_layers = 0
        filter_current = hp.get('filter_size_min')
        while filter_current <= filter_max:
            x = self.residual_block(x, filter_current)
            x = SpatialDropout2D(0.1)(x)
            filter_current *= 2
            n_conv_layers += 1

        conv_layers = hp.Fixed('convolutional_layers', n_conv_layers)

        x = MaxPooling2D(
            pool_size=4, 
            padding='same'
        )(x)    
        x = BatchNormalization()(x)
        
        x = Flatten()(x)

        units_min = hp.Choice('units_min', [256, 128, 64])
        units_max = hp.Choice('units_max', [2048, 1024, 512])

        n_dense_layers = 0
        units_current = hp.get('units_max')
        while units_current >= units_min:
            x = Dense(
                units_current, 
                activation='relu',
                kernel_regularizer='l1l2'
            )(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            units_current //= 2
            n_dense_layers += 1

        dense_layers = hp.Fixed('dense_layers', n_dense_layers)
        
        outputs = Dense(
            self.selected_classes.shape[1],
            bias_initializer=tf.keras.initializers.Constant(self.data_summary['initial_bias']),
            activation='sigmoid',   
        )(x)
        m = tf.keras.models.Model(
            inputs=[sentinel_10m_input, sentinel_20m_input], 
            outputs=outputs
        )
        m.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy', 
            metrics=self.get_metrics()
        )
        return m

    def get_train_test(self):
        training_ids, test_ids = train_test_split(
            self.selected_classes.index, test_size=10000, random_state=42
        )
        gen_params = dict(
            selected_classes=self.selected_classes,
            data_summary=self.data_summary,
            years=self.training_years,
            batch_size=64,
        )
        training_generator = data_generator.DataGenerator(
            training_ids, **gen_params)
        testing_generator = data_generator.DataGenerator(
            test_ids, **gen_params)
        return training_generator, testing_generator

    def fit(self, hp, model, **kwargs):
        kwargs['callbacks'] += self.get_callbacks()
        training_generator, testing_generator = self.get_train_test()
        return model.fit(
            x=training_generator,
            validation_data=testing_generator,
            class_weight=self.data_summary['class_weights'],
            **kwargs,
        )