import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.initializers import *

from sklearn.model_selection import train_test_split

import keras_tuner as kt

from importlib import reload
import sentinel_utils
import data_generator

reload(sentinel_utils)
reload(data_generator)

class BuildHyperModel(kt.HyperModel):
    def __init__(self, trials_dir, trial_metric):
        super().__init__()

        self.trials_dir = trials_dir
        self.trial_metric = trial_metric
        
    def get_callbacks(self):
        return [
            tf.keras.callbacks.TensorBoard(self.trials_dir.joinpath('board')),
            tf.keras.callbacks.EarlyStopping(
                monitor=self.trial_metric, patience=10, mode='max'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=self.trial_metric, factor=0.5, patience=4, min_lr=1e-7,
            ),
        ]

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

    def initialise_vars(self, hp):
        batch_choice = hp.Choice('batch_size', [8, 16, 32, 64, 128])
        use_weights = hp.Boolean('class_weight')
        
        training_years_choice = hp.Choice(
            'training_years', ['2017_2018_2019', '2017']
        )
        utils = sentinel_utils.SentinelUtils(min_occurrences=20000)
        self.selected_classes = utils.get_processed_labels()
        self.data_summary = utils.get_data_summary(
            self.selected_classes, training_years=training_years_choice
        )
        
    def build(self, hp):
        self.initialise_vars(hp)
        
        sentinel_10m_input = Input((100, 100, 2))
        sentinel_20m_input = Input((50, 50, 2))
        x = concatenate([sentinel_10m_input, UpSampling2D(2)(sentinel_20m_input)])

        pooling = {'max': MaxPooling2D, 'average': AveragePooling2D}
        pooling_choice = hp.Choice('pooling', list(pooling.keys()))
        
        filter_power = hp.Int('filters_power', min_value=2, max_value=5, step=1)
        kernel_size = hp.Choice('kernel_size', [3, 5])
        for filters_scale in [2**x for x in range(3, filter_power+1)]:
            x = Conv2D(
                filters=filters_scale,
                kernel_size=kernel_size,
                padding='same',
                activation='relu',
            )(x)
            x = pooling[pooling_choice](pool_size=2, strides=2, padding='same')(x)    
            x = BatchNormalization()(x)
        
        x = Flatten()(x)

        units_power = hp.Int(
            'units_power', min_value=2, max_value=5, step=1
        )
        for units_scale in reversed([2**x for x in range(4, filter_power+1)]):
            x = Dense(units_scale, activation='relu')(x)
            x = BatchNormalization()(x)

        x = Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1))(x)
            
        outputs = Dense(
            self.selected_classes.shape[1],
            bias_initializer=(
                tf.keras.initializers.Constant(self.data_summary['initial_bias']) 
                if hp.Boolean('output_bias') else None
            ),
            activation='sigmoid',   
        )(x)
        m = tf.keras.models.Model(
            inputs=[sentinel_10m_input, sentinel_20m_input], 
            outputs=outputs
        )
        m.compile(
            optimizer='adam',
            loss=hp.Choice('loss', ['binary_crossentropy', 'binary_focal_crossentropy']), 
            metrics=self.get_metrics()
        )
        return m

    def fit(self, hp, model, **kwargs):
        training_ids, test_ids = train_test_split(
            self.selected_classes.index, test_size=10000, random_state=42
        )
        gen_params = dict(
            selected_classes=self.selected_classes,
            data_summary=self.data_summary,
            years=hp.get('training_years'),
            batch_size=hp.get('batch_size'),
        )
        training_generator = data_generator.DataGenerator(
            training_ids, **gen_params)
        testing_generator = data_generator.DataGenerator(
            test_ids, **gen_params)
        
        return model.fit(
            x=training_generator,
            validation_data=testing_generator,
            class_weight=(self.data_summary['class_weights'] 
                          if hp.get('class_weight') else None),
            **kwargs,
        )