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
        batch_choice = hp.Choice('batch_size', [8, 16, 32, 64, 128], default=64)
        use_weights = hp.Boolean('class_weight', default=True)
        
        training_years_choice = hp.Choice(
            'training_years', ['2017_2018_2019', '2017'], default='2017'
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
        
        filter_power = hp.Int('filters_power', min_value=4, max_value=7, step=1)
        kernel_regularizer = hp.Choice('kernel_regularizer', ['l1', 'l2', 'l1l2'])
        
        for filters_scale in [2**x for x in range(3, filter_power+1)]:
            x = Conv2D(
                filters=filters_scale,
                kernel_size=hp.Choice('kernel_size', [3, 5]),
                padding='same',
                activation=hp.Choice('activation', ['relu', 'leaky_relu']),
                kernel_regularizer=kernel_regularizer,
            )(x)
            x = MaxPooling2D(
                pool_size=hp.Choice('pool_size', [2, 4]), padding='same'
            )(x)    
            x = BatchNormalization()(x)

        x = SpatialDropout2D(hp.Float('spatial_dropout', min_value=0.0, max_value=0.2, step=0.1))(x)
        
        x = Flatten()(x)

        units_power = hp.Int('units_power', min_value=4, max_value=8, step=1)
        for units_scale in reversed([2**x for x in range(4, filter_power+1)]):
            x = Dense(
                units_scale, activation=hp.get('activation'),
                kernel_regularizer=kernel_regularizer
            )(x)
            x = BatchNormalization()(x)

        x = Dropout(hp.Float('dropout', min_value=0.1, max_value=0.3, step=0.1))(x)
            
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
        initial_lr = hp.Choice('dropout_rate', [1e-4, 1e-3], default=1e-4)
        m.compile(
            optimizer=Adam(learning_rate=initial_lr),
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