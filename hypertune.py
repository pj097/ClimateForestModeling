#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

from pathlib import Path

import keras_tuner as kt

from importlib import reload
import hyper_model_creator
import keras_model_creator
import sentinel_utils


# In[2]:


trials_dir = Path('trials', 'hyperband_resnet')
trial_metric = 'val_weightedf2score'


# In[3]:


callbacks = [
    tf.keras.callbacks.TensorBoard(trials_dir.joinpath('board')),
    tf.keras.callbacks.EarlyStopping(
        monitor=trial_metric, patience=20, mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor=trial_metric, factor=0.5, patience=8, min_lr=1e-6,
    ),
]


# In[ ]:


reload(hyper_model_creator)

hypermodel = hyper_model_creator.BuildHyperModel()

tuner = kt.Hyperband(
    hypermodel,
    objective=kt.Objective(trial_metric, direction='max'),
    directory=trials_dir.parent,
    project_name=trials_dir.name,
    hyperband_iterations=3,
    max_epochs=100,
    overwrite=False,
)
# tuner.search_space_summary()
tuner.search(
    verbose=0, 
    callbacks=callbacks
)


# In[ ]:


# tuner.results_summary(num_trials=2)


# In[ ]:


# %load_ext tensorboard
# tb_path = str(trials_dir.joinpath('board'))
# %tensorboard --logdir "$tb_path"


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




