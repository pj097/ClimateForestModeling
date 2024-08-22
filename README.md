The findings of this study are summarized in the document report/main.pdf. Please note that the views and conclusions presented in this document reflect the opinions of the author and do not necessarily represent those of any other individual or organization.

The Python files provided serve as tools for the Jupyter notebook files, which are best used in conjunction with the report.

To get started, it is recommended to create a new Python environment. Please refer to the guidelines for Conda or Pip for instructions on setting up a new environment.

Begin with explore_euforest.ipynb, which explores the EU-Forest dataset (link to dataset) through interactive visualizations.

Use the download_data.ipynb notebook to download data from Google Earth Engine (Google Earth Engine Python API). The downloaded data can then be explored using visualise_features.ipynb.

For hyperparameter tuning, refer to hypertune.ipynb, which is configured via hyper_model_creator.py. If you prefer, you can skip the hyperparameter tuning and create the model directly using keras_model_creator.py, then train it in other notebooks such as climate.ipynb.

The final notebook, climate.ipynb, requires ERA5 data. Instructions for obtaining this data can be found in download_data.ipynb. This notebook performs correlations between tree genus classification results and climate variables and implements a narrow neural network to predict tree change maps based on climate change data.
